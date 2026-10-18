// k6 load test for POST /search with a mixed cache-warm / cache-cold workload.
//
//   docker compose --profile loadtest run --rm loadtest      (or: make loadtest)
//
// Workload: setup() waits for /ready, then downloads a few seed images through the API
// (GET /configs -> sample_images, or K6_IMAGE_IDS), so it needs nothing on disk. Every
// iteration uploads one of them:
//   * warm  - the original bytes: after the first request the embedding comes from Redis
//   * cold  - the same image with random bytes appended after the JPEG end marker: the
//             picture decodes identically but its SHA-256 differs, so the embedding cache
//             misses and the backbone runs (a realistic "new query" cost)
// Each request is tagged {type: warm|cold}; thresholds apply per tag and globally.
// A JSON summary is written to $RESULTS_DIR/summary-<timestamp>.json and summary-latest.json.
import http from "k6/http";
import { check, fail, sleep } from "k6";
import { Counter, Trend } from "k6/metrics";
import exec from "k6/execution";

const API = __ENV.API_URL || "http://localhost:8000";
const MAX_VUS = parseInt(__ENV.K6_MAX_VUS || "10", 10);
const SCALE = parseFloat(__ENV.K6_DURATION_SCALE || "1");
const COLD_RATIO = parseFloat(__ENV.K6_COLD_RATIO || "0.3");
const BACKBONE = __ENV.K6_BACKBONE || "dinov2_vits14";
const INDEX = __ENV.K6_INDEX || "hnsw";
const RERANK = __ENV.K6_RERANK || "none";
const K = __ENV.K6_K || "20";
const P95_WARM = parseFloat(__ENV.K6_P95_WARM_MS || "1500");
const P95_COLD = parseFloat(__ENV.K6_P95_COLD_MS || "4000");
const MAX_ERR = parseFloat(__ENV.K6_MAX_ERROR_RATE || "0.01");
const RESULTS_DIR = __ENV.RESULTS_DIR || "results/loadtest";
const N_IMAGES = parseInt(__ENV.K6_N_IMAGES || "4", 10);

const dur = (s) => `${Math.max(1, Math.round(s * SCALE))}s`;

export const options = {
  scenarios: {
    ramp: {
      executor: "ramping-vus",
      startVUs: 1,
      stages: [
        { duration: dur(30), target: Math.max(1, Math.round(MAX_VUS / 4)) },
        { duration: dur(60), target: Math.max(1, Math.round(MAX_VUS / 2)) },
        { duration: dur(60), target: MAX_VUS },
        { duration: dur(60), target: MAX_VUS },
        { duration: dur(20), target: 0 },
      ],
      gracefulRampDown: "30s",
    },
  },
  setupTimeout: "30m",
  thresholds: {
    "http_req_failed{scenario:ramp}": [`rate<${MAX_ERR}`],
    "http_req_duration{type:warm}": [`p(95)<${P95_WARM}`],
    "http_req_duration{type:cold}": [`p(95)<${P95_COLD}`],
    checks: [`rate>${1 - MAX_ERR}`],
  },
  summaryTrendStats: ["min", "med", "avg", "p(90)", "p(95)", "p(99)", "max"],
};

const serverTotal = new Trend("server_total_ms", true);
const serverExtract = new Trend("server_extract_ms", true);
const serverSearch = new Trend("server_search_ms", true);
const serverRerank = new Trend("server_rerank_ms", true);
const cacheHits = new Counter("embedding_cache_hits");
const cacheMisses = new Counter("embedding_cache_misses");

function waitReady(timeoutS) {
  const deadline = Date.now() + timeoutS * 1000;
  while (Date.now() < deadline) {
    const res = http.get(`${API}/ready`, { tags: { name: "ready" } });
    if (res.status === 200) return;
    let stage = "";
    try {
      const body = res.json();
      stage = `${body.stage}: ${body.message}`;
      if (body.stage === "error") fail(`API warm-up failed: ${body.error}`);
    } catch (_e) {
      stage = `HTTP ${res.status}`;
    }
    console.log(`waiting for /ready (${stage})`);
    sleep(10);
  }
  fail("API did not become ready in time");
}

export function setup() {
  waitReady(1800);
  // seed queries: K6_IMAGE_IDS (comma-separated global ids) or the API's sample_images
  let ids = (__ENV.K6_IMAGE_IDS || "").split(",").filter((s) => s.length > 0);
  if (ids.length === 0) {
    const cfg = http.get(`${API}/configs`, { tags: { name: "bootstrap" } });
    if (cfg.status !== 200) fail(`GET /configs failed with HTTP ${cfg.status}`);
    ids = cfg.json("sample_images") || [];
  }
  const images = [];
  for (const id of ids.slice(0, N_IMAGES)) {
    const res = http.get(`${API}/images/${id}`, { responseType: "binary", tags: { name: "bootstrap" } });
    if (res.status === 200) images.push({ id, bytes: Array.from(new Uint8Array(res.body)) });
  }
  if (images.length === 0) fail("could not download any seed image from the API");
  console.log(`seed images: ${images.map((i) => i.id).join(", ")}`);
  return { images };
}

function coldVariant(bytes) {
  const out = new Uint8Array(bytes.length + 16);
  out.set(bytes);
  for (let i = 0; i < 16; i += 1) out[bytes.length + i] = Math.floor(Math.random() * 256);
  return out.buffer;
}

export default function (data) {
  const img = data.images[exec.scenario.iterationInTest % data.images.length];
  const cold = Math.random() < COLD_RATIO;
  const body = cold ? coldVariant(img.bytes) : new Uint8Array(img.bytes).buffer;
  const payload = {
    image: http.file(body, "query.jpg", "image/jpeg"),
    backbone: BACKBONE,
    index: INDEX,
    rerank: RERANK,
    k: K,
  };
  const res = http.post(`${API}/search`, payload, { tags: { type: cold ? "cold" : "warm", name: "search" }, timeout: "120s" });
  const ok = check(res, {
    "status is 200": (r) => r.status === 200,
    "has results": (r) => r.status === 200 && r.json("results").length > 0,
  });
  if (ok) {
    const lat = res.json("latency_ms");
    serverTotal.add(lat.total, { type: cold ? "cold" : "warm" });
    serverExtract.add(lat.extract, { type: cold ? "cold" : "warm" });
    serverSearch.add(lat.search);
    serverRerank.add(lat.rerank);
    if (res.json("cache") === "hit") cacheHits.add(1);
    else cacheMisses.add(1);
  }
  sleep(0.1);
}

export function handleSummary(data) {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const meta = {
    api: API,
    max_vus: MAX_VUS,
    duration_scale: SCALE,
    cold_ratio: COLD_RATIO,
    backbone: BACKBONE,
    index: INDEX,
    rerank: RERANK,
    k: K,
    thresholds: { p95_warm_ms: P95_WARM, p95_cold_ms: P95_COLD, max_error_rate: MAX_ERR },
  };
  const json = JSON.stringify({ meta, ...data }, null, 2);
  const lines = [
    `k6 load test ${stamp}`,
    `config: ${JSON.stringify(meta)}`,
    ...Object.entries(data.metrics)
      .filter(([name]) => /http_req_duration|http_req_failed|server_|embedding_cache|iterations|checks/.test(name))
      .map(([name, m]) => `${name}: ${JSON.stringify(m.values)}${m.thresholds ? ` thresholds=${JSON.stringify(m.thresholds)}` : ""}`),
  ];
  return {
    [`${RESULTS_DIR}/summary-${stamp}.json`]: json,
    [`${RESULTS_DIR}/summary-latest.json`]: json,
    stdout: `${lines.join("\n")}\n`,
  };
}
