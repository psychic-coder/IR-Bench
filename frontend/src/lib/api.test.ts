import { afterEach, describe, expect, it, vi } from "vitest";

import {
  API_BASE,
  ApiError,
  BenchmarkSchema,
  SearchResponseSchema,
  getConfigs,
  getReady,
  imageUrl,
  searchForm,
  toApiError,
} from "@/lib/api";

const searchResponse = {
  query_hash: "abc",
  dataset: "oxford5k",
  config: {
    backbone: "dinov2_vits14",
    pooling: "cls",
    index: "flat",
    index_params: {},
    rerank: "none",
    k: 2,
    bbox: null,
    pca_whitening: false,
  },
  results: [
    {
      rank: 1,
      id: "oxford5k/a",
      score: 0.9,
      image_url: "/images/oxford5k/a",
      thumbnail_url: "/thumbnails/oxford5k/a",
      inliers: null,
    },
  ],
  latency_ms: { extract: 10, search: 1, rerank: 0, total: 11 },
  cache: "miss",
  cache_hit: false,
  index: {
    type: "flat",
    n_vectors: 10,
    dim: 384,
    memory_bytes: 15360,
    bytes_per_vector: 1536,
    memory_method: "exact",
  },
};

afterEach(() => vi.unstubAllGlobals());

describe("schemas", () => {
  it("accepts a valid search response", () => {
    expect(SearchResponseSchema.safeParse(searchResponse).success).toBe(true);
  });

  it("rejects a search response with a wrong cache value", () => {
    expect(SearchResponseSchema.safeParse({ ...searchResponse, cache: "maybe" }).success).toBe(false);
  });

  it("parses a minimal benchmark file and keeps unknown run fields", () => {
    const parsed = BenchmarkSchema.parse({
      meta: {
        benchmark: "fast",
        config_hash: "h",
        seed: 42,
        subset: 300,
        device: "cpu",
        generated_at: "2026-01-01T00:00:00+00:00",
        hardware: {},
        libraries: {},
        headline: { protocol: "revisited", setup: "medium" },
      },
      runs: [
        {
          run_id: "r1",
          status: "ok",
          dataset: "oxford5k",
          protocol: "revisited",
          label: "x",
          extractor: "hog",
          pooling: "none",
          index: "hnsw",
          rerank: "none",
          ann_recall_k: 10,
          "ann_recall@10": 0.9,
        },
      ],
      ablations: [],
    });
    expect((parsed.runs[0] as Record<string, unknown>)["ann_recall@10"]).toBe(0.9);
    expect(parsed.runs[0]?.experiments).toEqual([]);
  });
});

describe("client helpers", () => {
  it("encodes every path segment of an image id", () => {
    expect(imageUrl("oxford5k/all souls_1")).toBe(`${API_BASE}/thumbnails/oxford5k/all%20souls_1`);
    expect(imageUrl("distractors/a/b.jpg", "images")).toBe(`${API_BASE}/images/distractors/a/b.jpg`);
  });

  it("builds multipart form data with every search field", () => {
    const form = searchForm(new Blob(["x"], { type: "image/jpeg" }), {
      backbone: "resnet50",
      pooling: "gem",
      index: "ivfpq",
      rerank: "aqe+sv",
      k: 7,
    });
    expect(form.get("backbone")).toBe("resnet50");
    expect(form.get("rerank")).toBe("aqe+sv");
    expect(form.get("k")).toBe("7");
    expect(form.get("image")).toBeInstanceOf(Blob);
  });

  it("turns the API error schema into an ApiError", async () => {
    const res = new Response(
      JSON.stringify({ error: { code: "invalid_image", message: "bad", details: null } }),
      { status: 400 },
    );
    const err = await toApiError(res);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe("invalid_image");
    expect(err.status).toBe(400);
  });

  it("treats a 503 readiness body as a successful read", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({
              ready: false,
              stage: "extracting",
              message: "m",
              progress: { done: 1, total: 4 },
              built: [],
              dataset: "oxford5k",
            }),
            { status: 503 },
          ),
      ),
    );
    const status = await getReady();
    expect(status.ready).toBe(false);
    expect(status.progress?.total).toBe(4);
  });

  it("reports schema mismatches as ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ unexpected: true }), { status: 200 })),
    );
    await expect(getConfigs()).rejects.toMatchObject({ code: "schema_mismatch" });
  });

  it("reports network failures as ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new TypeError("fetch failed"))),
    );
    await expect(getConfigs()).rejects.toMatchObject({ code: "network_error" });
  });
});
