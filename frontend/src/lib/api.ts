/**
 * Typed API client. Schemas mirror the Pydantic models in backend/src/irbench/api/schemas.py
 * and every response is validated with zod before it reaches a component.
 *
 * Base URL: NEXT_PUBLIC_API_URL when set at build time (browser calls the API directly, CORS
 * must allow the UI origin), otherwise the same-origin proxy route /api/backend.
 */
import { z } from "zod";

export const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "") || "/api/backend";

// ------------------------------------------------------------------ schemas
export const ErrorBodySchema = z.object({
  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.unknown().nullable().optional(),
    request_id: z.string().nullable().optional(),
  }),
});

export const PrebuiltSchema = z.object({
  backbone: z.string(),
  pooling: z.string(),
  index: z.string(),
  status: z.enum(["pending", "building", "ready", "failed"]),
  error: z.string().nullable().optional(),
});

export const ReadySchema = z.object({
  ready: z.boolean(),
  stage: z.string(),
  message: z.string(),
  progress: z.record(z.number()).nullable().optional(),
  error: z.string().nullable().optional(),
  built: z.array(PrebuiltSchema).default([]),
  dataset: z.string(),
  n_images: z.number().nullable().optional(),
});
export type Ready = z.infer<typeof ReadySchema>;

export const BackboneSchema = z.object({
  name: z.string(),
  family: z.string(),
  poolings: z.array(z.string()),
  default_pooling: z.string(),
  image_size: z.number().nullable(),
  description: z.string(),
});
export type Backbone = z.infer<typeof BackboneSchema>;

export const ConfigsSchema = z.object({
  dataset: z.string(),
  protocol: z.string(),
  pca_whitening: z.boolean(),
  backbones: z.array(BackboneSchema),
  indexes: z.array(z.object({ name: z.string(), params: z.record(z.unknown()) })),
  rerankers: z.array(z.string()),
  default_k: z.number(),
  max_k: z.number(),
  prebuilt: z.array(PrebuiltSchema),
  sample_images: z.array(z.string()).default([]),
});
export type Configs = z.infer<typeof ConfigsSchema>;

export const SearchResultSchema = z.object({
  rank: z.number(),
  id: z.string(),
  score: z.number(),
  image_url: z.string(),
  thumbnail_url: z.string(),
  inliers: z.number().nullable().optional(),
});
export type SearchResult = z.infer<typeof SearchResultSchema>;

export const LatencySchema = z.object({
  extract: z.number(),
  search: z.number(),
  rerank: z.number(),
  total: z.number(),
});
export type Latency = z.infer<typeof LatencySchema>;

export const SearchResponseSchema = z.object({
  query_hash: z.string(),
  dataset: z.string(),
  config: z.object({
    backbone: z.string(),
    pooling: z.string(),
    index: z.string(),
    index_params: z.record(z.unknown()),
    rerank: z.string(),
    k: z.number(),
    bbox: z.array(z.number()).nullable().optional(),
    pca_whitening: z.boolean(),
  }),
  results: z.array(SearchResultSchema),
  latency_ms: LatencySchema,
  cache: z.enum(["hit", "miss", "disabled"]),
  cache_hit: z.boolean(),
  index: z.object({
    type: z.string(),
    n_vectors: z.number(),
    dim: z.number(),
    memory_bytes: z.number(),
    bytes_per_vector: z.number(),
    memory_method: z.string(),
  }),
});
export type SearchResponse = z.infer<typeof SearchResponseSchema>;

const StatsSchema = z
  .object({
    p50: z.number().nullable(),
    p95: z.number().nullable(),
    p99: z.number().nullable(),
    mean: z.number().nullable().optional(),
  })
  .passthrough();

const SetupMetricsSchema = z
  .object({
    map: z.number().nullable(),
    "recall@1": z.number().nullable().optional(),
    "recall@5": z.number().nullable().optional(),
    "recall@10": z.number().nullable().optional(),
  })
  .passthrough();

export const RunSchema = z
  .object({
    run_id: z.string(),
    status: z.enum(["ok", "skipped", "failed"]),
    dataset: z.string(),
    protocol: z.string(),
    label: z.string(),
    family: z.string().optional(),
    extractor: z.string(),
    pooling: z.string().nullable(),
    index: z.string(),
    rerank: z.string(),
    reason: z.string().nullable().optional(),
    headline_map: z.number().nullable().optional(),
    metrics: z.record(SetupMetricsSchema).optional(),
    latency_ms: z.record(StatsSchema).optional(),
    memory: z
      .object({
        total_bytes: z.number(),
        bytes_per_vector: z.number(),
        encoding_bytes_per_vector: z.number(),
        method: z.string(),
      })
      .passthrough()
      .optional(),
    index_params_effective: z.record(z.unknown()).optional(),
    ann_recall_k: z.number().optional(),
    n_distractors: z.number().optional(),
    vector_dim: z.number().optional(),
    experiments: z.array(z.object({ name: z.string(), ablation: z.string() })).default([]),
  })
  .passthrough();
export type Run = z.infer<typeof RunSchema>;

export const BenchmarkSchema = z.object({
  meta: z
    .object({
      benchmark: z.string(),
      config_hash: z.string(),
      seed: z.number(),
      subset: z.number().nullable().optional(),
      device: z.string(),
      generated_at: z.string(),
      hardware: z.record(z.unknown()),
      libraries: z.record(z.string()),
      headline: z.object({ protocol: z.string(), setup: z.string() }).passthrough(),
    })
    .passthrough(),
  runs: z.array(RunSchema),
  ablations: z.array(
    z
      .object({
        key: z.string(),
        title: z.string(),
        columns: z.array(z.string()),
        rows: z.array(z.array(z.unknown())),
        notes: z.array(z.string()),
        plot: z.string().nullable(),
      })
      .passthrough(),
  ),
});
export type Benchmark = z.infer<typeof BenchmarkSchema>;

const ReturnedSchema = z.object({
  rank: z.number(),
  id: z.string(),
  score: z.number(),
  label: z.enum(["positive", "junk", "negative", "distractor"]),
});

export const HardQueriesSchema = z.object({
  generated_at: z.string(),
  threshold_ap: z.number(),
  datasets: z.record(
    z.object({
      best_run_id: z.string(),
      best_config: z.string(),
      best_map_medium: z.number().nullable(),
      n_queries: z.number(),
      n_hard: z.number(),
      queries: z.array(
        z.object({
          query_id: z.string(),
          bbox: z.array(z.number()).nullable(),
          ap: z.record(z.number().nullable()),
          n_positives: z.number(),
          n_junk: z.number(),
          expected_positives: z.array(z.string()),
          first_positive_rank_in_top50: z.number().nullable(),
          returned: z.array(ReturnedSchema),
        }),
      ),
    }),
  ),
});
export type HardQueries = z.infer<typeof HardQueriesSchema>;

// ------------------------------------------------------------------ client
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
    public readonly details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Build an absolute (or proxy-relative) URL for an API path such as `/thumbnails/<id>`. */
export function apiUrl(path: string): string {
  const clean = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${clean}`;
}

/** URL of an image or thumbnail given a global id like `oxford5k/all_souls_000013`. */
export function imageUrl(id: string, kind: "images" | "thumbnails" = "thumbnails"): string {
  return apiUrl(`/${kind}/${id.split("/").map(encodeURIComponent).join("/")}`);
}

export async function toApiError(res: Response): Promise<ApiError> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  const parsed = ErrorBodySchema.safeParse(body);
  if (parsed.success) {
    return new ApiError(
      res.status,
      parsed.data.error.code,
      parsed.data.error.message,
      parsed.data.error.details,
    );
  }
  return new ApiError(res.status, "http_error", `request failed with status ${res.status}`);
}

async function request<S extends z.ZodTypeAny>(
  path: string,
  schema: S,
  init?: RequestInit,
): Promise<z.infer<S>> {
  let res: Response;
  try {
    res = await fetch(apiUrl(path), { cache: "no-store", ...init });
  } catch (err) {
    throw new ApiError(0, "network_error", err instanceof Error ? err.message : "network error");
  }
  if (!res.ok) throw await toApiError(res);
  const json: unknown = await res.json();
  const parsed = schema.safeParse(json);
  if (!parsed.success) {
    throw new ApiError(
      res.status,
      "schema_mismatch",
      `unexpected response shape: ${parsed.error.issues[0]?.message ?? ""}`,
    );
  }
  return parsed.data as z.infer<S>;
}

/** /ready answers 503 with a valid body while warming up; both are "successful" reads. */
export async function getReady(): Promise<Ready> {
  let res: Response;
  try {
    res = await fetch(apiUrl("/ready"), { cache: "no-store" });
  } catch (err) {
    throw new ApiError(0, "network_error", err instanceof Error ? err.message : "network error");
  }
  const json: unknown = await res.json().catch(() => null);
  const parsed = ReadySchema.safeParse(json);
  if (parsed.success) return parsed.data;
  const err = ErrorBodySchema.safeParse(json);
  if (err.success) throw new ApiError(res.status, err.data.error.code, err.data.error.message);
  throw new ApiError(res.status, "backend_unreachable", "the API did not return a readiness status");
}

export const getConfigs = () => request("/configs", ConfigsSchema);
export const getBenchmark = () => request("/benchmark", BenchmarkSchema);
export const getHardQueries = () => request("/hard-queries", HardQueriesSchema);

export interface SearchParams {
  backbone: string;
  pooling: string;
  index: string;
  rerank: string;
  k: number;
}

export function searchForm(file: Blob, params: SearchParams, filename = "query.jpg"): FormData {
  const form = new FormData();
  form.append("image", file, filename);
  form.append("backbone", params.backbone);
  form.append("pooling", params.pooling);
  form.append("index", params.index);
  form.append("rerank", params.rerank);
  form.append("k", String(params.k));
  return form;
}

export function search(file: File, params: SearchParams): Promise<SearchResponse> {
  return request("/search", SearchResponseSchema, {
    method: "POST",
    body: searchForm(file, params, file.name),
  });
}
