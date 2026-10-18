"use client";

import { useEffect, useState, type FormEvent } from "react";

import { ConfigForm, defaultParams } from "@/components/ConfigForm";
import { Dropzone } from "@/components/Dropzone";
import { LatencyBreakdown } from "@/components/LatencyBreakdown";
import { ReadinessGate } from "@/components/ReadinessGate";
import { ResultGrid } from "@/components/ResultGrid";
import { EmptyState, ErrorState, Loading } from "@/components/StateViews";
import { ApiError, getConfigs, search, type SearchParams, type SearchResponse } from "@/lib/api";
import { fmtBytes } from "@/lib/format";
import { useAsync } from "@/lib/hooks";

function SearchView() {
  const configs = useAsync(getConfigs);
  const [file, setFile] = useState<File | null>(null);
  const [params, setParams] = useState<SearchParams | null>(null);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (configs.data && !params) setParams(defaultParams(configs.data));
  }, [configs.data, params]);

  if (configs.loading) return <Loading label="Loading configurations…" />;
  if (configs.error) return <ErrorState error={configs.error} onRetry={configs.reload} />;
  if (!configs.data || !params) return <Loading />;

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await search(file, params));
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, "unknown", String(err)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <form onSubmit={onSubmit} className="card space-y-4" aria-label="Search form">
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
          <Dropzone file={file} onFile={setFile} />
          <div className="space-y-4">
            <ConfigForm configs={configs.data} value={params} onChange={setParams} />
            <p className="text-xs text-slate-500">
              Database: <span className="font-mono">{configs.data.dataset}</span> ({configs.data.protocol}{" "}
              protocol)
              {configs.data.pca_whitening ? ", PCA-whitened" : ""}.
            </p>
            <button type="submit" className="btn-primary" disabled={!file || busy}>
              {busy ? "Searching…" : "Search"}
            </button>
          </div>
        </div>
      </form>

      {busy ? <Loading label="Running extract → search → rerank…" /> : null}
      {error ? <ErrorState error={error} /> : null}
      {!busy && !error && !result ? (
        <EmptyState title="No search yet">
          Choose a query image and a configuration, then press Search.
        </EmptyState>
      ) : null}
      {result && !busy ? (
        <section className="space-y-4" aria-label="Search results">
          <div className="card space-y-3">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="text-sm font-semibold">
                {result.results.length} results · {result.config.backbone}/{result.config.pooling} ·{" "}
                {result.config.index} · rerank {result.config.rerank}
              </h2>
              <p className="text-xs text-slate-500">
                index: {result.index.n_vectors.toLocaleString()} vectors × {result.index.dim}d,{" "}
                {fmtBytes(result.index.memory_bytes)} ({result.index.memory_method})
              </p>
            </div>
            <LatencyBreakdown latency={result.latency_ms} cache={result.cache} />
          </div>
          <ResultGrid results={result.results} />
        </section>
      ) : null}
    </div>
  );
}

export default function SearchPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Search</h1>
      <ReadinessGate>
        <SearchView />
      </ReadinessGate>
    </div>
  );
}
