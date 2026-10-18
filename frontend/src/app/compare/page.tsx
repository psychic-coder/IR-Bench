"use client";

import { useEffect, useMemo, useState, type FormEvent } from "react";

import { ConfigForm, defaultParams } from "@/components/ConfigForm";
import { Dropzone } from "@/components/Dropzone";
import { LatencyBreakdown } from "@/components/LatencyBreakdown";
import { ReadinessGate } from "@/components/ReadinessGate";
import { ResultGrid } from "@/components/ResultGrid";
import { EmptyState, ErrorState, Loading } from "@/components/StateViews";
import {
  ApiError,
  getConfigs,
  search,
  type Configs,
  type SearchParams,
  type SearchResponse,
} from "@/lib/api";
import { useAsync } from "@/lib/hooks";

function secondDefault(configs: Configs, first: SearchParams): SearchParams {
  const other = configs.prebuilt.find(
    (p) => p.status === "ready" && (p.backbone !== first.backbone || p.index !== first.index),
  );
  return other
    ? { ...first, backbone: other.backbone, pooling: other.pooling, index: other.index }
    : { ...first, rerank: "aqe" };
}

function CompareView() {
  const configs = useAsync(getConfigs);
  const [file, setFile] = useState<File | null>(null);
  const [a, setA] = useState<SearchParams | null>(null);
  const [b, setB] = useState<SearchParams | null>(null);
  const [ra, setRa] = useState<SearchResponse | null>(null);
  const [rb, setRb] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (configs.data && !a) {
      const first = defaultParams(configs.data);
      setA(first);
      setB(secondDefault(configs.data, first));
    }
  }, [configs.data, a]);

  const [onlyA, onlyB, overlap] = useMemo(() => {
    if (!ra || !rb) return [new Set<string>(), new Set<string>(), 0];
    const ids = (r: SearchResponse) => new Set(r.results.map((x) => x.id));
    const sa = ids(ra);
    const sb = ids(rb);
    return [
      new Set([...sa].filter((x) => !sb.has(x))),
      new Set([...sb].filter((x) => !sa.has(x))),
      [...sa].filter((x) => sb.has(x)).length,
    ];
  }, [ra, rb]);

  if (configs.loading) return <Loading label="Loading configurations…" />;
  if (configs.error) return <ErrorState error={configs.error} onRetry={configs.reload} />;
  if (!configs.data || !a || !b) return <Loading />;

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    setRa(null);
    setRb(null);
    try {
      // sequential, so the two latency measurements do not compete for the same CPU
      const resA = await search(file, a);
      setRa(resA);
      setRb(await search(file, b));
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, "unknown", String(err)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <form onSubmit={onSubmit} className="card space-y-4" aria-label="Compare form">
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
          <Dropzone file={file} onFile={setFile} />
          <div className="space-y-4">
            <ConfigForm title="Configuration A" configs={configs.data} value={a} onChange={setA} />
            <ConfigForm title="Configuration B" configs={configs.data} value={b} onChange={setB} />
            <button type="submit" className="btn-primary" disabled={!file || busy}>
              {busy ? "Running both…" : "Compare"}
            </button>
          </div>
        </div>
      </form>
      {error ? <ErrorState error={error} /> : null}
      {busy ? <Loading label="Running configuration A, then B…" /> : null}
      {!busy && !error && !ra ? (
        <EmptyState title="Nothing compared yet">
          The same query runs through both configurations; results unique to one side are highlighted.
        </EmptyState>
      ) : null}
      {ra && rb ? (
        <p className="text-sm text-slate-600 dark:text-slate-300">
          {overlap} results in common · {onlyA.size} only in A · {onlyB.size} only in B
        </p>
      ) : null}
      <div className="grid gap-6 xl:grid-cols-2">
        {[
          { name: "A", res: ra, unique: onlyA },
          { name: "B", res: rb, unique: onlyB },
        ].map(({ name, res, unique }) =>
          res ? (
            <section key={name} className="space-y-3" aria-label={`Results for configuration ${name}`}>
              <h2 className="text-sm font-semibold">
                {name}: {res.config.backbone}/{res.config.pooling} · {res.config.index} · rerank{" "}
                {res.config.rerank}
              </h2>
              <div className="card">
                <LatencyBreakdown latency={res.latency_ms} cache={res.cache} />
              </div>
              <ResultGrid results={res.results} highlight={unique} />
            </section>
          ) : null,
        )}
      </div>
    </div>
  );
}

export default function ComparePage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">Compare two configurations</h1>
      <ReadinessGate>
        <CompareView />
      </ReadinessGate>
    </div>
  );
}
