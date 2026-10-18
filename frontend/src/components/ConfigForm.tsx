"use client";

import { useId } from "react";

import type { Configs, SearchParams } from "@/lib/api";

export function defaultParams(configs: Configs): SearchParams {
  const bb = configs.backbones[0];
  const ready = configs.prebuilt.find((p) => p.status === "ready");
  return {
    backbone: ready?.backbone ?? bb?.name ?? "",
    pooling: ready?.pooling ?? bb?.default_pooling ?? "",
    index: ready?.index ?? configs.indexes[0]?.name ?? "flat",
    rerank: "none",
    k: configs.default_k,
  };
}

function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  const id = useId();
  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500">
        {label}
      </label>
      <select id={id} className="input" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

/** Backbone / pooling / index / re-ranker / k selectors driven by GET /configs. */
export function ConfigForm({
  configs,
  value,
  onChange,
  title,
}: {
  configs: Configs;
  value: SearchParams;
  onChange: (p: SearchParams) => void;
  title?: string;
}) {
  const kId = useId();
  const backbone = configs.backbones.find((b) => b.name === value.backbone) ?? configs.backbones[0];
  const isPrebuilt = configs.prebuilt.some(
    (p) =>
      p.backbone === value.backbone &&
      p.pooling === value.pooling &&
      p.index === value.index &&
      p.status === "ready",
  );
  return (
    <fieldset className="grid grid-cols-2 gap-3 sm:grid-cols-5">
      {title ? <legend className="col-span-full mb-1 text-sm font-semibold">{title}</legend> : null}
      <Select
        label="Backbone"
        value={value.backbone}
        options={configs.backbones.map((b) => ({ value: b.name, label: `${b.name} (${b.family})` }))}
        onChange={(name) => {
          const bb = configs.backbones.find((b) => b.name === name);
          onChange({ ...value, backbone: name, pooling: bb?.default_pooling ?? value.pooling });
        }}
      />
      <Select
        label="Pooling"
        value={value.pooling}
        options={(backbone?.poolings ?? []).map((p) => ({ value: p, label: p }))}
        onChange={(pooling) => onChange({ ...value, pooling })}
      />
      <Select
        label="Index"
        value={value.index}
        options={configs.indexes.map((i) => ({ value: i.name, label: i.name }))}
        onChange={(index) => onChange({ ...value, index })}
      />
      <Select
        label="Re-rank"
        value={value.rerank}
        options={configs.rerankers.map((r) => ({ value: r, label: r }))}
        onChange={(rerank) => onChange({ ...value, rerank })}
      />
      <div>
        <label
          htmlFor={kId}
          className="mb-1 block text-xs font-medium uppercase tracking-wide text-slate-500"
        >
          k
        </label>
        <input
          id={kId}
          type="number"
          min={1}
          max={configs.max_k}
          className="input"
          value={value.k}
          onChange={(e) =>
            onChange({ ...value, k: Math.max(1, Math.min(configs.max_k, Number(e.target.value) || 1)) })
          }
        />
      </div>
      {!isPrebuilt ? (
        <p className="col-span-full text-xs text-amber-700 dark:text-amber-400">
          Not prebuilt: the first search with this configuration extracts database descriptors and builds the
          index (can take minutes on CPU).
        </p>
      ) : null}
    </fieldset>
  );
}
