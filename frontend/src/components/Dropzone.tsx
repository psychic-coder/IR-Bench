"use client";

import { useEffect, useId, useRef, useState, type DragEvent } from "react";

const ACCEPT = "image/jpeg,image/png,image/webp,image/bmp,image/gif,image/tiff";

/** Drag-and-drop or click-to-choose image picker with a preview (keyboard accessible). */
export function Dropzone({ file, onFile }: { file: File | null; onFile: (f: File | null) => void }) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const accept = (f: File | undefined) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError(`"${f.name}" is not an image`);
      return;
    }
    setError(null);
    onFile(f);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    accept(e.dataTransfer.files?.[0]);
  };

  return (
    <div>
      <label htmlFor={inputId} className="mb-1 block text-sm font-medium">
        Query image
      </label>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex min-h-40 flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-4 text-center text-sm transition ${
          dragging
            ? "border-accent bg-accent-soft/60 dark:bg-blue-950/40"
            : "border-slate-300 dark:border-slate-700"
        }`}
      >
        {preview ? (
          <img src={preview} alt="Selected query" className="max-h-48 rounded object-contain" />
        ) : (
          <p className="text-slate-500 dark:text-slate-400">Drop an image here, or choose a file</p>
        )}
        <div className="flex gap-2">
          <button type="button" className="btn-secondary" onClick={() => inputRef.current?.click()}>
            {file ? "Choose another" : "Choose file"}
          </button>
          {file ? (
            <button type="button" className="btn-secondary" onClick={() => onFile(null)}>
              Clear
            </button>
          ) : null}
        </div>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={ACCEPT}
          className="sr-only"
          onChange={(e) => accept(e.target.files?.[0])}
        />
        {file ? <p className="font-mono text-xs text-slate-500">{file.name}</p> : null}
        {error ? (
          <p role="alert" className="text-xs text-red-600">
            {error}
          </p>
        ) : null}
      </div>
    </div>
  );
}
