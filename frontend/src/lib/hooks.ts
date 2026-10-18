"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, getReady, type Ready } from "@/lib/api";

export interface ReadyState {
  status: Ready | null;
  error: ApiError | null;
  loading: boolean;
}

/** Poll /ready every `intervalMs` until the backend reports ready (then poll slowly). */
export function useReady(intervalMs = 3000): ReadyState {
  const [state, setState] = useState<ReadyState>({ status: null, error: null, loading: true });
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const status = await getReady();
        if (cancelled) return;
        setState({ status, error: null, loading: false });
        timer = setTimeout(tick, status.ready ? intervalMs * 10 : intervalMs);
      } catch (err) {
        if (cancelled) return;
        setState((s) => ({ status: s.status, error: err as ApiError, loading: false }));
        timer = setTimeout(tick, intervalMs);
      }
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [intervalMs]);
  return state;
}

export interface AsyncState<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  reload: () => void;
}

/** Run an async loader once (and on `reload`), tracking loading/error state. */
export function useAsync<T>(loader: () => Promise<T>): AsyncState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    loaderRef
      .current()
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof ApiError ? e : new ApiError(0, "unknown", String(e)));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [nonce]);
  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { data, error, loading, reload };
}
