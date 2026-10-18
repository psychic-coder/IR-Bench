"use client";

import type { ReactNode } from "react";

import { ErrorState, Loading, WarmingUp } from "@/components/StateViews";
import { useReady } from "@/lib/hooks";

/**
 * blocking=true: children render only once /ready says ready (search, compare).
 * blocking=false: children always render; a banner shows while the backend warms up.
 */
export function ReadinessGate({ children, blocking = true }: { children: ReactNode; blocking?: boolean }) {
  const { status, error, loading } = useReady();
  if (loading) return <Loading label="Contacting the API…" />;
  if (!status && error) {
    return blocking ? (
      <ErrorState error={error} />
    ) : (
      <>
        <ErrorState error={error} />
        <div className="mt-6">{children}</div>
      </>
    );
  }
  if (status && !status.ready) {
    return blocking ? (
      <WarmingUp status={status} />
    ) : (
      <>
        <WarmingUp status={status} />
        <div className="mt-6">{children}</div>
      </>
    );
  }
  return <>{children}</>;
}
