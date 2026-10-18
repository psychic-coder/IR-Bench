import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

/** Liveness of the UI process itself (used by the Docker healthcheck). */
export function GET(): NextResponse {
  return NextResponse.json({ status: "ok" });
}
