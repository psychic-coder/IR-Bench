/**
 * Same-origin proxy to the FastAPI service: /api/backend/<path> -> ${API_INTERNAL_URL}/<path>.
 * API_INTERNAL_URL is read at request time (docker: http://api:8000), so the image never
 * hardcodes a host and the browser never needs CORS.
 */
import { NextResponse, type NextRequest } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const PASS_RESPONSE_HEADERS = ["content-type", "cache-control", "x-request-id", "content-length"];

function backendBase(): string {
  return (process.env.API_INTERNAL_URL ?? "http://localhost:8000").replace(/\/$/, "");
}

async function proxy(req: NextRequest, { params }: { params: { path: string[] } }): Promise<NextResponse> {
  const target = new URL(`${backendBase()}/${params.path.map(encodeURIComponent).join("/")}`);
  target.search = req.nextUrl.search;
  const headers = new Headers();
  for (const name of ["content-type", "accept", "x-request-id"]) {
    const v = req.headers.get(name);
    if (v) headers.set(name, v);
  }
  const body = req.method === "GET" || req.method === "HEAD" ? undefined : await req.arrayBuffer();
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: req.method,
      headers,
      body,
      cache: "no-store",
      redirect: "manual",
    });
  } catch (err) {
    return NextResponse.json(
      {
        error: {
          code: "backend_unreachable",
          message: `cannot reach the API at ${backendBase()}: ${err instanceof Error ? err.message : String(err)}`,
          details: null,
          request_id: null,
        },
      },
      { status: 502 },
    );
  }
  const out = new Headers();
  for (const name of PASS_RESPONSE_HEADERS) {
    const v = upstream.headers.get(name);
    if (v) out.set(name, v);
  }
  return new NextResponse(upstream.body, { status: upstream.status, headers: out });
}

export { proxy as GET, proxy as POST };
