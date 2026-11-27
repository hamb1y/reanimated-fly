import type { RequestHandler } from './$types';
import { env } from '$env/dynamic/private';

const target = (svc: string) =>
  svc === 'brain' ? env.BRAIN_URL ?? 'http://brain:5031' : env.PHYSICS_URL ?? 'http://physics:5032';

const handler: RequestHandler = async ({ request, params, url }) => {
  const dest = target(params.svc!).replace(/\/$/, '') + '/' + params.path + url.search;
  const hasBody = !['GET', 'HEAD'].includes(request.method);
  try {
    const r = await fetch(dest, {
      method: request.method,
      headers: { 'content-type': request.headers.get('content-type') ?? 'application/json' },
      body: hasBody ? await request.arrayBuffer() : undefined,
      signal: AbortSignal.timeout(15000)
    });
    const h = new Headers();
    const ct = r.headers.get('content-type');
    if (ct) h.set('content-type', ct);
    h.set('cache-control', 'no-store');
    return new Response(r.body, { status: r.status, headers: h });
  } catch (e) {
    return new Response(JSON.stringify({ error: 'upstream unreachable', detail: String(e) }), {
      status: 502, headers: { 'content-type': 'application/json' }
    });
  }
};
export const GET = handler, POST = handler, PUT = handler, DELETE = handler, PATCH = handler;
