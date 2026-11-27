// Production entry: SvelteKit handler + WebSocket proxies (/ws/brain, /ws/physics).
import http from 'node:http';
import { WebSocketServer, WebSocket } from 'ws';
import { handler } from './build/handler.js';

const PORT = Number(process.env.PORT ?? 5030);
const HOST = process.env.HOST ?? '0.0.0.0';
const BRAIN = process.env.BRAIN_URL ?? 'http://brain:5031';
const PHYSICS = process.env.PHYSICS_URL ?? 'http://physics:5032';
const toWs = (u) => u.replace(/^http/, 'ws').replace(/\/$/, '');
const routes = { '/ws/brain': toWs(BRAIN) + '/stream', '/ws/physics': toWs(PHYSICS) + '/stream' };

const server = http.createServer(handler);
const wss = new WebSocketServer({ noServer: true });

server.on('upgrade', (req, socket, head) => {
  const path = new URL(req.url, 'http://x').pathname;
  const upstreamUrl = routes[path];
  if (!upstreamUrl) return socket.destroy();
  wss.handleUpgrade(req, socket, head, (client) => {
    const up = new WebSocket(upstreamUrl);
    const pending = [];
    up.on('open', () => pending.splice(0).forEach(([d, b]) => up.send(d, { binary: b })));
    up.on('message', (d, bin) => client.readyState === 1 && client.send(d, { binary: bin }));
    up.on('close', () => client.close(1011, 'upstream closed'));
    up.on('error', () => client.close(1011, 'upstream unreachable'));
    client.on('message', (d, bin) => (up.readyState === 1 ? up.send(d, { binary: bin }) : pending.push([d, bin])));
    client.on('close', () => up.close());
    client.on('error', () => up.close());
  });
});

server.listen(PORT, HOST, () => console.log(`flybrain web on ${HOST}:${PORT} brain=${BRAIN} physics=${PHYSICS}`));
