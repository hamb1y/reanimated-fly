# syntax=docker/dockerfile:1
FROM node:22-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build && npm prune --omit=dev

FROM node:22-slim
ENV NODE_ENV=production PORT=5030 HOST=0.0.0.0 BRAIN_URL=http://brain:5031 PHYSICS_URL=http://physics:5032
WORKDIR /app
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/build ./build
COPY package.json server.js ./
COPY mock ./mock
COPY test ./test
USER node
EXPOSE 5030
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s CMD node -e "fetch('http://127.0.0.1:5030/').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["node","server.js"]
