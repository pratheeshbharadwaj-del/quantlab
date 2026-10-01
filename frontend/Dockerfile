# Multi-stage build: compile the Vite/React app, then serve the static
# output with nginx. The backend API URL is injected at *container start*
# (not build time) via docker-entrypoint.sh, so one built image can be
# deployed against different backend URLs (staging/prod) without rebuilding.

FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY docker-entrypoint.sh /docker-entrypoint.d/40-qnt-config.sh
RUN chmod +x /docker-entrypoint.d/40-qnt-config.sh

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1:80/ >/dev/null || exit 1
