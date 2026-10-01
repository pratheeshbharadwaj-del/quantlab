#!/bin/sh
# Runs as part of nginx's standard /docker-entrypoint.d/ hook chain (files are
# executed in name order before nginx starts). Writes the backend API URL
# into config.js so the frontend can reach the backend without a rebuild.
set -e

CONFIG_FILE=/usr/share/nginx/html/config.js
API_URL=${QNT_API_URL:-}

echo "window.__QNT_API_URL__ = '${API_URL}';" > "$CONFIG_FILE"
echo "Wrote runtime config: QNT_API_URL='${API_URL}' -> $CONFIG_FILE"
