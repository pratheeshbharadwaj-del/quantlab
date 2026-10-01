// Default runtime config for local development (Vite dev server / `npm run preview`).
// In production containers, this file is generated at container startup by
// docker-entrypoint.sh from the QNT_API_URL environment variable, so the same
// built frontend image can point at different backend URLs without a rebuild.
window.__QNT_API_URL__ = '';
