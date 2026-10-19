# QuantLab deployment (Cloudflare R2 + Render + Cloudflare Pages)

Architecture: browser → **Cloudflare Pages** (static UI) → **Render** (FastAPI, runs inference) ↔ **Cloudflare R2** (private model bucket).
Browsers upload models straight to R2 with pre-signed URLs; the backend then pulls the object into its cache disk (`/data/models`) to run inference. Your PC stores nothing.

Costs: Pages free, R2 free up to 10 GB, Render Starter ~$7/mo (needed for the persistent disk and enough RAM; large models may need a bigger plan).
No custom domain needed: you get `*.pages.dev` and `*.onrender.com` URLs.

## 1. Push to GitHub (repo already exists)
```powershell
cd "d:\qnt app"
git add -A
git status            # confirm no .env or model files are listed
git commit -m "Add cloud storage and deployment config"
git push origin main
```

## 2. Cloudflare R2 bucket
1. Sign up at https://dash.cloudflare.com/sign-up, then go to **R2 Object Storage** (add a payment method if asked; the free tier is not charged).
2. **Create bucket** named `quantlab-models`.
3. **R2 → Manage API tokens → Create API token**, permission *Object Read & Write*, scoped to this bucket. Copy the Access Key ID, Secret and the S3 endpoint (`https://<ACCOUNT_ID>.r2.cloudflarestorage.com`). The secret is shown once.
4. Bucket **Settings → CORS policy** (replace the origin with your real Pages URL after step 4):
```json
[{"AllowedOrigins": ["https://quantlab.pages.dev"], "AllowedMethods": ["PUT"], "AllowedHeaders": ["*"], "MaxAgeSeconds": 3600}]
```
Keep the bucket private (do not enable public access).

## 3. Render backend
1. Sign up at https://render.com with GitHub.
2. **New → Blueprint**, pick the `quantlab` repo. Render reads `render.yaml`.
3. Fill the prompted env vars: `QNT_S3_ENDPOINT`, `QNT_S3_BUCKET=quantlab-models`, `QNT_S3_ACCESS_KEY_ID`, `QNT_S3_SECRET_ACCESS_KEY`, and `QNT_ALLOWED_ORIGINS` (temporarily `https://quantlab.pages.dev`).
4. After deploy, note the URL, e.g. `https://quantlab-api.onrender.com`, and test `https://<it>/health` (expect `"cloud_storage": true`).

Optional runtimes (torch, llama-cpp-python, onnxruntime) are not installed by default; add them to `backend/requirements.txt` for the formats you need.

## 4. Cloudflare Pages frontend
1. Dashboard → **Workers & Pages → Create → Pages → Connect to Git**, select the repo.
2. Settings: root directory `frontend`, build command `npm run build`, output directory `dist`.
3. Environment variable: `VITE_API_URL` = your Render URL.
4. Deploy. If your Pages URL differs from `quantlab.pages.dev`, update `QNT_ALLOWED_ORIGINS` in Render and the R2 CORS origin to the exact URL (no trailing slash), then redeploy the backend.

## 5. Verify a production build locally
```powershell
cd "d:\qnt app\frontend"
$env:VITE_API_URL="https://quantlab-api.onrender.com"; npm.cmd run build; npm.cmd run preview
cd ..\backend; ..\.venv\Scripts\python.exe -m pytest tests -q
```
Optional: `docker compose up --build` tests the containers.

## CORS summary
- Browser → Render: `QNT_ALLOWED_ORIGINS` must equal the Pages origin.
- Browser → R2 (PUT upload): bucket CORS must allow that same origin and the `PUT` method.

## Checklist
- [ ] `.env` is not tracked (`git ls-files | Select-String "\.env"` shows only `.env.example`)
- [ ] R2 token is scoped to one bucket; bucket is private
- [ ] `/health` reports `cloud_storage: true`
- [ ] Uploading a small `.gguf` works end to end
- [ ] Origins in Render and R2 CORS match the Pages URL exactly
- [ ] Render disk is mounted at `/data/models`

## Security warning
The API has no login. Anyone with the URL can upload and run models on your paid server. Before sharing the link widely, put Cloudflare Access (free, up to 50 users) in front of the Pages site and API, or add an API-key check.

## Rollback
- Backend: Render → service → **Events/Deploys** → pick the previous deploy → **Rollback**.
- Frontend: Pages → **Deployments** → previous → **Rollback to this deployment**.
- Code: `git revert <commit>` and push.
- Leaked credentials: R2 → delete the API token, create a new one, update Render env vars.
- Local-only fallback: unset the `QNT_S3_*` variables and the app uses local disk mode again.

## Local model files
`backend/models/*.zip` (about 700 MB) are git-ignored and untouched. Once you have uploaded them through the app and confirmed they work, you can delete them yourself.
