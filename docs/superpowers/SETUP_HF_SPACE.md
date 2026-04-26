# HF Space Setup Checklist

One-time manual steps to create and configure the Hugging Face Space.
Run these BEFORE executing `scripts/deploy_space.py`.

---

## 1. Create a HF account

Go to https://huggingface.co and sign up (free).

## 2. Create the Space

1. Click your profile → "New Space"
2. Fill in:
   - **Owner:** your HF username
   - **Space name:** `rag-pt-backend` (or any name — just keep it consistent)
   - **SDK:** Docker
   - **Visibility:** Public (required for free CPU tier)
3. Click "Create Space"

## 3. Generate a write-access token

1. Go to HF Settings → Access Tokens
2. Click "New token"
3. Name it (e.g. `deploy-token`), set role to **write**
4. Copy the token — you will use it as `HF_TOKEN` in your shell

**Never commit this token.** Set it only as a shell variable.

## 4. Add secrets to the Space

In your Space page → Settings → Variables and secrets → "New secret":

| Name | Value |
|---|---|
| `GROQ_API_KEY` | Your Groq key from console.groq.com/keys |
| `CEREBRAS_API_KEY` | Your Cerebras key from cloud.cerebras.ai |

## 5. Populate `data/chroma_db/` locally

The Chroma vector store must be built locally before deploying:

```bash
.venv/Scripts/python.exe -m data.main
```

This populates `data/chroma_db/` with ~2298 PT-BR chunks.

## 6. Run the deploy script

From the project root:

```bash
HF_SPACE_ID=yourusername/rag-pt-backend \
HF_TOKEN=hf_yourtoken \
  .venv/Scripts/python.exe scripts/deploy_space.py
```

On Windows PowerShell:
```powershell
$env:HF_SPACE_ID="yourusername/rag-pt-backend"
$env:HF_TOKEN="hf_yourtoken"
.venv\Scripts\python.exe scripts\deploy_space.py
```

## 7. Watch the build

Go to your Space → "Logs" tab.
First build takes ~5–10 minutes (downloading torch + sentence-transformers).
Subsequent deploys rebuild from the Docker layer cache — much faster.

## 8. Verify the deploy

Once the Space shows "Running", verify:

```bash
curl https://yourusername-rag-pt-backend.hf.space/health
# Expected: {"status":"ok","models_loaded":true,"chroma_count":2298}
```

Then run the full smoke test:

```bash
HF_SPACE_URL=https://yourusername-rag-pt-backend.hf.space \
  .venv/Scripts/python.exe scripts/smoke_test.py
```

Expected: `Smoke test PASSED.`

## 9. Demo warmup note

HF Space free tier sleeps after ~48h of inactivity (cold start ~30s).
Before a demo, call `/health` at least 5 minutes in advance to warm up the container.

```bash
curl https://yourusername-rag-pt-backend.hf.space/health
```
