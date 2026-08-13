# PixelGuard — Operations Guide

## If Modal credit runs out: switching to a new account

1. **Create a new Modal account**
   Sign up at https://modal.com with a different email/GitHub account
   than the current one. This gives a fresh no-card free credit under
   a new workspace name.

2. **Authenticate the new account**
   ```powershell
   python -m modal setup
   ```
   This opens the browser for the new account's login. Modal stores
   each account as a separate "profile" in `~/.modal.toml`, so this
   won't erase the old one. Switch back later with:
   ```powershell
   modal profile activate pixelguardteam
   ```

3. **Confirm the new profile is active**
   ```powershell
   modal profile current
   ```
   If it still shows the old workspace, run:
   ```powershell
   modal profile activate <new-workspace-name>
   ```

4. **Redeploy**
   ```powershell
   python -m modal deploy modal_app.py
   ```
   This deploys under the new account and prints new URLs, since the
   workspace name is baked into the URL itself, e.g.
   `https://<new-workspace>--pixelguard-immunize.modal.run`

5. **Update the URLs**
   Open `config.js` and replace both URLs with the new ones printed
   in step 4. This is the only file that needs editing — the HTML
   reads its endpoint URLs from here.

6. **Push**
   ```powershell
   git add -A
   git commit -m "switch to new modal account"
   git push
   ```
   Vercel auto-redeploys in ~30-60s, and the public link points at
   the new account's engines for every visitor.

### Important caveat
This is a manual stopgap, not a permanent fix — the public link has
real downtime between the moment credit runs out and the moment this
swap is done. For a backend that never needs this at all, migrate the
Protect engine to Hugging Face Spaces (free CPU, no credit balance,
nothing to run out) — ask to have this built when ready.

## Local development

```powershell
python -m http.server 8000
```
Then open `http://localhost:8000/pixelguard.html`.

## Files

- `modal_app.py` — the two backend engines (Protect + Verify), deployed to Modal
- `pixelguard.html` — the frontend, deployed to Vercel
- `config.js` — the two engine URLs, edited independently of the HTML
- `logo.png` — brand logo, referenced by the HTML