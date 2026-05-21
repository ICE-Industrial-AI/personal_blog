# Root-URL redirect shim

This folder contains a tiny site that redirects visitors hitting `https://ice-industrial-ai.github.io/` to the actual blog at `https://ice-industrial-ai.github.io/personal_blog/`.

## What's in here

- `index.html` — redirects the root path to `/personal_blog/`.
- `404.html` — catches any unknown path (e.g. someone typing `/blog/foo` at the root) and rewrites it to `/personal_blog/blog/foo`. GitHub Pages serves `404.html` for any unmatched URL, so this works as a generic path-preserving redirect.

## Setup (one-time, ~5 minutes)

1. **Create a new repo** in the `ice-industrial-ai` organisation, named **exactly** `ice-industrial-ai.github.io` (this naming is what makes GitHub serve it at the org root).
2. **Copy both files** (`index.html` and `404.html`) from this folder into the **root** of that new repo. Do not put them in a subdirectory.
3. **Push to `main`**.
4. In the new repo's **Settings → Pages**, set:
   - Source: **Deploy from a branch**
   - Branch: **`main`** / folder: **`/ (root)`**
   - Save.
5. Wait ~1–2 minutes for the first deploy. Visit `https://ice-industrial-ai.github.io/` — you should be bounced to `/personal_blog/`.

## What this folder is *not*

It is not part of the Astro build for this repo. Nothing in `astro.config.mjs` references it; nothing in `dist/` includes it. Safe to leave here as the source of truth for the redirect content, or to delete after you've copied it across.
