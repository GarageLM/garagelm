# garagelm.org — site

Static site for garagelm.org: home, models, notes (+ dedicated note pages), learn (the forward-pass explorer), team.

## Structure

- `index.html` — home
- `models.html` — model cards + benchmark chart
- `notes.html` — lab notes + experiment roadmap
- `note-600x-tokens.html` — dedicated write-up (milestones 07+09)
- `learn.html` — forward-pass explorer (learn/01)
- `team.html` — purpose, team, hardware, outputs
- `support.js` — page runtime (required, same directory)
- `llm-engine.js` — the 188-param transformer + training loop (used by learn)
- `Matrix.dc.html` — heatmap component (imported by learn; keep this exact filename)
- `logo.png`

## Run locally

Any static server from this directory, e.g.:

    python3 -m http.server 8000

## Deploy

No build step. Options:
- **GitHub Pages**: put this folder at repo root or `/docs`, enable Pages, point garagelm.org via CNAME.
- **Cloudflare Pages / Netlify**: deploy the folder as-is.

Notes: theme + visitor counter persist in localStorage (client-only, no tracking). All demos compute in the browser; there is no backend.
