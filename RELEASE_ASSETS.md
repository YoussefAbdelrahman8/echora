# Release Assets

To keep `git clone` lightweight, ECHORA's large binaries live on the
**[GitHub Release](https://github.com/YoussefAbdelrahman8/echora/releases)** instead of in
the repository. This page lists what to upload to a release (e.g. tag `v1.0`) and how the
files map back into the project.

## What to upload

| Asset | Build from (local path) | Unpacks to |
|-------|-------------------------|------------|
| `echora-models.zip` | `assets/models/` | `assets/models/` |
| `Echora-Promo.mp4` | `presentation/Echora-Promo.mp4` | — (watch online) |
| `Echora.mp4` | `presentation/Echora.mp4` | — (full demo) |
| `echora-thesis.pdf` | `documentation/.../main.pdf` | — (the written thesis) |
| `echora-presentation.zip` *(optional)* | `presentation/*.html` + slides | — |
| `echora-finetune.zip` *(optional)* | fine-tuning dataset / sources | training only |

> **Do not** upload `assets/database/` — it contains real face embeddings (private biometric data).

## Building the model bundle

```bash
# from the repo root
zip -r echora-models.zip assets/models
```

Users then download and unzip it back into place:

```bash
curl -L -o echora-models.zip <RELEASE_ASSET_URL>
unzip echora-models.zip          # restores assets/models/
```

## Creating the release

```bash
# requires the GitHub CLI: https://cli.github.com/
gh release create v1.0 \
  echora-models.zip \
  presentation/Echora-Promo.mp4 \
  presentation/Echora.mp4 \
  --title "ECHORA v1.0" \
  --notes "Model weights, demo videos, and supporting assets for ECHORA."
```

After creating the release, replace the `<RELEASE_ASSET_URL>` placeholders in
[`README.md`](README.md) with the real asset URLs (visible on the release page).
