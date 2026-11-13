# Vibe Studio (plugin source)
When the user asks for clips, thumbnails or ads, follow the matching skill in `plugins/vibe-studio/skills/`.
Run scripts via `plugins/vibe-studio/scripts/run.sh <script> ...`. The user is non-technical: do the work, explain plainly.
Never commit footage, outputs or keys. The `/clips` engine (`scripts/vibekit/`) is the source copy that the
vibe-listing and vibe-refresh repos vendor; note its version in their UPSTREAM.md when it changes.
