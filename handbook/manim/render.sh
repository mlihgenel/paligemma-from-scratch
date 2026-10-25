#!/bin/bash
# Renders the Manim scenes in both languages into handbook/public/videos/<Scene>.<lang>.mp4
# Usage (from the project root): bash handbook/manim/render.sh [Scene ...]
set -e
cd "$(dirname "$0")/../.."
SCENES=${@:-"ResidualHighway RMSNormGeometry GeGLUGating WeightTying ConvAsPatchLinear FlattenTranspose MultiHeadReshape"}
OUT=handbook/public/videos
MEDIA=handbook/manim/media
mkdir -p "$OUT"
for lang in en tr; do
  for s in $SCENES; do
    HANDBOOK_LANG=$lang .venv/bin/python -m manim -qm --progress_bar none --media_dir "$MEDIA" --disable_caching handbook/manim/scenes.py "$s" >/dev/null
    cp "$MEDIA/videos/scenes/720p30/$s.mp4" "$OUT/$s.$lang.mp4"
    echo "$s -> $OUT/$s.$lang.mp4"
  done
done
