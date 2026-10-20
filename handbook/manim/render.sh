#!/bin/bash
# Manim sahnelerini render edip handbook/public/videos/ altına koyar.
# Kullanım (proje kökünden): bash handbook/manim/render.sh [Sahne ...]
set -e
cd "$(dirname "$0")/../.."
SCENES=${@:-"ResidualHighway ScaleRoundTrip RMSNormGeometry GeGLUGating WeightTying ConvAsPatchLinear FlattenTranspose MultiHeadReshape"}
OUT=handbook/public/videos
MEDIA=handbook/manim/media
mkdir -p "$OUT"
for s in $SCENES; do
  .venv/bin/python -m manim -qm --progress_bar none --media_dir "$MEDIA" --disable_caching handbook/manim/scenes.py "$s" >/dev/null
  cp "$MEDIA/videos/scenes/720p30/$s.mp4" "$OUT/$s.mp4"
  echo "$s -> $OUT/$s.mp4"
done
