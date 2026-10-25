#!/bin/bash
# Builds the handbook for GitHub Pages and force-pushes the build to the gh-pages branch.
# The data and videos are not in git (they come from the gated checkpoint), so they must exist locally:
#   run the handbook/export scripts and handbook/manim/render.sh first.
# Usage (from anywhere): bash handbook/scripts/deploy.sh
set -e
cd "$(dirname "$0")/.."
REPO_URL=$(git remote get-url origin)
REPO_NAME=$(basename -s .git "$REPO_URL")

test -f public/data/pipeline.json || { echo "public/data is empty: run the export scripts first"; exit 1; }
ls public/videos/*.mp4 >/dev/null 2>&1 || { echo "public/videos is empty: run manim/render.sh first"; exit 1; }

HANDBOOK_BASE="/$REPO_NAME/" npm run build
touch dist/.nojekyll

cd dist
rm -rf .git
git init -q
git checkout -q -b gh-pages
git add -A
git commit -q -m "Deploy the handbook"
git push -f "$REPO_URL" gh-pages
rm -rf .git
echo "Pushed to gh-pages: https://$(echo "$REPO_URL" | sed -E 's#.*github.com[:/]([^/]+)/.*#\1#').github.io/$REPO_NAME/"
