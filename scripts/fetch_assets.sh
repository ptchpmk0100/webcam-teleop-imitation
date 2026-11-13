#!/usr/bin/env bash
# Fetch the SO-101 model from mujoco_menagerie at a pinned commit (Apache-2.0). Idempotent.
set -euo pipefail
SHA=370c43965323c0944b433b202aac5058b5f907a9
DEST="$(cd "$(dirname "$0")/.." && pwd)/assets/mujoco_menagerie"
if [ -f "$DEST/robotstudio_so101/so101.xml" ]; then echo "assets present: $DEST"; exit 0; fi
mkdir -p "$DEST"
cd "$DEST"
git init -q
git remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
git fetch -q --depth 1 origin "$SHA"
git checkout -q FETCH_HEAD
test -f robotstudio_so101/so101.xml
echo "fetched menagerie $SHA"
