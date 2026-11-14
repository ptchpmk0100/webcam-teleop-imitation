#!/usr/bin/env bash
# Fetch the SO-101 model from mujoco_menagerie at a pinned commit (Apache-2.0). Idempotent.
set -euo pipefail
SHA=370c43965323c0944b433b202aac5058b5f907a9
DEST="$(cd "$(dirname "$0")/.." && pwd)/assets/mujoco_menagerie"
if [ -f "$DEST/robotstudio_so101/so101.xml" ]; then
  echo "menagerie present: $DEST"
else
  mkdir -p "$DEST"
  cd "$DEST"
  git init -q
  git remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
  git fetch -q --depth 1 origin "$SHA"
  git checkout -q FETCH_HEAD
  test -f robotstudio_so101/so101.xml
  echo "fetched menagerie $SHA"
fi

# MediaPipe hand landmarker model (Apache-2.0, Google), pinned by SHA-256.
MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/assets/models"
MODEL="$MODEL_DIR/hand_landmarker.task"
MODEL_SHA=fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1
if [ ! -f "$MODEL" ]; then
  mkdir -p "$MODEL_DIR"
  curl -sSL -o "$MODEL" https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
fi
echo "$MODEL_SHA  $MODEL" | sha256sum -c --quiet - && echo "hand model ok"
