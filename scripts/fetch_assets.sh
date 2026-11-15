#!/usr/bin/env bash
# Fetch the SO-101 model from mujoco_menagerie at a pinned commit (Apache-2.0). Idempotent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"          # absolute, taken before any cd below
SHA=370c43965323c0944b433b202aac5058b5f907a9
DEST="$ROOT/assets/mujoco_menagerie"
if [ -f "$DEST/robotstudio_so101/so101.xml" ]; then
  echo "menagerie present: $DEST"
else
  mkdir -p "$DEST"
  git -C "$DEST" init -q
  git -C "$DEST" remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
  git -C "$DEST" fetch -q --depth 1 origin "$SHA"
  git -C "$DEST" checkout -q FETCH_HEAD
  test -f "$DEST/robotstudio_so101/so101.xml"
  echo "fetched menagerie $SHA"
fi

# MediaPipe hand landmarker model (Apache-2.0, Google), pinned by SHA-256.
MODEL_DIR="$ROOT/assets/models"
MODEL="$MODEL_DIR/hand_landmarker.task"
MODEL_SHA=fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1
if [ ! -f "$MODEL" ]; then
  mkdir -p "$MODEL_DIR"
  curl -sSL -o "$MODEL" https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
fi
echo "$MODEL_SHA  $MODEL" | sha256sum -c --quiet - && echo "hand model ok"

# ONNX Runtime C/C++ release (MIT, Microsoft) for the C++ policy rollout (ticket 14); same version as the Python
# onnxruntime package, pinned by SHA-256.
ORT_VERSION=1.32.0
ORT_SHA=bb169b7e374731730d2ae406fb39de7be4b315f413a9132efe896261c31bb503
ORT_ROOT="$ROOT/assets/onnxruntime"
ORT_DIR="$ORT_ROOT/onnxruntime-linux-x64-$ORT_VERSION"
if [ ! -f "$ORT_DIR/include/onnxruntime_cxx_api.h" ]; then
  mkdir -p "$ORT_ROOT"
  curl -sSL -o "$ORT_ROOT/ort.tgz" "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-linux-x64-$ORT_VERSION.tgz"
  echo "$ORT_SHA  $ORT_ROOT/ort.tgz" | sha256sum -c --quiet -
  tar xzf "$ORT_ROOT/ort.tgz" -C "$ORT_ROOT"
fi
echo "onnxruntime $ORT_VERSION ok: $ORT_DIR"
