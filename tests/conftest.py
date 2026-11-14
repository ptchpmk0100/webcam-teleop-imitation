import os

# Rendering in tests is offscreen (no window, works without a display); must be chosen before MuJoCo is imported.
os.environ.setdefault("MUJOCO_GL", "egl")
