"""One window for the live demo (redesign grill Q6): view from behind the arm, top and side views, the camera, a panel.

Rendered offscreen (MUJOCO_GL=egl) and shown with OpenCV, so every key works in the one window and there is no MuJoCo
viewer thread (it segfaulted at exit). A 5 cm grid lies on the table and a marker under the gripper shows where it
is over the table: red normally, green inside the grasp-assist zone, yellow while the assist runs.
"""
from __future__ import annotations

import mujoco
import numpy as np

_MARKER = {"off": (1.0, 0.2, 0.2, 0.9), "ready": (0.1, 1.0, 0.1, 0.9), "active": (1.0, 0.9, 0.1, 0.9)}
_CAMERAS = {                                        # azimuth, elevation, distance, look-at
    "main": (0.0, -58.0, 0.75, (0.24, 0.0, 0.02)),  # from behind the arm, like the user sits: user's right = screen right
    "top": (0.0, -89.9, 0.75, (0.22, 0.0, 0.0)),
    "side": (90.0, -5.0, 0.80, (0.22, 0.0, 0.06)),
}


class Display:
    def __init__(self, env, size=(480, 640), small=(240, 320)) -> None:
        self._env = env
        self._main = mujoco.Renderer(env.model, *size)
        self._small = mujoco.Renderer(env.model, *small)
        self._size, self._small_size = size, small
        self._cams = {}
        for name, (az, el, dist, look) in _CAMERAS.items():
            cam = mujoco.MjvCamera()
            cam.azimuth, cam.elevation, cam.distance = az, el, dist
            cam.lookat[:] = look
            self._cams[name] = cam

    @staticmethod
    def _add(scene, kind, size, pos, rgba) -> None:
        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, np.array(size, float), np.array(pos, float),
                                np.eye(3).flatten(), np.array(rgba, np.float32))
            scene.ngeom += 1

    def _decorate(self, scene, assist: str) -> None:
        for x in np.arange(0.10, 0.351, 0.05):
            self._add(scene, mujoco.mjtGeom.mjGEOM_BOX, [0.0008, 0.20, 0.0002], [x, 0, 0.0002], (0.6, 0.6, 0.6, 1))
        for y in np.arange(-0.20, 0.201, 0.05):
            self._add(scene, mujoco.mjtGeom.mjGEOM_BOX, [0.125, 0.0008, 0.0002], [0.225, y, 0.0002], (0.6, 0.6, 0.6, 1))
        ee = self._env.ee_position()
        self._add(scene, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.008, 0.008, 0.0005], [ee[0], ee[1], 0.001],
                  _MARKER.get(assist, _MARKER["off"]))

    def _render(self, renderer, name: str, assist: str) -> np.ndarray:
        renderer.update_scene(self._env.data, self._cams[name])
        self._decorate(renderer.scene, assist)
        return renderer.render()[:, :, ::-1].copy()             # RGB to BGR for OpenCV

    def compose(self, camera_bgr, lines, assist: str = "off", banner: str = "") -> np.ndarray:
        """The whole window as one BGR image: main view | top over side | camera over the text panel."""
        import cv2
        h, w = self._small_size
        main = self._render(self._main, "main", assist)
        top, side = self._render(self._small, "top", assist), self._render(self._small, "side", assist)
        for img, label in ((top, "top"), (side, "side")):
            cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if banner:
            cv2.putText(main, banner, (self._size[1] // 3, self._size[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                        (0, 255, 0), 3)
        cam = np.zeros((h, w, 3), np.uint8) if camera_bgr is None else cv2.resize(cv2.flip(camera_bgr, 1), (w, h))
        panel = np.zeros((h, w, 3), np.uint8)
        for k, text in enumerate(lines[:10]):
            cv2.putText(panel, text, (8, 20 + 23 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 255), 1)
        return np.hstack([main, np.vstack([top, side]), np.vstack([cam, panel])])

    def close(self) -> None:
        self._main.close()
        self._small.close()
