# tests/test_mesh.py
import numpy as np
from face_mesh import FaceMeshEstimator
from kalman import FaceKalman
from quality import visibility_score, normal_score

IMG_H, IMG_W = 480, 640

def dummy_face_frame():
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    cv2.circle(img, (IMG_W//2, IMG_H//2), 120, (180,130,90), -1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def dummy_verts():
    return np.random.rand(468,3).astype(np.float32)

def dummy_uv():
    return np.random.rand(468,2).astype(np.float32)

def dummy_triangles():
    return np.random.randint(0,468,(912,3))


###################################################################################

def test_face_mesh():
    mesh = FaceMeshEstimator()
    frame = dummy_face_frame()
    verts, ok = mesh(frame)

    assert ok
    assert verts.shape == (468,3)
    assert np.isfinite(verts).all()


def test_kalman():
    kf = FaceKalman()
    base = np.ones((468,3))
    noisy = base + np.random.randn(20,468,3)*0.05

    smoothed = [kf.update(n) for n in noisy]
    diffs = [np.abs(smoothed[i] - smoothed[i-1]).mean() for i in range(1,20)]

    assert np.mean(diffs) < 0.05


def test_uv_projection():
    projector = UVProjector(dummy_uv(), dummy_triangles())
    uv, mask = projector.project(dummy_verts(), dummy_face_frame())

    assert uv.shape == (3,512,512)
    assert mask.shape == (1,512,512)
    assert uv.dtype == np.float32


def test_visibility():
    inside = np.random.rand(100,2)*100
    outside = np.random.rand(100,2)*100 + 200

    assert visibility_score(inside,100,100) == 1.0
    assert visibility_score(outside,100,100) == 0.0


def test_normals():
    facing = np.tile([0,0,1], (100,1))
    away = np.tile([0,0,-1], (100,1))

    assert normal_score(facing) == 1.0
    assert normal_score(away) == 0.0


def test_temporal_uv_stability():
    mesh = FaceMeshEstimator()
    kalman = FaceKalman()
    projector = UVProjector(dummy_uv(), dummy_triangles())

    frame = dummy_face_frame()
    prev_uv = None
    diffs = []

    for _ in range(15):
        verts, ok = mesh(frame)
        assert ok
        verts = kalman.update(verts)
        uv, _ = projector.project(verts, frame)

        if prev_uv is not None:
            diffs.append(np.abs(uv - prev_uv).mean())
        prev_uv = uv

    assert np.mean(diffs) < 0.02


###################################################################################

if __name__ == "__main__":
    test_face_mesh()
    test_kalman()
    test_uv_projection()
    test_visibility()
    test_normals()
    test_temporal_uv_stability()
    print("✅ Phase 1: ALL TESTS PASSED")


