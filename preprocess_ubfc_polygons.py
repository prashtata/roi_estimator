import os
import cv2
import glob
import numpy as np
import torch
import mediapipe as mp
from tqdm import tqdm

# ============================================================
# Config
# ============================================================

UBFC_ROOT = "/share/crsp/lab/hungcao/share/UBFC/DATASET_2"
OUT_ROOT  = "./ubfc_polygons"
GRAPH_PATH = "facemesh_graph.pt"  # contains triangles, adjacency

os.makedirs(OUT_ROOT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# FaceMesh Tracker (CPU, offline)
# ============================================================

class FaceMeshTracker:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[p.x, p.y] for p in lm], dtype=np.float32)

# ============================================================
# Polygon Sampler (same as training, but CPU-safe)
# ============================================================

class PolygonSampler:
    def __init__(self, triangles, samples_per_tri=32):
        self.tris = triangles
        self.K = samples_per_tri

    def sample(self, frame, landmarks):
        H, W, _ = frame.shape

        lm = landmarks.copy()
        lm[:, 0] *= (W - 1)
        lm[:, 1] *= (H - 1)

        tris = lm[self.tris]   # [P,3,2]
        P = tris.shape[0]

        r1 = np.random.rand(P, self.K)
        r2 = np.random.rand(P, self.K)
        s1 = np.sqrt(r1)

        a = 1 - s1
        b = s1 * (1 - r2)
        c = s1 * r2

        pts = (
            a[..., None] * tris[:, 0, None] +
            b[..., None] * tris[:, 1, None] +
            c[..., None] * tris[:, 2, None]
        )

        x = np.clip(pts[..., 0], 0, W - 2)
        y = np.clip(pts[..., 1], 0, H - 2)

        x0 = x.astype(np.int32)
        y0 = y.astype(np.int32)

        dx = x - x0
        dy = y - y0

        c00 = frame[y0, x0]
        c10 = frame[y0, x0 + 1]
        c01 = frame[y0 + 1, x0]
        c11 = frame[y0 + 1, x0 + 1]

        samples = (
            c00 * (1 - dx[..., None]) * (1 - dy[..., None]) +
            c10 * dx[..., None] * (1 - dy[..., None]) +
            c01 * (1 - dx[..., None]) * dy[..., None] +
            c11 * dx[..., None] * dy[..., None]
        )

        return samples.mean(axis=1)  # [P,3]

# ============================================================
# Main Preprocessing Loop
# ============================================================

def preprocess():
    graph = torch.load(GRAPH_PATH, map_location="cpu")
    triangles = graph["triangles"].cpu().numpy()

    sampler = PolygonSampler(triangles)
    fm = FaceMeshTracker()

    videos = sorted(glob.glob(os.path.join(UBFC_ROOT, "*", "*.avi")))

    for vid in tqdm(videos, desc="Preprocessing UBFC"):
        cap = cv2.VideoCapture(vid)

        poly_seq = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            lm = fm(frame)
            if lm is None:
                continue

            frame = frame.astype(np.float32) / 255.0
            poly_rgb = sampler.sample(frame, lm)  # [P,3]
            poly_seq.append(poly_rgb)

            frame_idx += 1

        cap.release()

        if len(poly_seq) < 64:
            print(f"[WARN] Skipping short video: {vid}")
            continue

        poly_seq = np.stack(poly_seq)  # [T,P,3]

        out_name = os.path.splitext(os.path.basename(vid))[0]
        out_path = os.path.join(OUT_ROOT, out_name + "_polys.npy")

        np.save(out_path, poly_seq)

    print("Preprocessing complete.")

# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    preprocess()
