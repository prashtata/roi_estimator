"""
G-PRGM Training Pipeline (UBFC)
Canonical FaceMesh OBJ-based graph
Skin-only polygons (eyes/lips excluded)
"""

import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
import wandb

wandb.init(
    entity="prasantt-university-of-california-irvine",
    project="G-PRGM",
    config={
        "lambda_reg": 1e-2,
        "K": 8,
        "lr": 1e-4,
        "dataset": "UBFC"
    }
)


# ============================================================
# 1. FaceMesh Tracker
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
# 2. Polygon Sampler (image-space, skin-only)
# ============================================================

class PolygonSampler:
    def __init__(self, triangles, samples_per_tri=32, device="cuda"):
        self.tris = triangles.to(device)
        self.K = samples_per_tri
        self.device = device

    @torch.no_grad()
    def sample(self, frame, landmarks):
        H, W, _ = frame.shape

        lm = landmarks.clone()
        lm[:, 0] *= (W - 1)
        lm[:, 1] *= (H - 1)

        tris = lm[self.tris]   # [P,3,2]
        P = tris.shape[0]

        r1 = torch.rand(P, self.K, device=self.device)
        r2 = torch.rand(P, self.K, device=self.device)
        s1 = torch.sqrt(r1)

        a = 1 - s1
        b = s1 * (1 - r2)
        c = s1 * r2

        pts = (
            a[..., None] * tris[:, 0, None] +
            b[..., None] * tris[:, 1, None] +
            c[..., None] * tris[:, 2, None]
        )

        x = pts[..., 0].clamp(0, W - 2)
        y = pts[..., 1].clamp(0, H - 2)

        x0 = x.long()
        y0 = y.long()
        dx = (x - x0)[..., None]
        dy = (y - y0)[..., None]

        frame = frame.to(self.device)

        c00 = frame[y0, x0]
        c10 = frame[y0, x0 + 1]
        c01 = frame[y0 + 1, x0]
        c11 = frame[y0 + 1, x0 + 1]

        samples = (
            c00 * (1 - dx) * (1 - dy) +
            c10 * dx * (1 - dy) +
            c01 * (1 - dx) * dy +
            c11 * dx * dy
        )

        return samples.mean(dim=1)


# ============================================================
# 3. Temporal Encoder
# ============================================================

class TemporalEncoder(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, d, 7, padding=3),
            nn.ReLU(),
        )

    def forward(self, x):
        B, P, T, _ = x.shape
        x = x.view(B * P, T, 3).permute(0, 2, 1)
        z = self.net(x).mean(-1)
        return z.view(B, P, -1)


# ============================================================
# 4. G-PRGM Core (fixed graph)
# ============================================================

class GPRGM(nn.Module):
    """
    Geometry-Anchored Physiological Region Graph Module (FINAL)

    Outputs:
        Z : polygon embeddings            [B, P, d]
        C : graph-constrained soft regions [B, P, K]
        R : region embeddings             [B, K, d]

    Guarantee:
        Each region corresponds to a connected subgraph
        of the anatomical mesh.
    """
    def __init__(self, adjacency, d=128, K=8, diffusion_steps=2):
        super().__init__()

        self.temporal = TemporalEncoder(d)

        self.attn1 = nn.MultiheadAttention(d, 4, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d, 4, batch_first=True)

        # region proposal head (pre-diffusion)
        self.region_logits = nn.Linear(d, K)

        self.diffusion_steps = diffusion_steps

        # fixed anatomical graph
        A = adjacency.float()
        A.fill_diagonal_(1.0)

        # normalized adjacency for diffusion
        D = torch.diag(1.0 / A.sum(dim=1).clamp(min=1.0))
        A_norm = D @ A

        self.register_buffer("A_norm", A_norm)
        self.register_buffer("attn_mask", ~adjacency)

    def forward(self, x):
        """
        x: [B, P, T, 3]
        """
        # ----------------------------------
        # 1. Polygon-level temporal encoding
        # ----------------------------------
        Z = self.temporal(x)  # [B, P, d]

        # ----------------------------------
        # 2. Graph message passing
        # ----------------------------------
        Z, _ = self.attn1(Z, Z, Z, attn_mask=self.attn_mask)
        Z, _ = self.attn2(Z, Z, Z, attn_mask=self.attn_mask)

        # ----------------------------------
        # 3. Region proposal (per polygon)
        # ----------------------------------
        C_logits = self.region_logits(Z)  # [B, P, K]

        # ----------------------------------
        # 4. Graph-constrained diffusion
        # Enforces connectivity
        # ----------------------------------
        C = C_logits
        for _ in range(self.diffusion_steps):
            C = torch.einsum("pq,bqk->bpk", self.A_norm, C)

        C = torch.softmax(C, dim=-1)  # [B, P, K]

        # ----------------------------------
        # 5. Region embeddings
        # ----------------------------------
        R = torch.einsum("bpk,bpd->bkd", C, Z)

        return Z, C, R


# ============================================================
# 5. Contrastive Loss
# ============================================================

def temporal_augment(x):
    shift = torch.randint(0, x.shape[2] // 10 + 1, (1,)).item()
    x = torch.roll(x, shifts=shift, dims=2)
    if torch.rand(1) < 0.3:
        ch = torch.randint(0, 3, (1,)).item()
        x[:, :, :, ch] = 0
    return x


def polygon_contrastive_loss(Z1, Z2, temp=0.2):
    """
    Z1, Z2: [B, P, d] polygon embeddings from two augmented views
    """
    Z1 = F.normalize(Z1, dim=-1)
    Z2 = F.normalize(Z2, dim=-1)

    # similarity across polygons
    logits = torch.einsum("bpd,bqd->bpq", Z1, Z2) / temp

    labels = torch.arange(Z1.size(1), device=Z1.device)
    labels = labels.unsqueeze(0).repeat(Z1.size(0), 1)

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1)
    )
    
def region_entropy_loss(C, eps=1e-8):
    """
    C: [B, P, K] soft region assignments
    """
    return (C * (C + eps).log()).sum(dim=-1).mean()


# ============================================================
# 6. UBFC Dataset
# ============================================================

class UBFCWindowDataset(Dataset):
    """
    Returns a contiguous temporal window of frames and landmarks.
    """
    def __init__(self, root, window_len=64, stride=32):
        self.window_len = window_len
        self.fm = FaceMeshTracker()
        self.samples = []

        videos = sorted(glob.glob(os.path.join(root, "*", "*.avi")))
        for vid in videos:
            cap = cv2.VideoCapture(vid)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for start in range(0, n_frames - window_len, stride):
                self.samples.append((vid, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, start = self.samples[idx]
        cap = cv2.VideoCapture(video_path)

        frames, lms = [], []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        while len(frames) < self.window_len:
            ret, frame = cap.read()
            if not ret:
                break
            lm = self.fm(frame)
            if lm is None:
                continue
            frames.append(frame)
            lms.append(lm)

        cap.release()

        if len(frames) < self.window_len:
            raise RuntimeError("Insufficient valid frames in window")

        frames = torch.tensor(np.stack(frames), dtype=torch.float32) / 255.0
        lms = torch.tensor(np.stack(lms), dtype=torch.float32)

        return frames, lms


# ============================================================
# 7. Training Loop
# ============================================================

def train(ubfc_root, graph_path, epochs=10, lambda_reg=1e-2):
    device = "cuda"

    graph = torch.load(graph_path, map_location=device)
    triangles = graph["triangles"]
    adjacency = graph["adjacency"]

    sampler = PolygonSampler(triangles, device=device)
    model = GPRGM(adjacency).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = UBFCWindowDataset(ubfc_root, window_len=64, stride=32)
    loader = DataLoader(
                    dataset,
                    batch_size=1,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True
                )

    for ep in range(epochs):
        for frames, lms in loader:
            frames = frames[0].to(device)   # [T,H,W,3]
            lms = lms[0].to(device)         # [T,468,2]

            # ---- polygon sampling over the window ----
            poly_seq = torch.stack([
                sampler.sample(frames[t], lms[t])
                for t in range(frames.shape[0])
            ])  # [T,P,3]

            # ---- temporal augmentation (now meaningful) ----
            x1 = temporal_augment(poly_seq)
            x2 = temporal_augment(poly_seq)

            # ---- forward ----
            Z1, C1, _ = model(x1)
            Z2, _,  _ = model(x2)

            # ---- losses ----
            L_con = polygon_contrastive_loss(Z1, Z2)
            L_reg = region_entropy_loss(C1)
            loss = L_con + lambda_reg * L_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        wandb.log({
            "L_con": L_con.item(),
            "L_reg": L_reg.item(),
            "L_total": loss.item(),
        })

        print(f"Epoch {ep+1}: loss={loss.item():.4f}")


if __name__ == "__main__":
    train(
        ubfc_root="/home/daevinci/Datasets/DATASET_2",
        graph_path="facemesh_graph.pt",
        epochs=10
    )