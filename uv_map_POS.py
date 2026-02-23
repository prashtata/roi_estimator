import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
from collections import deque
from mp_face_model_rasterizer import build_canonical_face_uv_map, TRIANGLES, UV_LANDMARKS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
from pytictoc import TicToc

t = TicToc()

# =========================
# 1. MediaPipe FaceMesh
# =========================

class FaceMeshTracker:
    def __init__(self, device='cuda'):
        self.device = device
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mp_face.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[p.x, p.y, p.z] for p in lm], np.float32)

# # =========================
# # 2. Canonical UV Rasterizer (CPU version, CUDA‑ready)
# # =========================

# class UVRasterizer:
#     def __init__(self, tex_size=256):
#         self.tex_size = tex_size

#     def rasterize(self, frame, landmarks):
#         h, w, _ = frame.shape
#         uv = np.zeros((self.tex_size, self.tex_size, 3), np.float32)
#         for (x,y,_ ) in landmarks:
#             u = int(np.clip(x * self.tex_size, 0, self.tex_size-1))
#             v = int(np.clip(y * self.tex_size, 0, self.tex_size-1))
#             uv[v,u] = frame[int(y*h), int(x*w)] / 255.0
#         return uv


# 3. Anatomical Polygon Sampler


class PolygonSampler:
    
    def __init__(self, mp_triangulation, tex_size=256, device='cuda'):
        self.tris = np.array(mp_triangulation, np.int32) # [P,3] landmark indices
        self.P = len(self.tris)
        self.tex_size = tex_size
        self.device = device
        self.centers = None # computed after first frame


    def build_uv_centers(self, uv_landmarks):
        # uv_landmarks: [468,2] canonical UV coords in [0,1]
        centers = []
        for i0,i1,i2 in self.tris:
            c = (uv_landmarks[i0] + uv_landmarks[i1] + uv_landmarks[i2]) / 3.0
            centers.append(c)
        self.centers = torch.tensor(np.array(centers), device=self.device, dtype=torch.float32)


    def sample(self, uv_tex, uv_landmarks):
        """
        Vectorized polygon sampler (CPU, NumPy) — no Python loops over polygons.
        Each polygon is one MediaPipe triangle. We rasterize all triangles in one pass.
        Returns: [P,3]
        """
        
        if self.centers is None:
            self.build_uv_centers(uv_landmarks)


        H, W, _ = uv_tex.shape


        # Triangle vertex pixel coordinates: [P,3,2]
        tris_uv = uv_landmarks[self.tris] # [P,3,2] in [0,1]
        tris_px = (tris_uv * np.array([W, H])).astype(np.int32)


        # Precompute full image grid once: [H,W,2]
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        grid = np.stack([xs, ys], axis=-1) # [H,W,2]


        # Barycentric test (vectorized)
        p0 = tris_px[:,0]; p1 = tris_px[:,1]; p2 = tris_px[:,2]
        v0 = p1 - p0; v1 = p2 - p0
        v2 = grid[None] - p0[:,None,None]


        d00 = (v0*v0).sum(-1)[:,None,None]
        d01 = (v0*v1).sum(-1)[:,None,None]
        d11 = (v1*v1).sum(-1)[:,None,None]
        d20 = (v2*v0[:,None,None]).sum(-1)
        d21 = (v2*v1[:,None,None]).sum(-1)
        denom = d00*d11 - d01*d01 + 1e-6


        a = (d11*d20 - d01*d21) / denom
        b = (d00*d21 - d01*d20) / denom
        c = 1 - a - b


        mask = (a>=0)&(b>=0)&(c>=0) # [P,H,W]


        # Sum RGB per triangle
        rgb = uv_tex[None] * mask[...,None]
        sums = rgb.sum(axis=(1,2))
        counts = mask.sum(axis=(1,2))[:,None] + 1e-6
        polys = sums / counts


        return polys.astype(np.float32)
    
    
class PolygonSamplerMC:
    def __init__(self, mp_triangulation, samples_per_tri=256, device='cuda'):
        self.tris = torch.tensor(mp_triangulation, device=device)
        self.K = samples_per_tri
        self.device = device
        self.centers = None
        
    def build_uv_centers(self, uv_landmarks):
        # uv_landmarks: [468,2] canonical UV coords in [0,1]
        centers = []
        for i0,i1,i2 in self.tris:
            c = (uv_landmarks[i0] + uv_landmarks[i1] + uv_landmarks[i2]) / 3.0
            centers.append(c)
        self.centers = torch.tensor(np.array(centers), device=self.device, dtype=torch.float32)


    def sample(self, uv_tex, uv_landmarks):
        """
        uv_tex: [H,W,3] torch tensor (float32, GPU)
        uv_landmarks: [468,2] torch tensor in [0,1]
        returns: [P,3]
        """
        
        if self.centers is None:
            self.build_uv_centers(uv_landmarks)
        
        uv_landmarks = uv_landmarks.to(self.device, non_blocking=True)
        uv_tex = uv_tex.to(self.device, non_blocking=True)

        H, W, _ = uv_tex.shape
        P = self.tris.shape[0]

        # triangle vertices [P,3,2]
        tris = uv_landmarks[self.tris]

        # random barycentric coords
        r1 = torch.rand(P, self.K, device=self.device)
        r2 = torch.rand(P, self.K, device=self.device)
        s1 = torch.sqrt(r1)

        a = 1 - s1
        b = s1 * (1 - r2)
        c = s1 * r2

        pts = (
            a[...,None] * tris[:,0,None] +
            b[...,None] * tris[:,1,None] +
            c[...,None] * tris[:,2,None]
        )  # [P,K,2]

        # UV → pixel
        px = pts.clone()
        px[...,0] *= W - 1
        px[...,1] *= H - 1

        # bilinear sample
        x0 = px[...,0].long().clamp(0, W-2)
        y0 = px[...,1].long().clamp(0, H-2)
        dx = px[...,0] - x0.float()
        dy = px[...,1] - y0.float()
        dx = dx.unsqueeze(-1)
        dy = dy.unsqueeze(-1)

        tex = uv_tex
        c00 = tex[y0, x0]
        c10 = tex[y0, x0+1]
        c01 = tex[y0+1, x0]
        c11 = tex[y0+1, x0+1]

        samples = (
            c00*(1-dx)*(1-dy) +
            c10*dx*(1-dy) +
            c01*(1-dx)*dy +
            c11*dx*dy
        )

        return samples.mean(dim=1)


# 4. Temporal Encoder


class TemporalEncoder(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3,32,3,padding=1), nn.ReLU(),
            nn.Conv1d(32,64,5,padding=2), nn.ReLU(),
            nn.Conv1d(64,d,7,padding=3), nn.ReLU(),
        )

    def forward(self,x):
        B,P,T,_ = x.shape
        x = x.view(B*P,T,3).permute(0,2,1)
        z = self.net(x).mean(-1)
        return z.view(B,P,-1)


# 5. Graph Builder


def build_graph(z, centers):
    dist = torch.cdist(centers, centers)
    geo = dist < 0.15
    sim = torch.einsum('bpd,bqd->bpq', z, z).mean(0)
    pulse = sim > 0.6
    return geo | pulse


# 6. G‑PRGM Core


class GPRGM(nn.Module):
    def __init__(self, P=120, d=128, K=8):
        super().__init__()
        self.temporal = TemporalEncoder(d)
        self.attn1 = nn.MultiheadAttention(d,4,batch_first=True)
        self.attn2 = nn.MultiheadAttention(d,4,batch_first=True)
        self.cluster = nn.Linear(d,K)

    def forward(self,x,centers):
        z = self.temporal(x)
        E = build_graph(z, centers)
        z,_ = self.attn1(z,z,z)
        z,_ = self.attn2(z,z,z)
        C = F.gumbel_softmax(self.cluster(z),tau=0.5,hard=False)
        return z,E,C


# 7. rPPG Head (POS)


class POSHead(nn.Module):
    def forward(self, regions):
        X = regions.mean(1)
        X = X - X.mean(0)
        return X[:,0] - X[:,1]


# 8. Real‑Time Inference


class RealTimePipeline:
    def __init__(self):
        self.fm = FaceMeshTracker()
        # self.uv = UVRasterizer()
        # print(FACEMESH_TESSELATION)
        # self.triangles = np.array(list(FACEMESH_TESSELATION), dtype=np.int32)
        self.triangles = TRIANGLES
        self.uv_lms = torch.Tensor(UV_LANDMARKS)
        self.sampler = PolygonSamplerMC(self.triangles)
        self.model = GPRGM(P=len(self.triangles)).cuda().half().eval()
        self.head = POSHead().cuda().half()
        self.buffer = deque(maxlen=64)

    def run(self):
        cap = cv2.VideoCapture(0)
        running = True
        while running:
            ret,frame = cap.read()
            if not ret: break
            lm = self.fm(frame)
            if lm is None: continue
            # uv = self.uv.rasterize(frame,lm)
            t.tic()
            uv = torch.Tensor(build_canonical_face_uv_map(frame, lm))
            t.toc('UV Map construction : ')
            t.tic()
            poly = self.sampler.sample(uv, self.uv_lms)
            self.buffer.append(poly)
            t.toc('UV sampling : ')
            print(len(self.buffer))
            if len(self.buffer) < 64: continue
            x = torch.stack(list(self.buffer))[None]  # [1,T,P,3]
            # shape: [B=1, T=64, P, 3]
            x = x.permute(0,2,1,3)   # → [B, P, T, 3]
            t.tic()
            with torch.amp.autocast('cuda'):
                z,E,C = self.model(x,self.sampler.centers)
                regions = C.transpose(1,2) @ z
                bvp = self.head(regions)
            t.toc('Model run : ')
            print(f"HR proxy: {bvp[-1].item():.3f}")
            cv2.imshow('frame',frame)
            if cv2.waitKey(1)==27: break
            # running = False
        cap.release()
        cv2.destroyAllWindows()


# 9. Entry Point


if __name__ == '__main__':
    RealTimePipeline().run()
