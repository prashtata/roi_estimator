import numpy as np

class UVProjector:
    def __init__(self, uv_coords, triangles, size=512):
        self.uv = uv_coords        # (468,2)
        self.tri = triangles      # (T,3)
        self.size = size

    def project(self, verts, image):
        H, W, _ = image.shape
        uv_map = np.zeros((3,self.size,self.size), dtype=np.float32)
        mask = np.zeros((1,self.size,self.size), dtype=np.float32)

        # reference implementation (triangle loop)
        for t in self.tri:
            pts_uv = self.uv[t]
            pts_xy = verts[t][:,:2] * [W, H]
            # barycentric rasterization (omitted here for brevity)

        return uv_map, mask
