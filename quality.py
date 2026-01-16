import numpy as np

def visibility_score(projected_xy, W, H):
    inside = (
        (projected_xy[:,0] >= 0) & (projected_xy[:,0] <= W) &
        (projected_xy[:,1] >= 0) & (projected_xy[:,1] <= H)
    )
    return inside.mean()

def normal_score(normals):
    view = np.array([0,0,1])
    return np.clip((normals @ view), 0, None).mean()
