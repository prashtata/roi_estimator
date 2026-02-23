"""
Build fixed, skin-only anatomical graph from MediaPipe canonical face model.

Source of truth:
- canonical_face_model.obj (MediaPipe)

Output:
- facemesh_graph.pt
    - triangles: LongTensor [P,3]
    - adjacency: BoolTensor [P,P]
"""

import torch
import numpy as np
import os


# ------------------------------------------------------------
# Landmark exclusion sets (MediaPipe indices)
# ------------------------------------------------------------

LIPS = {
    61,146,91,181,84,17,314,405,321,375,
    291,308,324,318,402,317,14,87,178,88,
    95,185,40,39,37,0,267,269,270,409,
    415,310,311,312,13,82,81,42,183,78
}

LEFT_EYE = {
    33,7,163,144,145,153,154,155,
    133,173,157,158,159,160,161,246
}

RIGHT_EYE = {
    263,249,390,373,374,380,381,382,
    362,398,384,385,386,387,388,466
}

EXCLUDE_LANDMARKS = LIPS | LEFT_EYE | RIGHT_EYE


# ------------------------------------------------------------
# Load triangles from canonical OBJ
# ------------------------------------------------------------

def load_obj_triangles(obj_path):
    triangles = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("f "):
                parts = line.strip().split()
                # OBJ is 1-indexed
                tri = [int(p.split("/")[0]) - 1 for p in parts[1:4]]
                triangles.append(tri)
    return np.array(triangles, dtype=np.int64)


# ------------------------------------------------------------
# Build graph
# ------------------------------------------------------------

def build_facemesh_graph(
    obj_path,
    save_path="facemesh_graph.pt"
):
    assert os.path.exists(obj_path), f"OBJ not found: {obj_path}"

    # Load canonical triangles
    triangles_all = load_obj_triangles(obj_path)
    print(f"[INFO] Total triangles in canonical model: {len(triangles_all)}")

    # Filter out eyes + lips
    triangles = []
    for tri in triangles_all:
        if not any(v in EXCLUDE_LANDMARKS for v in tri):
            triangles.append(tri)
    triangles = np.array(triangles, dtype=np.int64)

    P = triangles.shape[0]
    print(f"[INFO] Retained skin-only polygons: {P}")

    # Build adjacency (shared vertex)
    adjacency = torch.zeros((P, P), dtype=torch.bool)

    vertex_to_polys = {}
    for i, tri in enumerate(triangles):
        for v in tri:
            vertex_to_polys.setdefault(v, []).append(i)

    for polys in vertex_to_polys.values():
        for i in polys:
            for j in polys:
                adjacency[i, j] = True

    adjacency.fill_diagonal_(True)

    graph = {
        "triangles": torch.from_numpy(triangles).long(),
        "adjacency": adjacency
    }

    torch.save(graph, save_path)

    print(f"[INFO] Saved graph to {save_path}")
    print(f"[INFO] Adjacency shape: {adjacency.shape}")


if __name__ == "__main__":
    # CHANGE THIS PATH ONCE
    canonical_obj = "canonical_face_model.obj"

    build_facemesh_graph(
        obj_path=canonical_obj,
        save_path="facemesh_graph.pt"
    )