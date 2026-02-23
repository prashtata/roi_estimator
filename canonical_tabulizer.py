"""
mediapipe_canonical_face_obj.py

Minimal OBJ loader specialized for MediaPipe's canonical_face_model.obj.

Usage:
    from mediapipe_canonical_face_obj import (
        load_canonical_face_model,
        build_vertex_uv_table,
    )

    vertices, uvs, faces_v, faces_vt = load_canonical_face_model("canonical_face_model.obj")

    # Suppose you get runtime landmark positions from MediaPipe:
    # runtime_vertices: np.ndarray, shape (N, 3), same indexing as OBJ 'v'
    #
    # Then:
    vertex_uvs, triangles = build_vertex_uv_table(uvs, faces_v, faces_vt)

    # triangles: (T, 3) indices into runtime_vertices
    # vertex_uvs: (N, 2) UV per vertex (NaN where vertex never appears in faces)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class ObjMesh:
    vertices: np.ndarray        # (num_vertices, 3)
    uvs: np.ndarray             # (num_uvs, 2)
    faces_v: np.ndarray         # (num_faces, 3) vertex indices (0-based)
    faces_vt: np.ndarray        # (num_faces, 3) uv indices (0-based)


def load_canonical_face_model(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a Wavefront OBJ file containing v, vt and f lines of the form:

        v  x y z
        vt u v
        f  v1/vt1 v2/vt2 v3/vt3

    Normals and materials are ignored.

    Returns:
        vertices: (Nv, 3) float32
        uvs:      (Nu, 2) float32
        faces_v:  (F, 3)  int32  (0-based vertex indices)
        faces_vt: (F, 3)  int32  (0-based uv indices)
    """
    verts: List[Tuple[float, float, float]] = []
    texcoords: List[Tuple[float, float]] = []
    faces_v: List[Tuple[int, int, int]] = []
    faces_vt: List[Tuple[int, int, int]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):  # vertex position
                # v x y z
                _, x, y, z = line.split()
                verts.append((float(x), float(y), float(z)))

            elif line.startswith("vt "):  # texture coordinate
                # vt u v [w]
                parts = line.split()
                if len(parts) < 3:
                    continue
                _, u, v = parts[:3]
                texcoords.append((float(u), float(v)))

            elif line.startswith("f "):  # face
                # f v1/vt1[/vn1] v2/vt2[/vn2] v3/vt3[/vn3]
                parts = line.split()[1:]
                if len(parts) != 3:
                    # This canonical model only uses triangles; ignore others
                    continue

                v_idx = []
                vt_idx = []
                for p in parts:
                    # p is like "v", "v/vt", "v/vt/vn"
                    tokens = p.split("/")
                    # Wavefront indices are 1-based (and may be negative, but this file uses positive).
                    v = int(tokens[0]) - 1
                    v_idx.append(v)

                    if len(tokens) > 1 and tokens[1] != "":
                        vt = int(tokens[1]) - 1
                        vt_idx.append(vt)
                    else:
                        vt_idx.append(-1)

                faces_v.append(tuple(v_idx))
                faces_vt.append(tuple(vt_idx))

    vertices = np.asarray(verts, dtype=np.float32)
    uvs = np.asarray(texcoords, dtype=np.float32)
    faces_v = np.asarray(faces_v, dtype=np.int32)
    faces_vt = np.asarray(faces_vt, dtype=np.int32)

    return vertices, uvs, faces_v, faces_vt


def build_vertex_uv_table(
    uvs: np.ndarray,
    faces_v: np.ndarray,
    faces_vt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build:
      - vertex_uvs: (Nv, 2) UV per vertex index (0-based, NaN if never referenced)
      - triangles:  (F, 3) vertex indices (simply faces_v)

    MediaPipe's runtime landmarks will replace the vertex positions, but the
    topology (faces_v) and UV mapping (faces_vt) come from the canonical OBJ.

    Args:
        uvs:      (Nu, 2) UV array from OBJ (vt)
        faces_v:  (F, 3) vertex indices from OBJ faces (already 0-based)
        faces_vt: (F, 3) uv indices from OBJ faces (already 0-based)

    Returns:
        vertex_uvs: (Nv, 2) float32, NaN-filled then overwritten where known.
        triangles:  (F, 3) int32, same as faces_v.
    """
    if faces_v.shape != faces_vt.shape:
        raise ValueError("faces_v and faces_vt must have the same shape")

    num_vertices = int(faces_v.max()) + 1
    vertex_uvs = np.full((num_vertices, 2), np.nan, dtype=np.float32)

    # For each corner of each triangle, assign UV to its vertex index.
    # Since canonical_face_model.obj is consistent, each vertex should map
    # to a single UV; if there are duplicates, the last write wins.
    for fv_row, fvt_row in zip(faces_v, faces_vt):
        for v_idx, vt_idx in zip(fv_row, fvt_row):
            if vt_idx < 0:
                continue
            vertex_uvs[v_idx] = uvs[vt_idx]

    triangles = faces_v.copy()
    return vertex_uvs, triangles


# Example of how you would hook this into a rasterizer
def rasterization_inputs(
    obj_path: str
) -> Tuple[np.uint16, np.ndarray, np.ndarray]:
    """
    Prepare buffers for a typical GPU or software rasterizer.

    Args:
        obj_path:         Path to canonical_face_model.obj
        runtime_vertices: (Nv, 3) MediaPipe landmark positions, same indexing as OBJ v.

    Returns:
        positions: (Nv, 3) float32  (runtime_vertices, sanity-checked)
        uvs:       (Nv, 2) float32  per-vertex UVs
        triangles: (F, 3) int32     vertex indices
    """
    vertices, uvs_raw, faces_v, faces_vt = load_canonical_face_model(obj_path)

    vertex_uvs, triangles = build_vertex_uv_table(uvs_raw, faces_v, faces_vt)

    return vertices.shape[0], vertex_uvs, triangles


if __name__ == "__main__":
    # Small smoke test (no rendering).
    import pathlib

    obj_path = pathlib.Path("canonical_face_model.obj")

    # Fake runtime vertices: just re-use canonical vertices.
    verts_len, vertex_uvs, triangles = rasterization_inputs(
        str(obj_path)
    )

    print("Loaded vertices:", verts_len)
    print("Vertex UVs:", vertex_uvs.shape)
    print("Triangles:", triangles.shape)