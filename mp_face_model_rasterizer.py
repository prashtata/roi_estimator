import cv2
import numpy as np
import mediapipe as mp
import pathlib
from canonical_tabulizer import rasterization_inputs

canonical_face_model_path = pathlib.Path("canonical_face_model.obj")
verts_len, vertex_uvs, triangles = rasterization_inputs(str(canonical_face_model_path))

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Create a UV-space texture that we will fill by sampling the camera frame.
# You can pick any resolution.
UV_TEX_H, UV_TEX_W = 512, 512

def draw_face_triangle_to_uv(
    img,            # current BGR frame (H, W, 3)
    p0, p1, p2,     # image-space vertices (x,y) in pixels
    uv0, uv1, uv2,  # canonical UVs for the same vertices (u,v in [0,1])
    tex             # UV texture (UV_TEX_H, UV_TEX_W, 3)
):
    h_tex, w_tex, _ = tex.shape

    q0 = np.array([uv0[0] * w_tex, (1.0 - uv0[1]) * h_tex], dtype=np.float32)
    q1 = np.array([uv1[0] * w_tex, (1.0 - uv1[1]) * h_tex], dtype=np.float32)
    q2 = np.array([uv2[0] * w_tex, (1.0 - uv2[1]) * h_tex], dtype=np.float32)

    uv_tri = np.stack([q0, q1, q2], axis=0)
    min_x = max(int(np.floor(uv_tri[:, 0].min())), 0)
    max_x = min(int(np.ceil(uv_tri[:, 0].max())), w_tex - 1)
    min_y = max(int(np.floor(uv_tri[:, 1].min())), 0)
    max_y = min(int(np.ceil(uv_tri[:, 1].max())), h_tex - 1)

    A, B, C = q0, q1, q2
    denom = ((B[1] - C[1]) * (A[0] - C[0]) +
             (C[0] - B[0]) * (A[1] - C[1]))
    if abs(denom) < 1e-6:
        return

    h_img, w_img, _ = img.shape

    xs = np.arange(min_x, max_x + 1, dtype=np.float32)
    ys = np.arange(min_y, max_y + 1, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    w1 = ((B[1] - C[1]) * (X - C[0]) +
          (C[0] - B[0]) * (Y - C[1])) / denom
    w2 = ((C[1] - A[1]) * (X - C[0]) +
          (A[0] - C[0]) * (Y - C[1])) / denom
    w3 = 1.0 - w1 - w2

    mask = (w1 >= 0) & (w2 >= 0) & (w3 >= 0)

    if not np.any(mask):
        return

    P0 = np.array(p0, dtype=np.float32)
    P1 = np.array(p1, dtype=np.float32)
    P2 = np.array(p2, dtype=np.float32)

    P = (w1[..., None] * P0 +
         w2[..., None] * P1 +
         w3[..., None] * P2)

    u_img = np.clip(P[..., 0].astype(np.int32), 0, w_img - 1)
    v_img = np.clip(P[..., 1].astype(np.int32), 0, h_img - 1)

    tex_y, tex_x = np.where(mask)
    tex[min_y + tex_y, min_x + tex_x] = img[v_img[mask], u_img[mask]]


def build_canonical_face_uv_map(frame_bgr, landmarks):
    """
    landmarks: list of 468 MediaPipe landmarks; each has x,y in [0,1] image coords.
    Returns: UV texture image (canonical atlas) with this frame's face texture.
    """
    h, w = frame_bgr.shape[:2]
    tex = np.zeros((UV_TEX_H, UV_TEX_W, 3), dtype=np.uint8)

    for tri in triangles:
        i0, i1, i2 = tri

        # Image-space positions for this frame
        lm0 = landmarks[i0]
        lm1 = landmarks[i1]
        lm2 = landmarks[i2]
        p0 = (lm0[0] * w, lm0[1] * h)
        p1 = (lm1[0] * w, lm1[1] * h)
        p2 = (lm2[0] * w, lm2[1] * h)

        # Canonical UVs (fixed)
        uv0 = vertex_uvs[i0]
        uv1 = vertex_uvs[i1]
        uv2 = vertex_uvs[i2]

        if (np.any(np.isnan(uv0)) or
            np.any(np.isnan(uv1)) or
            np.any(np.isnan(uv2))):
            continue

        draw_face_triangle_to_uv(frame_bgr, p0, p1, p2, uv0, uv1, uv2, tex)

    return tex


# Start webcam
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,           # Detect only 1 face
    refine_landmarks=False,     # Include iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.8
) as face_mesh:
    
    vertices = np.zeros((468, 3), dtype=np.float32)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process the frame
        results = face_mesh.process(rgb_frame)
        
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]  # single face
            # Efficiently convert to ndarray
            vertices[:] = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark], dtype=np.float32)

            if vertices.shape[0] != verts_len:
                raise ValueError(
                    f"Runtime vertex count {vertices.shape[0]} "
                    f"!= canonical OBJ vertex count {verts_len}"
                )
                
        # Build UV texture by sampling from the frame
        uv_tex = build_canonical_face_uv_map(frame, vertices)

        # Show both the webcam and the UV texture
        cv2.imshow("Frame", frame)
        cv2.imshow("Face UV Texture", uv_tex)

        if cv2.waitKey(1) & 0xFF == 27:
            break
            
        else:
            continue

cap.release()
cv2.destroyAllWindows()