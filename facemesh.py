# face_mesh.py
import mediapipe as mp
import numpy as np

class FaceMeshEstimator:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, frame_rgb):
        res = self.mp_face.process(frame_rgb)
        if not res.multi_face_landmarks:
            return None, False

        lm = res.multi_face_landmarks[0].landmark
        verts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        return verts, True
