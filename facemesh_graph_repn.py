import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh


def get_exclusion_landmarks():
    """Get landmark indices for eyes and mouth to exclude - MINIMAL set."""
    
    # Left eye - just the core eye region
    left_eye = set([
        # Eye outline only
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        # Iris
        468, 469, 470, 471, 472
    ])
    
    # Right eye - just the core eye region
    right_eye = set([
        # Eye outline only
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
        # Iris
        473, 474, 475, 476, 477
    ])
    
    # Mouth - just lips
    mouth = set([
        # Outer lips only
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
        # Inner lips only
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191
    ])
    
    return left_eye | right_eye | mouth


def plot_face_mesh_graph(image_path):
    """
    Plot face mesh graph from image.
    
    Parameters:
    -----------
    image_path : str
        Path to face image
    """
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Get face mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            raise ValueError("No face detected in image!")
        
        # Get landmarks
        landmarks = results.multi_face_landmarks[0]
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Flip Y to make face upright
        landmarks_array[:, 1] = 1.0 - landmarks_array[:, 1]
    
    # Get exclusions
    exclusions = get_exclusion_landmarks()
    
    # Get ALL triangles from tesselation
    from collections import defaultdict
    adjacency = defaultdict(set)
    
    for v1, v2 in mp_face_mesh.FACEMESH_TESSELATION:
        adjacency[v1].add(v2)
        adjacency[v2].add(v1)
    
    # Find all triangles
    all_triangles = set()
    for v1, v2 in mp_face_mesh.FACEMESH_TESSELATION:
        common = adjacency[v1] & adjacency[v2]
        for v3 in common:
            tri = tuple(sorted([v1, v2, v3]))
            all_triangles.add(tri)
    
    all_triangles = list(all_triangles)
    
    # Filter out excluded triangles
    kept_triangles = []
    for v1, v2, v3 in all_triangles:
        # Keep triangle if NONE of its vertices are in exclusion zones
        if v1 not in exclusions and v2 not in exclusions and v3 not in exclusions:
            kept_triangles.append((v1, v2, v3))
    
    print(f"Total triangles: {len(all_triangles)}")
    print(f"Kept triangles: {len(kept_triangles)}")
    print(f"Excluded triangles: {len(all_triangles) - len(kept_triangles)}")
    
    # Compute centroids
    centroids = []
    triangle_verts = []
    
    for v1, v2, v3 in kept_triangles:
        verts = landmarks_array[[v1, v2, v3]]
        centroid = verts.mean(axis=0)
        centroids.append(centroid)
        triangle_verts.append(verts)
    
    centroids = np.array(centroids)
    
    # Build adjacency between triangles
    edges = []
    n = len(kept_triangles)
    
    for i in range(n):
        tri_i = set(kept_triangles[i])
        for j in range(i + 1, n):
            tri_j = set(kept_triangles[j])
            # Adjacent if they share exactly 2 vertices (an edge)
            if len(tri_i & tri_j) == 2:
                edges.append((i, j))
    
    print(f"Graph edges: {len(edges)}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 14))
    
    # Draw triangles
    for verts in triangle_verts:
        poly = Polygon(verts[:, :2], facecolor='lightblue', 
                      edgecolor='gray', alpha=0.2, linewidth=0.3)
        ax.add_patch(poly)
    
    # Draw edges between centroids
    edge_lines = [[centroids[i, :2], centroids[j, :2]] for i, j in edges]
    lc = LineCollection(edge_lines, colors='darkblue', linewidths=0.8, alpha=0.7, label='Adjacency Edges')
    ax.add_collection(lc)
    
    # Draw centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=15, 
              zorder=10, alpha=0.9, label='Triangle Centroids')
    
    # Add dummy patch for triangles legend
    from matplotlib.patches import Patch
    triangle_patch = Patch(facecolor='lightblue', edgecolor='gray', alpha=0.2, label='Face Mesh Triangles')
    ax.legend(handles=[triangle_patch, 
                      plt.Line2D([0], [0], color='darkblue', linewidth=0.8, alpha=0.7, label='Adjacency Edges'),
                      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Triangle Centroids')],
             fontsize=10, loc='upper right')
    
    ax.set_aspect('equal')
    # ax.set_xlabel('X coordinate', fontsize=12)
    # ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title('Face Mesh Topology Graph (Excluding Eyes and Mouth)\n' + 
                'Vertices = Triangle Centroids, Edges = Adjacent Triangles', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    # ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # CHANGE THIS to your face image path
    image_path = 'sample_face_image.png'
    
    fig, ax = plot_face_mesh_graph(image_path)
    plt.savefig('face_mesh_graph_no_eyes_mouth.png', dpi=300, bbox_inches='tight')
    plt.show()