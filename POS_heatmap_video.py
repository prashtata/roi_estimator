import numpy as np
import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy import signal
from collections import defaultdict
import time
from imseqprocessor import ImageDirCapture


class FaceMeshRPPGOffline:
    """
    Process entire video to extract RGB signals and compute SNR for face mesh regions.
    """
    
    def __init__(self, n_samples=100):
        """
        Initialize the face mesh rPPG processor.
        
        Args:
            n_samples: Number of Monte Carlo samples per triangle per frame
        """
        self.n_samples = n_samples
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Storage for RGB streams
        self.rgb_streams = defaultdict(list)
        self.triangles = None
        self.frame_count = 0
        self.landmarks_avg = None  # Average landmark positions for visualization
        
        # Define regions to exclude (eyes and lips)
        # MediaPipe face mesh landmark indices
        self.excluded_landmarks = self.get_excluded_landmarks()
        
    def get_excluded_landmarks(self):
        """
        Get landmark indices for eyes and lips to exclude from analysis.
        Uses precise MediaPipe face mesh indices.
        
        Returns:
            excluded_set: Set of landmark indices to exclude
        """
        # Left eye contour (more precise)
        left_eye = {
            # Upper eyelid
            33, 246, 161, 160, 159, 158, 157, 173,
            # Lower eyelid  
            133, 155, 154, 153, 145, 144, 163, 7,
            # Inner eye region
            130, 247, 30, 29, 27, 28, 56, 190,
            # Eyebrow region (optional - comment out if you want to include)
            246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7
        }
        
        # Right eye contour (more precise)
        right_eye = {
            # Upper eyelid
            263, 466, 388, 387, 386, 385, 384, 398,
            # Lower eyelid
            362, 382, 381, 380, 374, 373, 390, 249,
            # Inner eye region
            359, 467, 260, 259, 257, 258, 286, 414,
            # Eyebrow region (optional - comment out if you want to include)
            466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249
        }
        
        # Outer lip contour
        lips_outer = {
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 375, 321, 405, 314, 17, 84, 181, 91, 146
        }
        
        # Inner lip contour
        lips_inner = {
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            308, 324, 318, 402, 317, 14, 87, 178, 88, 95
        }
        
        # Additional inner mouth cavity landmarks to exclude
        mouth_interior = {
            62, 96, 89, 179, 86, 316, 403, 319, 325, 292,
            # Teeth/tongue region
            76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306, 408
        }
        
        # Combine all excluded regions
        excluded = left_eye | right_eye | lips_outer | lips_inner | mouth_interior
        
        return excluded
    
    def is_triangle_excluded(self, triangle):
        """
        Check if a triangle should be excluded based on its vertices.
        
        Args:
            triangle: Tuple of (v0, v1, v2) vertex indices
        
        Returns:
            excluded: Boolean indicating if triangle should be excluded
        """
        v0, v1, v2 = triangle
        
        # Exclude if any vertex is in the excluded landmarks
        if v0 in self.excluded_landmarks or v1 in self.excluded_landmarks or v2 in self.excluded_landmarks:
            return True
        
        return False
    
    def extract_face_landmarks(self, frame):
        """
        Extract facial landmarks from a frame.
        
        Args:
            frame: RGB frame (numpy array)
        
        Returns:
            landmarks: Array of (x, y, z) coordinates normalized to [0, 1]
            connections: MediaPipe face mesh connections
        """
        results = self.face_mesh.process(frame)
        
        if not results.multi_face_landmarks:
            return None, None
        
        face_landmarks = results.multi_face_landmarks[0]
        
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        
        return np.array(landmarks), self.mp_face_mesh.FACEMESH_TESSELATION
    
    def get_triangles_from_connections(self, connections):
        """
        Convert MediaPipe connections to triangle list, excluding eyes and lips.
        
        Args:
            connections: MediaPipe face mesh connections
        
        Returns:
            triangles: List of triangle vertex indices (excluding eyes/lips)
        """
        connection_dict = {}
        
        for start_idx, end_idx in connections:
            if start_idx not in connection_dict:
                connection_dict[start_idx] = set()
            if end_idx not in connection_dict:
                connection_dict[end_idx] = set()
            connection_dict[start_idx].add(end_idx)
            connection_dict[end_idx].add(start_idx)
        
        triangles = set()
        for i in connection_dict:
            for j in connection_dict[i]:
                if j > i:
                    for k in connection_dict[j]:
                        if k > j and k in connection_dict[i]:
                            tri = tuple(sorted([i, j, k]))
                            # Only add if not excluded
                            if not self.is_triangle_excluded(tri):
                                triangles.add(tri)
        
        return list(triangles)
    
    def sample_polygon_rgb(self, frame, landmarks, connections):
        """
        Monte Carlo sample pixels from each triangle and compute mean RGB.
        
        Args:
            frame: RGB frame (numpy array)
            landmarks: Array of (x, y, z) coordinates normalized to [0, 1]
            connections: MediaPipe face mesh connections
        
        Returns:
            mean_rgb_list: List of mean RGB values for each polygon region
        """
        if landmarks is None:
            return None
        
        h, w = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        pixel_coords = landmarks.copy()
        pixel_coords[:, 0] *= w
        pixel_coords[:, 1] *= h
        
        # Get triangles (only compute once)
        if self.triangles is None:
            self.triangles = self.get_triangles_from_connections(connections)
            print(f"Using {len(self.triangles)} triangles (eyes and lips excluded)")
        
        mean_rgb_list = []
        
        for tri in self.triangles:
            v0, v1, v2 = tri
            
            # Get triangle vertices in pixel coordinates
            p0 = np.array([pixel_coords[v0][0], pixel_coords[v0][1]])
            p1 = np.array([pixel_coords[v1][0], pixel_coords[v1][1]])
            p2 = np.array([pixel_coords[v2][0], pixel_coords[v2][1]])
            
            # Monte Carlo sampling using barycentric coordinates
            r1 = np.random.random(self.n_samples)
            r2 = np.random.random(self.n_samples)
            
            sqrt_r1 = np.sqrt(r1)
            u = 1 - sqrt_r1
            v = sqrt_r1 * (1 - r2)
            w_bary = sqrt_r1 * r2
            
            # Compute sampled points
            sampled_points = u[:, None] * p0 + v[:, None] * p1 + w_bary[:, None] * p2
            
            # Extract RGB values
            rgb_values = []
            for point in sampled_points:
                x = int(round(point[0]))
                y = int(round(point[1]))
                
                if 0 <= x < w and 0 <= y < h:
                    rgb_values.append(frame[y, x])
            
            if rgb_values:
                mean_rgb = np.mean(rgb_values, axis=0)
                mean_rgb_list.append(mean_rgb)
            else:
                mean_rgb_list.append(np.array([np.nan, np.nan, np.nan]))
        
        return mean_rgb_list
    
    def process_frame(self, frame):
        """
        Process a single video frame and accumulate RGB values.
        
        Args:
            frame: RGB frame (numpy array)
        
        Returns:
            landmarks: Extracted landmarks
            success: Boolean indicating if face was detected
        """
        landmarks, connections = self.extract_face_landmarks(frame)
        
        if landmarks is None:
            return None, False
        
        # Accumulate landmarks for averaging
        if self.landmarks_avg is None:
            self.landmarks_avg = landmarks.copy()
        else:
            # Running average of landmark positions
            alpha = 0.9
            self.landmarks_avg = alpha * self.landmarks_avg + (1 - alpha) * landmarks
        
        # Sample RGB from polygons
        mean_rgb_values = self.sample_polygon_rgb(frame, landmarks, connections)
        
        if mean_rgb_values is None:
            return landmarks, False
        
        # Accumulate RGB values for each triangle
        for tri_idx, rgb_value in enumerate(mean_rgb_values):
            self.rgb_streams[tri_idx].append(rgb_value)
        
        self.frame_count += 1
        
        return landmarks, True
    
    def get_rgb_streams(self):
        """
        Get the accumulated RGB streams for all polygon regions.
        
        Returns:
            rgb_streams: Dictionary {triangle_idx: numpy array of shape (n_frames, 3)}
        """
        return {tri_idx: np.array(values) for tri_idx, values in self.rgb_streams.items()}
    
    def compute_pos_signal(self, rgb_stream, fps=30):
        """
        Compute POS (Plane-Orthogonal-to-Skin) rPPG signal from RGB stream.
        """
        if np.any(np.isnan(rgb_stream)):
            return np.full(len(rgb_stream), np.nan)
        
        # Normalize each channel
        rgb_norm = np.zeros_like(rgb_stream, dtype=np.float64)
        for i in range(3):
            mean_val = np.mean(rgb_stream[:, i])
            if mean_val > 0:
                rgb_norm[:, i] = rgb_stream[:, i] / mean_val
            else:
                return np.full(len(rgb_stream), np.nan)
        
        # POS algorithm
        X_s = rgb_norm[:, 0] - rgb_norm[:, 1]
        Y_s = rgb_norm[:, 0] + rgb_norm[:, 1] - 2 * rgb_norm[:, 2]
        
        window_size = int(fps * 1.6)
        if window_size < 2:
            window_size = 2
        
        pos_signal = np.zeros(len(rgb_stream))
        
        for t in range(len(rgb_stream)):
            start_idx = max(0, t - window_size + 1)
            end_idx = t + 1
            
            X_window = X_s[start_idx:end_idx]
            Y_window = Y_s[start_idx:end_idx]
            
            X_mean = np.mean(X_window)
            X_std = np.std(X_window)
            Y_mean = np.mean(Y_window)
            Y_std = np.std(Y_window)
            
            if X_std > 0:
                X_norm = (X_window[-1] - X_mean) / X_std
            else:
                X_norm = 0
            
            if Y_std > 0:
                Y_norm = (Y_window[-1] - Y_mean) / Y_std
            else:
                Y_norm = 0
            
            alpha = X_std / Y_std if Y_std > 0 else 1.0
            pos_signal[t] = X_norm - alpha * Y_norm
        
        pos_signal = self.bandpass_filter(pos_signal, fps, low_freq=0.7, high_freq=4.0)
        
        return pos_signal
    
    def bandpass_filter(self, signal_data, fps, low_freq=0.7, high_freq=4.0, order=4):
        """
        Apply bandpass filter to signal.
        """
        nyquist = fps / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        low = max(0.01, min(low, 0.99))
        high = max(0.01, min(high, 0.99))
        
        if low >= high:
            return signal_data
        
        try:
            b, a = signal.butter(order, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, signal_data)
            return filtered_signal
        except:
            return signal_data
    
    def compute_snr(self, pos_signal, fps=30, hr_range=(0.7, 4.0)):
        """
        Compute Signal-to-Noise Ratio (SNR) for a POS signal.
        """
        if np.any(np.isnan(pos_signal)) or len(pos_signal) < fps:
            return np.nan, np.nan, np.nan
        
        nperseg = min(len(pos_signal), int(fps * 8))
        
        try:
            freqs, psd = signal.welch(pos_signal, fs=fps, nperseg=nperseg, 
                                       scaling='density', detrend='constant')
        except:
            return np.nan, np.nan, np.nan
        
        hr_min, hr_max = hr_range
        
        hr_band_idx = np.where((freqs >= hr_min) & (freqs <= hr_max))[0]
        noise_band_idx = np.where(((freqs >= 0.1) & (freqs < hr_min)) | 
                                  ((freqs > hr_max) & (freqs <= 5.0)))[0]
        
        if len(hr_band_idx) == 0 or len(noise_band_idx) == 0:
            return np.nan, np.nan, np.nan
        
        signal_power = np.trapz(psd[hr_band_idx], freqs[hr_band_idx])
        noise_power = np.trapz(psd[noise_band_idx], freqs[noise_band_idx])
        
        if noise_power <= 0 or signal_power <= 0:
            return np.nan, np.nan, np.nan
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        peak_idx = hr_band_idx[np.argmax(psd[hr_band_idx])]
        peak_freq = freqs[peak_idx]
        peak_bpm = peak_freq * 60
        
        return snr_db, peak_freq, peak_bpm
    
    def compute_all_snr(self, fps=30):
        """
        Compute SNR for all polygon regions.
        
        Returns:
            snr_values: Dictionary {triangle_idx: (snr_db, peak_freq, peak_bpm)}
            pos_signals: Dictionary {triangle_idx: POS signal array}
        """
        rgb_streams = self.get_rgb_streams()
        snr_values = {}
        pos_signals = {}
        
        print(f"\nComputing POS signals and SNR for {len(rgb_streams)} regions...")
        
        valid_count = 0
        for tri_idx, rgb_stream in rgb_streams.items():
            # Compute POS signal
            pos_signal = self.compute_pos_signal(rgb_stream, fps)
            pos_signals[tri_idx] = pos_signal
            
            # Compute SNR
            snr_db, peak_freq, peak_bpm = self.compute_snr(pos_signal, fps)
            snr_values[tri_idx] = (snr_db, peak_freq, peak_bpm)
            
            if not np.isnan(snr_db):
                valid_count += 1
            
            if (tri_idx + 1) % 50 == 0:
                print(f"  Processed {tri_idx + 1}/{len(rgb_streams)} regions")
        
        print(f"SNR computation complete! Valid SNR values: {valid_count}/{len(rgb_streams)}")
        return snr_values, pos_signals
    
    def plot_snr_heatmap_face(self, snr_values, save_path='snr_heatmap_face.png', 
                            colormap='jet', figsize=(14, 12), show_edges=True):
        """
        Plot SNR heatmap in the shape of a face using the actual face mesh geometry.
        
        Args:
            snr_values: Dictionary {triangle_idx: (snr_db, peak_freq, peak_bpm)}
            save_path: Path to save the heatmap image
            colormap: Matplotlib colormap name
            figsize: Figure size tuple
            show_edges: Whether to show triangle edges
        """
        if self.landmarks_avg is None or self.triangles is None:
            print("No landmark data available for plotting")
            return
        
        # Get valid SNR values
        valid_snrs = [snr for snr, _, _ in snr_values.values() if not np.isnan(snr)]
        
        if len(valid_snrs) == 0:
            print("No valid SNR values to plot")
            return
        
        print(f"\nPlotting heatmap with {len(self.triangles)} triangles")
        print(f"Valid SNR values: {len(valid_snrs)}")
        
        # Normalize SNR values
        min_snr = np.percentile(valid_snrs, 5)
        max_snr = np.percentile(valid_snrs, 95)
        
        print(f"SNR range: {min_snr:.2f} to {max_snr:.2f} dB")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use normalized coordinates (0-1 range)
        landmarks_2d = self.landmarks_avg[:, :2]
        
        # Get colormap
        cmap = cm.get_cmap(colormap)
        
        # Create polygon patches for each triangle
        patches = []
        colors = []
        edge_colors = []
        
        plotted_count = 0
        skipped_count = 0
        
        for tri_idx, tri in enumerate(self.triangles):
            snr_value = snr_values.get(tri_idx, (np.nan, np.nan, np.nan))[0]
            
            if np.isnan(snr_value):
                skipped_count += 1
                continue
            
            v0, v1, v2 = tri
            
            # Get triangle vertices
            vertices = np.array([
                landmarks_2d[v0],
                landmarks_2d[v1],
                landmarks_2d[v2]
            ])
            
            # Create polygon
            poly = Polygon(vertices, closed=True)
            patches.append(poly)
            
            # Normalize SNR for coloring
            normalized_snr = (snr_value - min_snr) / (max_snr - min_snr) if max_snr > min_snr else 0.5
            normalized_snr = np.clip(normalized_snr, 0, 1)
            colors.append(normalized_snr)
            
            # Edge color - white or none
            edge_colors.append('white' if show_edges else 'none')
            plotted_count += 1
        
        print(f"Plotted triangles: {plotted_count}")
        print(f"Skipped triangles (NaN SNR): {skipped_count}")
        
        # Create patch collection
        p = PatchCollection(patches, cmap=colormap, alpha=0.9, edgecolors=edge_colors, linewidths=0.3)
        p.set_array(np.array(colors))
        p.set_clim([0, 1])
        
        ax.add_collection(p)
        
        # Optionally plot all landmarks to verify coverage
        # ax.scatter(landmarks_2d[:, 0], landmarks_2d[:, 1], c='black', s=1, alpha=0.3, zorder=10)
        
        # Set axis properties
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.invert_yaxis()  # Invert y-axis to match image coordinates
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add colorbar
        sm = cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min_snr, vmax=max_snr))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SNR (dB)', rotation=270, labelpad=20, fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        # Add title with statistics
        mean_snr = np.mean(valid_snrs)
        median_snr = np.median(valid_snrs)
        plt.title(f'Face Mesh SNR Heatmap (Eyes & Lips Excluded)\n'
                f'Mean SNR: {mean_snr:.2f} dB | Median SNR: {median_snr:.2f} dB | '
                f'Regions: {len(valid_snrs)}',
                fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved SNR heatmap to {save_path}")
        plt.show()
        
        
    def load_obj_uv_mapping(self, obj_path):
        """
        Load UV coordinates and face mappings from a MediaPipe OBJ file.
        
        Args:
            obj_path: Path to the OBJ file
        
        Returns:
            uv_coords: List of (u, v) UV coordinates
            uv_faces: List of triangles as indices into uv_coords
            vertex_to_uv: Mapping from vertex index to UV coordinate index
        """
        uv_coords = []
        faces_3d = []  # Vertex indices
        faces_uv = []  # UV indices
        
        print(f"Loading OBJ file: {obj_path}")
        
        with open(obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # UV coordinates (vt u v)
                if line.startswith('vt '):
                    parts = line.split()
                    u = float(parts[1])
                    v = float(parts[2])
                    uv_coords.append([u, v])
                
                # Faces (f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3)
                elif line.startswith('f '):
                    parts = line.split()
                    face_v = []
                    face_vt = []
                    
                    for i in range(1, 4):  # Triangles have 3 vertices
                        indices = parts[i].split('/')
                        v_idx = int(indices[0]) - 1  # OBJ is 1-indexed
                        vt_idx = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else None
                        
                        face_v.append(v_idx)
                        if vt_idx is not None:
                            face_vt.append(vt_idx)
                    
                    if len(face_v) == 3:
                        faces_3d.append(tuple(face_v))
                    if len(face_vt) == 3:
                        faces_uv.append(tuple(face_vt))
        
        print(f"Loaded {len(uv_coords)} UV coordinates")
        print(f"Loaded {len(faces_3d)} 3D faces")
        print(f"Loaded {len(faces_uv)} UV faces")
        
        # Create mapping from 3D vertex index to UV indices
        vertex_to_uv = {}
        for face_3d, face_uv in zip(faces_3d, faces_uv):
            for v_idx, uv_idx in zip(face_3d, face_uv):
                if v_idx not in vertex_to_uv:
                    vertex_to_uv[v_idx] = []
                if uv_idx not in vertex_to_uv[v_idx]:
                    vertex_to_uv[v_idx].append(uv_idx)
        
        return np.array(uv_coords), faces_3d, faces_uv, vertex_to_uv


    def match_triangles_to_uv_faces(self, faces_3d, faces_uv):
        """
        Match the triangles from face mesh to UV faces from OBJ.
        
        Args:
            faces_3d: List of 3D face vertex indices from OBJ
            faces_uv: List of UV face indices from OBJ
        
        Returns:
            triangle_to_uv_face: Dictionary mapping triangle index to UV face index
        """
        # Create a set of 3D faces for quick lookup
        face_3d_set = {tuple(sorted(face)): idx for idx, face in enumerate(faces_3d)}
        
        triangle_to_uv_face = {}
        
        for tri_idx, tri in enumerate(self.triangles):
            sorted_tri = tuple(sorted(tri))
            if sorted_tri in face_3d_set:
                uv_face_idx = face_3d_set[sorted_tri]
                triangle_to_uv_face[tri_idx] = uv_face_idx
        
        print(f"Matched {len(triangle_to_uv_face)} triangles to UV faces")
        
        return triangle_to_uv_face


    def plot_snr_heatmap_uv_from_obj(self, snr_values, obj_path, 
                                    save_path='snr_heatmap_uv_obj.png',
                                    colormap='jet', figsize=(16, 16),
                                    show_edges=True):
        """
        Plot SNR heatmap using UV coordinates from MediaPipe OBJ file.
        
        Args:
            snr_values: Dictionary {triangle_idx: (snr_db, peak_freq, peak_bpm)}
            obj_path: Path to MediaPipe OBJ file with UV mapping
            save_path: Path to save the heatmap image
            colormap: Matplotlib colormap name
            figsize: Figure size tuple
            show_edges: Whether to show triangle edges
        """
        if self.triangles is None:
            print("No triangle data available for plotting")
            return
        
        # Load UV mapping from OBJ file
        uv_coords, faces_3d, faces_uv, vertex_to_uv = self.load_obj_uv_mapping(obj_path)
        
        # Match our triangles to UV faces
        triangle_to_uv_face = self.match_triangles_to_uv_faces(faces_3d, faces_uv)
        
        # Get valid SNR values
        valid_snrs = [snr for snr, _, _ in snr_values.values() if not np.isnan(snr)]
        
        if len(valid_snrs) == 0:
            print("No valid SNR values to plot")
            return
        
        print(f"\nCreating UV-map heatmap from OBJ file")
        
        # Normalize SNR values
        min_snr = np.percentile(valid_snrs, 5)
        max_snr = np.percentile(valid_snrs, 95)
        
        print(f"SNR range: {min_snr:.2f} to {max_snr:.2f} dB")
        
        # Get colormap
        cmap = cm.get_cmap(colormap)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        patches = []
        colors = []
        
        plotted_count = 0
        skipped_no_match = 0
        skipped_nan = 0
        
        for tri_idx in range(len(self.triangles)):
            # Get SNR value
            snr_value = snr_values.get(tri_idx, (np.nan, np.nan, np.nan))[0]
            
            if np.isnan(snr_value):
                skipped_nan += 1
                continue
            
            # Get corresponding UV face
            if tri_idx not in triangle_to_uv_face:
                skipped_no_match += 1
                continue
            
            uv_face_idx = triangle_to_uv_face[tri_idx]
            uv_face = faces_uv[uv_face_idx]
            
            # Get UV coordinates for this triangle
            uv0, uv1, uv2 = uv_face
            triangle_uv_coords = np.array([
                uv_coords[uv0],
                uv_coords[uv1],
                uv_coords[uv2]
            ])
            
            # Create polygon
            poly = Polygon(triangle_uv_coords, closed=True)
            patches.append(poly)
            
            # Normalize SNR for coloring
            normalized_snr = (snr_value - min_snr) / (max_snr - min_snr) if max_snr > min_snr else 0.5
            normalized_snr = np.clip(normalized_snr, 0, 1)
            colors.append(normalized_snr)
            
            plotted_count += 1
        
        print(f"Plotted triangles: {plotted_count}")
        print(f"Skipped (no UV match): {skipped_no_match}")
        print(f"Skipped (NaN SNR): {skipped_nan}")
        
        # Create patch collection
        edge_color = 'white' if show_edges else 'none'
        edge_width = 0.2 if show_edges else 0
        
        p = PatchCollection(patches, cmap=colormap, alpha=0.95, 
                        edgecolors=edge_color, linewidths=edge_width)
        p.set_array(np.array(colors))
        p.set_clim([0, 1])
        
        ax.add_collection(p)
        
        # Set axis properties for UV space (typically 0-1)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add colorbar
        sm = cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min_snr, vmax=max_snr))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SNR (dB)', rotation=270, labelpad=20, fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        # Add title
        mean_snr = np.mean(valid_snrs)
        median_snr = np.median(valid_snrs)
        plt.title(f'SNR Heatmap - UV Map from OBJ\n'
                f'Mean SNR: {mean_snr:.2f} dB | Median SNR: {median_snr:.2f} dB | '
                f'Triangles: {plotted_count}',
                fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved UV-map heatmap to {save_path}")
        plt.show()


    def plot_snr_heatmap_uv_texture(self, snr_values, obj_path,
                                    save_path='snr_heatmap_texture.png',
                                    colormap='jet', texture_size=2048):
        """
        Create a texture map image of SNR values that can be applied to the 3D mesh.
        
        Args:
            snr_values: Dictionary {triangle_idx: (snr_db, peak_freq, peak_bpm)}
            obj_path: Path to MediaPipe OBJ file with UV mapping
            save_path: Path to save the texture image
            colormap: Matplotlib colormap name
            texture_size: Size of the output texture (width and height in pixels)
        """
        if self.triangles is None:
            print("No triangle data available for plotting")
            return
        
        # Load UV mapping from OBJ file
        uv_coords, faces_3d, faces_uv, vertex_to_uv = self.load_obj_uv_mapping(obj_path)
        
        # Match our triangles to UV faces
        triangle_to_uv_face = self.match_triangles_to_uv_faces(faces_3d, faces_uv)
        
        # Get valid SNR values
        valid_snrs = [snr for snr, _, _ in snr_values.values() if not np.isnan(snr)]
        
        if len(valid_snrs) == 0:
            print("No valid SNR values to plot")
            return
        
        print(f"\nCreating texture map ({texture_size}x{texture_size})")
        
        # Normalize SNR values
        min_snr = np.percentile(valid_snrs, 5)
        max_snr = np.percentile(valid_snrs, 95)
        
        # Get colormap
        cmap = cm.get_cmap(colormap)
        
        # Create texture image
        texture = np.ones((texture_size, texture_size, 4)) * 255  # RGBA, white background
        
        plotted_count = 0
        
        for tri_idx in range(len(self.triangles)):
            # Get SNR value
            snr_value = snr_values.get(tri_idx, (np.nan, np.nan, np.nan))[0]
            
            if np.isnan(snr_value):
                continue
            
            # Get corresponding UV face
            if tri_idx not in triangle_to_uv_face:
                continue
            
            uv_face_idx = triangle_to_uv_face[tri_idx]
            uv_face = faces_uv[uv_face_idx]
            
            # Get UV coordinates for this triangle
            uv0, uv1, uv2 = uv_face
            triangle_uv_coords = np.array([
                uv_coords[uv0],
                uv_coords[uv1],
                uv_coords[uv2]
            ])
            
            # Convert UV coordinates to pixel coordinates
            # UV space is typically (0,0) at bottom-left, but image is top-left
            pixel_coords = triangle_uv_coords.copy()
            pixel_coords[:, 0] *= texture_size  # U -> X
            pixel_coords[:, 1] = (1 - pixel_coords[:, 1]) * texture_size  # V -> Y (flip)
            pixel_coords = pixel_coords.astype(np.int32)
            
            # Normalize SNR for coloring
            normalized_snr = (snr_value - min_snr) / (max_snr - min_snr) if max_snr > min_snr else 0.5
            normalized_snr = np.clip(normalized_snr, 0, 1)
            
            # Get color from colormap
            color_rgba = cmap(normalized_snr)
            color_bgr = (int(color_rgba[2] * 255), 
                        int(color_rgba[1] * 255), 
                        int(color_rgba[0] * 255))
            
            # Draw filled triangle on texture
            cv2.fillPoly(texture, [pixel_coords], (*color_bgr, 255))
            
            plotted_count += 1
        
        print(f"Plotted {plotted_count} triangles on texture")
        
        # Convert to BGR for saving (OpenCV format)
        texture_bgr = texture[:, :, :3].astype(np.uint8)
        
        # Save texture
        cv2.imwrite(save_path, texture_bgr)
        print(f"Saved texture map to {save_path}")
        
        # Also create a figure for display
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Convert BGR to RGB for matplotlib
        texture_rgb = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(texture_rgb)
        ax.axis('off')
        
        # Add colorbar
        sm = cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min_snr, vmax=max_snr))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SNR (dB)', rotation=270, labelpad=20, fontsize=14)
        
        mean_snr = np.mean(valid_snrs)
        median_snr = np.median(valid_snrs)
        plt.title(f'SNR Texture Map ({texture_size}x{texture_size})\n'
                f'Mean SNR: {mean_snr:.2f} dB | Median SNR: {median_snr:.2f} dB',
                fontsize=16, pad=20)
        
        plt.tight_layout()
        display_path = save_path.replace('.png', '_display.png')
        plt.savefig(display_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved texture display to {display_path}")
        plt.show()
        
        
    def plot_snr_heatmap_uv_simple(self, snr_values, obj_path,
                                save_path='snr_heatmap_uv.png',
                                colormap='jet', figsize=(16, 16)):
        """
        Simple UV heatmap plot - no checks, just plot it.
        """
        # Load UV coords from OBJ
        uv_coords = []
        vertex_to_uv = {}
        
        with open(obj_path, 'r') as f:
            uv_idx = 0
            vertex_idx = 0
            for line in f:
                if line.startswith('vt '):
                    parts = line.split()
                    uv_coords.append([float(parts[1]), float(parts[2])])
                elif line.startswith('f '):
                    parts = line.split()[1:]
                    for p in parts:
                        v, vt = p.split('/')[:2]
                        v_idx = int(v) - 1
                        vt_idx = int(vt) - 1
                        if v_idx not in vertex_to_uv:
                            vertex_to_uv[v_idx] = vt_idx
        
        uv_coords = np.array(uv_coords)
        
        # Get SNR range
        valid_snrs = [snr for snr, _, _ in snr_values.values() if not np.isnan(snr)]
        min_snr = np.percentile(valid_snrs, 5)
        max_snr = np.percentile(valid_snrs, 95)
        cmap = cm.get_cmap(colormap)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        patches = []
        colors = []
        
        for tri_idx, tri in enumerate(self.triangles):
            snr_value = snr_values.get(tri_idx, (np.nan, np.nan, np.nan))[0]
            if np.isnan(snr_value):
                continue
            
            v0, v1, v2 = tri
            if v0 in vertex_to_uv and v1 in vertex_to_uv and v2 in vertex_to_uv:
                uv_tri = np.array([
                    uv_coords[vertex_to_uv[v0]],
                    uv_coords[vertex_to_uv[v1]],
                    uv_coords[vertex_to_uv[v2]]
                ])
                
                patches.append(Polygon(uv_tri, closed=True))
                normalized_snr = np.clip((snr_value - min_snr) / (max_snr - min_snr), 0, 1)
                colors.append(normalized_snr)
        
        p = PatchCollection(patches, cmap=colormap, alpha=0.95, edgecolors='white', linewidths=0.2)
        p.set_array(np.array(colors))
        p.set_clim([0, 1])
        ax.add_collection(p)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        sm = cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min_snr, vmax=max_snr))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SNR (dB)', rotation=270, labelpad=20, fontsize=14)
        
        plt.title(f'SNR UV Heatmap\nMean: {np.mean(valid_snrs):.2f} dB | Median: {np.median(valid_snrs):.2f} dB',
                fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
         
    def plot_snr_statistics(self, snr_values, save_path='snr_statistics.png'):
        """
        Plot SNR distribution and statistics.
        
        Args:
            snr_values: Dictionary {triangle_idx: (snr_db, peak_freq, peak_bpm)}
            save_path: Path to save the statistics plot
        """
        valid_snrs = [snr for snr, _, _ in snr_values.values() if not np.isnan(snr)]
        valid_bpms = [bpm for _, _, bpm in snr_values.values() if not np.isnan(bpm)]
        
        if len(valid_snrs) == 0:
            print("No valid SNR values to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # SNR histogram
        axes[0, 0].hist(valid_snrs, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].axvline(np.mean(valid_snrs), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(valid_snrs):.2f} dB')
        axes[0, 0].axvline(np.median(valid_snrs), color='green', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(valid_snrs):.2f} dB')
        axes[0, 0].set_xlabel('SNR (dB)', fontsize=11)
        axes[0, 0].set_ylabel('Number of Regions', fontsize=11)
        axes[0, 0].set_title('SNR Distribution', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # BPM histogram
        axes[0, 1].hist(valid_bpms, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
        axes[0, 1].axvline(np.mean(valid_bpms), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(valid_bpms):.1f} BPM')
        axes[0, 1].axvline(np.median(valid_bpms), color='green', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(valid_bpms):.1f} BPM')
        axes[0, 1].set_xlabel('Heart Rate (BPM)', fontsize=11)
        axes[0, 1].set_ylabel('Number of Regions', fontsize=11)
        axes[0, 1].set_title('Heart Rate Distribution', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # SNR box plot
        axes[1, 0].boxplot(valid_snrs, vert=True)
        axes[1, 0].set_ylabel('SNR (dB)', fontsize=11)
        axes[1, 0].set_title('SNR Box Plot', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Statistics text
        stats_text = f"""
        SNR Statistics:
        ─────────────────
        Valid Regions: {len(valid_snrs)}
        Mean SNR: {np.mean(valid_snrs):.2f} dB
        Median SNR: {np.median(valid_snrs):.2f} dB
        Std SNR: {np.std(valid_snrs):.2f} dB
        Max SNR: {np.max(valid_snrs):.2f} dB
        Min SNR: {np.min(valid_snrs):.2f} dB
        
        Heart Rate Statistics:
        ─────────────────────
        Mean HR: {np.mean(valid_bpms):.1f} BPM
        Median HR: {np.median(valid_bpms):.1f} BPM
        Std HR: {np.std(valid_bpms):.1f} BPM
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved statistics plot to {save_path}")
        plt.show()
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        self.face_mesh.close()
        
    def print_exclusion_diagnostics(self):
        """
        Print diagnostic information about excluded regions.
        """
        print("\n=== Exclusion Diagnostics ===")
        print(f"Total landmarks: 478")
        print(f"Excluded landmarks: {len(self.excluded_landmarks)}")
        print(f"Excluded landmarks: {sorted(self.excluded_landmarks)}")
        
        if self.triangles is not None:
            # Count how many triangles were found before exclusion
            # Re-compute triangles without exclusion for comparison
            mp_face_mesh = mp.solutions.face_mesh
            connections = mp_face_mesh.FACEMESH_TESSELATION
            
            connection_dict = {}
            for start_idx, end_idx in connections:
                if start_idx not in connection_dict:
                    connection_dict[start_idx] = set()
                if end_idx not in connection_dict:
                    connection_dict[end_idx] = set()
                connection_dict[start_idx].add(end_idx)
                connection_dict[end_idx].add(start_idx)
            
            all_triangles = set()
            for i in connection_dict:
                for j in connection_dict[i]:
                    if j > i:
                        for k in connection_dict[j]:
                            if k > j and k in connection_dict[i]:
                                all_triangles.add(tuple(sorted([i, j, k])))
            
            excluded_triangles = sum(1 for tri in all_triangles if self.is_triangle_excluded(tri))
            
            print(f"\nTotal possible triangles: {len(all_triangles)}")
            print(f"Excluded triangles: {excluded_triangles}")
            print(f"Included triangles: {len(self.triangles)}")
            print(f"Exclusion rate: {100*excluded_triangles/len(all_triangles):.1f}%")


def process_video_offline(video_path, n_samples=100, start = 0, end = None):
    """
    Process entire video and compute SNR for each region.
    
    Args:
        video_path: Path to video file
        n_samples: Number of Monte Carlo samples per triangle
    
    Returns:
        processor: FaceMeshRPPGOffline object with all data
        fps: Video frame rate
        snr_values: Dictionary of SNR values
    """
    processor = FaceMeshRPPGOffline(n_samples=n_samples)
    
    cap = cv2.VideoCapture(video_path)
    # cap = ImageDirCapture(video_path, start= start, end= end)
    
    if not cap.isOpened():
        print("Error opening video")
        return None, 30, None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0:
        fps = 30
    
    print(f"Video path: {video_path}")
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f} seconds")
    print("\nProcessing video...")
    
    frame_idx = 0
    successful_frames = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        landmarks, success = processor.process_frame(frame_rgb)
        
        if success:
            successful_frames += 1
        
        frame_idx += 1
        
        # Progress update
        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_idx / elapsed
            remaining = (total_frames - frame_idx) / fps_processing if fps_processing > 0 else 0
            print(f"  Processed {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%) | "
                  f"ETA: {remaining:.1f}s")
    
    cap.release()
    
    processing_time = time.time() - start_time
    
    print(f"\nVideo processing complete!")
    print(f"Frames processed: {frame_idx}")
    print(f"Successful frames: {successful_frames} ({100*successful_frames/frame_idx:.1f}%)")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Processing FPS: {frame_idx/processing_time:.1f}")
    
    # Compute SNR for all regions
    snr_values, pos_signals = processor.compute_all_snr(fps=fps)
    
    return processor, fps, snr_values


# if __name__ == "__main__":
#     # Process video file
#     video_path = "/home/daevinci/Datasets/DATASET_2/subject1/vid.avi"  # Replace with your video path
    
#     processor, fps, snr_values = process_video_offline(video_path, n_samples=100)
    
#     if processor is not None and snr_values is not None:
#         # Print exclusion diagnostics
#         processor.print_exclusion_diagnostics()
#         # Get statistics
#         valid_snrs = [snr for snr, _, _ in snr_values.values() if not np.isnan(snr)]
#         valid_bpms = [bpm for _, _, bpm in snr_values.values() if not np.isnan(bpm)]
        
#         print(f"\n=== Final SNR Statistics ===")
#         print(f"Valid regions: {len(valid_snrs)}/{len(snr_values)}")
#         print(f"Mean SNR: {np.mean(valid_snrs):.2f} dB")
#         print(f"Median SNR: {np.median(valid_snrs):.2f} dB")
#         print(f"Max SNR: {np.max(valid_snrs):.2f} dB")
#         print(f"Min SNR: {np.min(valid_snrs):.2f} dB")
#         print(f"\nMean Heart Rate: {np.mean(valid_bpms):.1f} BPM")
#         print(f"Median Heart Rate: {np.median(valid_bpms):.1f} BPM")
        
#         # Find best regions
#         best_regions = [(idx, snr, bpm) for idx, (snr, _, bpm) in snr_values.items() 
#                        if not np.isnan(snr)]
#         best_regions.sort(key=lambda x: x[1], reverse=True)
        
#         print(f"\n=== Top 10 Regions by SNR ===")
#         for rank, (tri_idx, snr_db, bpm) in enumerate(best_regions[:10], 1):
#             print(f"{rank}. Triangle {tri_idx}: SNR = {snr_db:.2f} dB, HR = {bpm:.1f} BPM")
        
#         # Plot face-shaped SNR heatmap
#         processor.plot_snr_heatmap_face(snr_values, save_path='snr_heatmap_face.png', 
#                                         colormap='jet')
        
#         # Plot statistics
#         processor.plot_snr_statistics(snr_values, save_path='snr_statistics.png')


if __name__ == "__main__":
    # Process video file
    video_path = "/home/daevinci/Datasets/DATASET_2/subject23/vid.avi"   # Replace with your video path
    # video_path = "/home/daevinci/Datasets/DATASET_2/PURE/01-01/01-01"
    obj_path = "canonical_face_model.obj"  # Path to MediaPipe OBJ file
    
    processor, fps, snr_values = process_video_offline(video_path, n_samples=300)
    
    if processor is not None and snr_values is not None:
        # Get statistics
        valid_snrs = [snr for snr, _, _ in snr_values.values() if not np.isnan(snr)]
        valid_bpms = [bpm for _, _, bpm in snr_values.values() if not np.isnan(bpm)]
        
        print(f"\n=== Final SNR Statistics ===")
        print(f"Valid regions: {len(valid_snrs)}/{len(snr_values)}")
        print(f"Mean SNR: {np.mean(valid_snrs):.2f} dB")
        print(f"Median SNR: {np.median(valid_snrs):.2f} dB")
        print(f"Max SNR: {np.max(valid_snrs):.2f} dB")
        print(f"Min SNR: {np.min(valid_snrs):.2f} dB")
        print(f"\nMean Heart Rate: {np.mean(valid_bpms):.1f} BPM")
        print(f"Median Heart Rate: {np.median(valid_bpms):.1f} BPM")
        
        # Find best regions
        best_regions = [(idx, snr, bpm) for idx, (snr, _, bpm) in snr_values.items() 
                       if not np.isnan(snr)]
        best_regions.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n=== Top 10 Regions by SNR ===")
        for rank, (tri_idx, snr_db, bpm) in enumerate(best_regions[:10], 1):
            print(f"{rank}. Triangle {tri_idx}: SNR = {snr_db:.2f} dB, HR = {bpm:.1f} BPM")
        
        # Plot face-shaped SNR heatmap (original)
        processor.plot_snr_heatmap_face(snr_values, save_path='snr_heatmap_face_23.png', 
                                        colormap='plasma')
        
        # Plot UV-map from OBJ file (vector style)
        # processor.plot_snr_heatmap_uv_from_obj(snr_values, obj_path,
        #                                         save_path='snr_heatmap_uv_obj.png',
        #                                         colormap='plasma', show_edges=True)
        
        processor.plot_snr_heatmap_uv_simple(snr_values, obj_path,
                                                save_path='snr_heatmap_uv_obj_23.png',
                                                colormap='plasma')
        
        # Create texture map (raster image that can be applied to 3D model)
        processor.plot_snr_heatmap_uv_texture(snr_values, obj_path,
                                               save_path='snr_heatmap_texture.png',
                                               colormap='plasma', texture_size=2048)
        
        # Plot statistics
        processor.plot_snr_statistics(snr_values, save_path='snr_statistics.png')