import numpy as np
import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
from matplotlib import cm
from collections import defaultdict, deque
from scipy import signal
import time


class FaceMeshRPPG:
    """
    Process video stream to extract RGB signals from face mesh polygon regions.
    """
    
    def __init__(self, n_samples=100, window_size=150, target_fps=30):
        """
        Initialize the face mesh rPPG processor.
        
        Args:
            n_samples: Number of Monte Carlo samples per triangle per frame
            window_size: Number of frames to use for SNR computation (sliding window)
            target_fps: Target sampling rate for uniform temporal sampling
        """
        self.n_samples = n_samples
        self.window_size = window_size
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps  # Time between frames in seconds
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Storage for RGB streams (using deque for sliding window)
        self.rgb_streams = defaultdict(lambda: deque(maxlen=window_size))
        self.frame_timestamps = deque(maxlen=window_size)  # Track frame timestamps
        self.triangles = None
        self.frame_count = 0
        self.last_process_time = None
        
        # SNR cache
        self.current_snr = {}
        self.snr_update_interval = 30  # Update SNR every 30 frames
        
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
        Convert MediaPipe connections to triangle list.
        
        Args:
            connections: MediaPipe face mesh connections
        
        Returns:
            triangles: List of triangle vertex indices
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
                            triangles.add(tuple(sorted([i, j, k])))
        
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
        
        mean_rgb_list = []
        
        for tri in self.triangles:
            v0, v1, v2 = tri
            
            # Get triangle vertices in pixel coordinates (ensure they're 1D arrays)
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
    
    def should_process_frame(self, current_time):
        """
        Determine if enough time has passed to process the next frame.
        
        Args:
            current_time: Current timestamp
        
        Returns:
            should_process: Boolean indicating if frame should be processed
        """
        if self.last_process_time is None:
            self.last_process_time = current_time
            return True
        
        time_since_last = current_time - self.last_process_time
        
        if time_since_last >= self.frame_interval:
            self.last_process_time = current_time
            return True
        
        return False
    
    def process_frame(self, frame, timestamp):
        """
        Process a single video frame and accumulate RGB values.
        
        Args:
            frame: RGB frame (numpy array)
            timestamp: Frame timestamp (seconds)
        
        Returns:
            landmarks: Extracted landmarks (for visualization if needed)
            success: Boolean indicating if face was detected
        """
        landmarks, connections = self.extract_face_landmarks(frame)
        
        if landmarks is None:
            return None, False
        
        # Sample RGB from polygons
        mean_rgb_values = self.sample_polygon_rgb(frame, landmarks, connections)
        
        # Accumulate RGB values for each triangle
        for tri_idx, rgb_value in enumerate(mean_rgb_values):
            self.rgb_streams[tri_idx].append(rgb_value)
        
        # Store timestamp
        self.frame_timestamps.append(timestamp)
        
        self.frame_count += 1
        
        return landmarks, True
    
    def get_rgb_streams(self):
        """
        Get the accumulated RGB streams for all polygon regions.
        
        Returns:
            rgb_streams: Dictionary {triangle_idx: numpy array of shape (n_frames, 3)}
        """
        return {tri_idx: np.array(list(values)) for tri_idx, values in self.rgb_streams.items()}
    
    def get_effective_fps(self):
        """
        Calculate the effective sampling rate from stored timestamps.
        
        Returns:
            effective_fps: Actual sampling rate
        """
        if len(self.frame_timestamps) < 2:
            return self.target_fps
        
        timestamps = list(self.frame_timestamps)
        time_diffs = np.diff(timestamps)
        mean_interval = np.mean(time_diffs)
        
        if mean_interval > 0:
            return 1.0 / mean_interval
        else:
            return self.target_fps
    
    def compute_pos_signal(self, rgb_stream, fps=30):
        """
        Compute POS (Plane-Orthogonal-to-Skin) rPPG signal from RGB stream.
        
        Based on: Wang et al. "Algorithmic Principles of Remote PPG" (2017)
        
        Args:
            rgb_stream: Numpy array of shape (n_frames, 3) with RGB values
            fps: Video frame rate
        
        Returns:
            pos_signal: POS pulse signal
        """
        # Check for valid data
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
        X_s = rgb_norm[:, 0] - rgb_norm[:, 1]  # R - G
        Y_s = rgb_norm[:, 0] + rgb_norm[:, 1] - 2 * rgb_norm[:, 2]  # R + G - 2B
        
        # Window size for temporal normalization
        window_size = int(fps * 1.6)
        if window_size < 2:
            window_size = 2
        
        # Initialize POS signal
        pos_signal = np.zeros(len(rgb_stream))
        
        # Sliding window temporal normalization
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
        
        # Bandpass filter
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
    
    def update_snr_values(self):
        """
        Update SNR values for all regions using current window of data.
        """
        rgb_streams = self.get_rgb_streams()
        effective_fps = self.get_effective_fps()
        
        for tri_idx, rgb_stream in rgb_streams.items():
            if len(rgb_stream) >= 60:  # At least 2 seconds of data
                pos_signal = self.compute_pos_signal(rgb_stream, effective_fps)
                snr_db, peak_freq, peak_bpm = self.compute_snr(pos_signal, effective_fps)
                self.current_snr[tri_idx] = snr_db
            else:
                self.current_snr[tri_idx] = np.nan
    
    def create_snr_heatmap(self, frame, landmarks, alpha=0.6, colormap='jet'):
        """
        Create SNR heatmap overlay on the frame.
        
        Args:
            frame: RGB frame (numpy array)
            landmarks: Facial landmarks
            alpha: Transparency of heatmap (0-1)
            colormap: Matplotlib colormap name
        
        Returns:
            heatmap_frame: Frame with SNR heatmap overlay
        """
        if landmarks is None or self.triangles is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create a blank overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Convert landmarks to pixel coordinates
        pixel_coords = landmarks.copy()
        pixel_coords[:, 0] *= w
        pixel_coords[:, 1] *= h
        
        # Get valid SNR values for color normalization
        valid_snrs = [snr for snr in self.current_snr.values() if not np.isnan(snr)]
        
        if len(valid_snrs) == 0:
            return frame
        
        # Normalize SNR values to [0, 1] for colormap
        min_snr = np.percentile(valid_snrs, 5)  # Use 5th percentile to avoid outliers
        max_snr = np.percentile(valid_snrs, 95)  # Use 95th percentile
        
        # Get colormap
        cmap = cm.get_cmap(colormap)
        
        # Draw each triangle with color based on SNR
        for tri_idx, tri in enumerate(self.triangles):
            snr_value = self.current_snr.get(tri_idx, np.nan)
            
            if np.isnan(snr_value):
                continue
            
            # Normalize SNR to [0, 1]
            normalized_snr = (snr_value - min_snr) / (max_snr - min_snr) if max_snr > min_snr else 0.5
            normalized_snr = np.clip(normalized_snr, 0, 1)
            
            # Get color from colormap
            color = cmap(normalized_snr)[:3]  # RGB values in [0, 1]
            color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
            
            # Get triangle vertices
            v0, v1, v2 = tri
            pts = np.array([
                pixel_coords[v0][:2],
                pixel_coords[v1][:2],
                pixel_coords[v2][:2]
            ], dtype=np.int32)
            
            # Fill triangle
            cv2.fillPoly(overlay, [pts], color_bgr)
        
        # Blend overlay with original frame
        heatmap_frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        # Add colorbar legend
        heatmap_frame = self.add_colorbar(heatmap_frame, min_snr, max_snr, colormap)
        
        return heatmap_frame
    
    def add_colorbar(self, frame, min_val, max_val, colormap='jet'):
        """
        Add a colorbar to the frame.
        
        Args:
            frame: Frame to add colorbar to
            min_val: Minimum SNR value
            max_val: Maximum SNR value
            colormap: Colormap name
        
        Returns:
            frame_with_colorbar: Frame with colorbar added
        """
        h, w = frame.shape[:2]
        
        # Colorbar dimensions
        bar_width = 30
        bar_height = 200
        bar_x = w - bar_width - 20
        bar_y = 50
        
        # Create colorbar
        cmap = cm.get_cmap(colormap)
        
        for i in range(bar_height):
            # Normalized position (inverted so high values are at top)
            norm_pos = 1 - (i / bar_height)
            color = cmap(norm_pos)[:3]
            color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
            
            cv2.rectangle(frame, (bar_x, bar_y + i), (bar_x + bar_width, bar_y + i + 1), 
                         color_bgr, -1)
        
        # Add border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Max value at top
        cv2.putText(frame, f'{max_val:.1f}', (bar_x + bar_width + 5, bar_y + 10),
                   font, font_scale, (255, 255, 255), thickness)
        
        # Min value at bottom
        cv2.putText(frame, f'{min_val:.1f}', (bar_x + bar_width + 5, bar_y + bar_height),
                   font, font_scale, (255, 255, 255), thickness)
        
        # SNR label
        cv2.putText(frame, 'SNR (dB)', (bar_x - 20, bar_y - 10),
                   font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def reset(self):
        """Reset the accumulated RGB streams."""
        self.rgb_streams = defaultdict(lambda: deque(maxlen=self.window_size))
        self.frame_timestamps = deque(maxlen=self.window_size)
        self.triangles = None
        self.frame_count = 0
        self.current_snr = {}
        self.last_process_time = None
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        self.face_mesh.close()


def process_video_realtime_with_heatmap(video_path, n_samples=100, window_size=150, 
                                         target_fps=30, show_mesh=False, colormap='jet'):
    """
    Process video in real-time with SNR heatmap visualization and uniform temporal sampling.
    
    Args:
        video_path: Path to video file or 0 for webcam
        n_samples: Number of Monte Carlo samples per triangle
        window_size: Number of frames for sliding window analysis
        target_fps: Target sampling rate (frames will be uniformly sampled at this rate)
        show_mesh: Whether to show mesh edges
        colormap: Colormap for heatmap ('jet', 'hot', 'viridis', etc.)
    
    Returns:
        processor: FaceMeshRPPG object with accumulated data
        actual_fps: Actual achieved sampling rate
    """
    processor = FaceMeshRPPG(n_samples=n_samples, window_size=window_size, target_fps=target_fps)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video")
        return None, 30
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30
    
    print(f"Video FPS: {video_fps}")
    print(f"Target sampling rate: {target_fps} fps")
    print(f"Window size: {window_size} frames ({window_size/target_fps:.1f} seconds)")
    print("Press 'q' to quit, 's' to save current frame, 'm' to toggle mesh overlay")
    print("Heatmap will appear after collecting sufficient data...")
    
    frame_idx = 0
    processed_frames = 0
    show_mesh_overlay = show_mesh
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        current_time = time.time() - start_time
        
        # Check if we should process this frame for uniform temporal sampling
        if processor.should_process_frame(current_time):
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with timestamp
            landmarks, success = processor.process_frame(frame_rgb, current_time)
            processed_frames += 1
            
            # Update SNR values periodically
            if processed_frames % processor.snr_update_interval == 0 and processed_frames > 0:
                processor.update_snr_values()
            
            # Create visualization frame
            if success:
                # Convert back to BGR for display
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Create heatmap overlay
                if len(processor.current_snr) > 0:
                    display_frame = processor.create_snr_heatmap(frame_bgr, landmarks, 
                                                                 alpha=0.5, colormap=colormap)
                else:
                    display_frame = frame_bgr.copy()
                
                # Optionally draw mesh edges
                if show_mesh_overlay and processor.triangles is not None:
                    h, w = frame_bgr.shape[:2]
                    pixel_coords = landmarks.copy()
                    pixel_coords[:, 0] *= w
                    pixel_coords[:, 1] *= h
                    
                    for tri in processor.triangles:
                        v0, v1, v2 = tri
                        pts = np.array([
                            pixel_coords[v0][:2],
                            pixel_coords[v1][:2],
                            pixel_coords[v2][:2]
                        ], dtype=np.int32)
                        
                        cv2.polylines(display_frame, [pts], True, (255, 255, 255), 1)
            else:
                display_frame = frame.copy()
            
            # Calculate effective FPS
            effective_fps = processor.get_effective_fps()
            
            # Add info text
            info_text = f"Processed: {processed_frames} | Regions: {len(processor.triangles) if processor.triangles else 0}"
            cv2.putText(display_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            fps_text = f"Effective FPS: {effective_fps:.1f}"
            cv2.putText(display_frame, fps_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add status
            if processed_frames < window_size:
                status = f"Collecting data: {processed_frames}/{window_size}"
                cv2.putText(display_frame, status, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                valid_snrs = [s for s in processor.current_snr.values() if not np.isnan(s)]
                if len(valid_snrs) > 0:
                    mean_snr = np.mean(valid_snrs)
                    status = f"Mean SNR: {mean_snr:.2f} dB"
                    cv2.putText(display_frame, status, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if not success:
                cv2.putText(display_frame, "No face detected", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('Face Mesh rPPG - SNR Heatmap', display_frame)
        else:
            # Display the frame even if not processing (for smooth video playback)
            cv2.imshow('Face Mesh rPPG - SNR Heatmap', frame)
        
        # Handle keyboard input (non-blocking)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            save_path = f"heatmap_{processed_frames:04d}.jpg"
            cv2.imwrite(save_path, display_frame if 'display_frame' in locals() else frame)
            print(f"Saved frame to {save_path}")
        elif key == ord('m'):
            show_mesh_overlay = not show_mesh_overlay
            print(f"Mesh overlay: {'ON' if show_mesh_overlay else 'OFF'}")
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    actual_fps = processor.get_effective_fps()
    
    print(f"\nProcessing complete!")
    print(f"Total frames read: {frame_idx}")
    print(f"Total frames processed: {processed_frames}")
    print(f"Target FPS: {target_fps}")
    print(f"Actual FPS: {actual_fps:.2f}")
    print(f"Sampling uniformity: {(actual_fps/target_fps)*100:.1f}%")
    
    return processor, actual_fps


if __name__ == "__main__":
    # Real-time SNR heatmap processing with uniform temporal sampling
    
    # Option 1: Webcam with heatmap
    # processor, actual_fps = process_video_realtime_with_heatmap(
    #     0,  # Webcam
    #     n_samples=100,
    #     window_size=150,  # 5 seconds at 30fps
    #     target_fps=30,  # Sample at 30 fps uniformly
    #     show_mesh=False,
    #     colormap='jet'  # Options: 'jet', 'hot', 'viridis', 'plasma', 'coolwarm'
    # )
    
    # Option 2: Video file with heatmap
    processor, actual_fps = process_video_realtime_with_heatmap(
        "/home/daevinci/Datasets/DATASET_2/subject1/vid.avi",
        n_samples=100,
        window_size=150,
        target_fps=30,
        show_mesh=False,
        colormap='plasma'
    )
    
    if processor is not None and len(processor.current_snr) > 0:
        # Final statistics
        valid_snrs = [s for s in processor.current_snr.values() if not np.isnan(s)]
        
        if len(valid_snrs) > 0:
            print(f"\n=== Final SNR Statistics ===")
            print(f"Valid regions: {len(valid_snrs)}/{len(processor.current_snr)}")
            print(f"Mean SNR: {np.mean(valid_snrs):.2f} dB")
            print(f"Median SNR: {np.median(valid_snrs):.2f} dB")
            print(f"Max SNR: {np.max(valid_snrs):.2f} dB")
            print(f"Min SNR: {np.min(valid_snrs):.2f} dB")