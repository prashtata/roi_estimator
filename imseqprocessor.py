import cv2
from pathlib import Path

class ImageDirCapture:
    def __init__(
        self,
        img_dir,
        fps=30,
        start=0,
        end=None,
        extensions=(".png", ".jpg", ".jpeg", ".bmp"),
    ):
        self.img_dir = Path(img_dir)
        if not self.img_dir.is_dir():
            raise ValueError(f"{img_dir} is not a directory")

        all_files = sorted(
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in extensions
        )

        if end is None:
            end = len(all_files)

        if not (0 <= start < end <= len(all_files)):
            raise ValueError("Invalid image range")

        # Slice defines the visible sequence
        self.files = all_files[start:end]

        self.start = start
        self.end = end
        self.pos = 0
        self.fps = fps
        self.opened = len(self.files) > 0

        self._width = None
        self._height = None
        if self.opened:
            img = cv2.imread(str(self.files[0]))
            if img is None:
                raise RuntimeError("Failed to read first image")
            self._height, self._width = img.shape[:2]

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.opened or self.pos >= len(self.files):
            return False, None

        img = cv2.imread(str(self.files[self.pos]))
        if img is None:
            return False, None

        self.pos += 1
        return True, img

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.files))
        elif prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(self.pos)
        elif prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width) if self._width else 0.0
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height) if self._height else 0.0
        return 0.0

    def set(self, prop_id, value):
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            idx = int(value)
            if 0 <= idx < len(self.files):
                self.pos = idx
                return True
            return False
        return False

    def release(self):
        self.opened = False
