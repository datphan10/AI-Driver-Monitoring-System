import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class WFLWDataset(Dataset):
    def __init__(self, annotation_file, image_dir):
        self.image_dir = image_dir

        with open(annotation_file, 'r') as f:
            self.lines = f.readlines()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split()

        # ===== 1. LOAD LANDMARK (98 điểm = 196 giá trị) =====
        coords = list(map(float, line[:196]))
        landmarks = np.array(coords).reshape(-1, 2)

        # ===== 2. LOAD BBOX =====
        x1 = int(float(line[196]))
        y1 = int(float(line[197]))
        x2 = int(float(line[198]))
        y2 = int(float(line[199]))

        # ===== 3. LOAD IMAGE =====
        img_path = os.path.join(self.image_dir, line[-1])
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"❌ Không đọc được ảnh: {img_path}")

        h, w, _ = image.shape

        # ===== 4. CLAMP BBOX =====
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        # ===== 5. CROP FACE =====
        face = image[y1:y2, x1:x2]

        if face.shape[0] == 0 or face.shape[1] == 0:
            # fallback: dùng full image nếu bbox lỗi
            face = image
            x1, y1 = 0, 0
            x2, y2 = w, h

        face_h, face_w, _ = face.shape

        # ===== 6. NORMALIZE LANDMARK =====
        landmarks[:, 0] = (landmarks[:, 0] - x1) / (x2 - x1 + 1e-6)
        landmarks[:, 1] = (landmarks[:, 1] - y1) / (y2 - y1 + 1e-6)

        # ===== 7. CLIP (TRÁNH OUT OF RANGE) =====
        landmarks = np.clip(landmarks, 0, 1)

        # ===== 8. TRANSFORM IMAGE =====
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = self.transform(face)

        # ===== 9. FLATTEN LANDMARK =====
        landmarks = torch.tensor(landmarks.flatten(), dtype=torch.float32)

        return face, landmarks