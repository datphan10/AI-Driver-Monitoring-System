from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms

from model import LandmarkModel


# ============================================================
# PROJECT PATHS
# ============================================================

# src/test_image.py -> src -> project root
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "landmark.pth"
INPUT_IMAGE_PATH = BASE_DIR / "assets" / "test.jpg"
OUTPUT_IMAGE_PATH = BASE_DIR / "assets" / "output.jpg"


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cpu")

model = LandmarkModel().to(device)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy model: {MODEL_PATH}\n"
        "Hãy đặt file landmark.pth tại thư mục gốc của project."
    )

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])


# ============================================================
# MEDIAPIPE FACE DETECTION
# ============================================================

mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)


# ============================================================
# LOAD INPUT IMAGE
# ============================================================

image = cv2.imread(str(INPUT_IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Không thể đọc ảnh: {INPUT_IMAGE_PATH}"
    )

h, w, _ = image.shape

rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

results = face_detector.process(rgb_image)


# ============================================================
# FACE + LANDMARK DETECTION
# ============================================================

if not results.detections:
    print("Không tìm thấy khuôn mặt trong ảnh!")

else:
    for detection in results.detections:

        bbox = detection.location_data.relative_bounding_box

        # Convert relative coordinates to pixels
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)

        box_w = int(bbox.width * w)
        box_h = int(bbox.height * h)

        # ----------------------------------------------------
        # Convert bounding box to square + 15% padding
        # ----------------------------------------------------

        cx = x1 + box_w // 2
        cy = y1 + box_h // 2

        side = int(max(box_w, box_h) * 1.15)

        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)

        x2 = min(w - 1, cx + side // 2)
        y2 = min(h - 1, cy + side // 2)

        final_w = x2 - x1
        final_h = y2 - y1

        if final_w < 10 or final_h < 10:
            continue

        # ----------------------------------------------------
        # Crop face
        # ----------------------------------------------------

        face_crop = image[y1:y2, x1:x2]

        face_gray = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2GRAY
        )

        input_tensor = (
            transform(face_gray)
            .unsqueeze(0)
            .to(device)
        )

        # ----------------------------------------------------
        # Landmark prediction
        # ----------------------------------------------------

        with torch.no_grad():
            preds = (
                model(input_tensor)
                .squeeze()
                .cpu()
                .numpy()
            )

        # ----------------------------------------------------
        # Draw 98 landmarks
        # ----------------------------------------------------

        for i in range(98):

            px = preds[i * 2]
            py = preds[i * 2 + 1]

            real_x = int(x1 + px * final_w)
            real_y = int(y1 + py * final_h)

            cv2.circle(
                image,
                (real_x, real_y),
                2,
                (0, 255, 0),
                -1
            )

        # Draw face bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 120, 0),
            2
        )


# ============================================================
# SAVE RESULT
# ============================================================

OUTPUT_IMAGE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

cv2.imwrite(
    str(OUTPUT_IMAGE_PATH),
    image
)

print(
    f"Landmark result saved to: {OUTPUT_IMAGE_PATH}"
)