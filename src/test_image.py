import cv2
import torch
import numpy as np
from model import LandmarkModel
from torchvision import transforms
import mediapipe as mp

# ===== LOAD MODEL =====
device = torch.device("cpu")
model = LandmarkModel().to(device)
model.load_state_dict(torch.load("landmark.pth", map_location=device))
model.eval()

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

# ===== MEDIA PIPE FACE DETECT =====
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# ===== LOAD IMAGE =====
img_path = r"d:\DMS_Project1\test.JPG"
image = cv2.imread(img_path)

if image is None:
    print(f"Không thể đọc ảnh từ {img_path}")
    exit()

h, w, _ = image.shape
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_detector.process(rgb_image)

if not results.detections:
    print("Không tìm thấy khuôn mặt trong ảnh!")
else:
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        
        # Chuyển đổi trạng thái tương đối sang pixel
        x1 = int(bboxC.xmin * w)
        y1 = int(bboxC.ymin * h)
        box_w = int(bboxC.width * w)
        box_h = int(bboxC.height * h)
        
        # [QUAN TRỌNG NHẤT] ÉP BOX THÀNH HÌNH VUÔNG (SQUARE) VÀ CÓ PADDING
        # Chống méo ảnh (Aspect Ratio Distortion) làm điểm Landmark bị rời khỏi viền mắt
        cx = x1 + box_w // 2
        cy = y1 + box_h // 2
        
        # Lấy cạnh lớn nhất để làm chuẩn hình vuông, cộng thêm 15% Margin
        side = int(max(box_w, box_h) * 1.15)
        
        # Cắt với toạ độ mới đảm bảo không lọt ra ngoài ảnh
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w - 1, cx + side // 2)
        y2 = min(h - 1, cy + side // 2)
        
        final_w = x2 - x1
        final_h = y2 - y1
        
        if final_w < 10 or final_h < 10:
            continue
            
        # Cắt xén (crop) khuôn mặt ra khỏi ảnh
        face_crop = image[y1:y2, x1:x2]
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        input_tensor = transform(face_gray).unsqueeze(0).to(device)
        
        # Chạy dự đoán
        with torch.no_grad():
            preds = model(input_tensor).squeeze().cpu().numpy()
            
        # Tính toán lại tọa độ hiển thị 
        for i in range(98):
            px = preds[i*2]
            py = preds[i*2+1]
            real_x = int(x1 + px * final_w)
            real_y = int(y1 + py * final_h)
            cv2.circle(image, (real_x, real_y), 2, (0, 255, 0), -1)
            
        # Vẽ khung Face crop
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 120, 0), 2)

# ===== SHOW KẾT QUẢ =====
output_path = r"d:\DMS_Project1\output.jpg"
cv2.imwrite(output_path, image)
print(f"Đã lưu ảnh chứa Landmark tại: {output_path}")