import torch
from torch.utils.data import DataLoader
from dataset import WFLWDataset
from model import LandmarkModel
import os

# ====== 1. CONFIG ======
BATCH_SIZE = 64
EPOCHS = 80
LR = 0.001
WEIGHT_DECAY = 1e-5

import math
class WingLoss(torch.nn.Module):
    def __init__(self, w=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.c = self.w - self.w * math.log(1 + self.w / self.epsilon)
        
        # Cấu hình Trọng số Không gian (Spatial Weights)
        # Khởi tạo trọng số cơ bản = 1.0 cho 196 toạ độ (98 điểm x 2)
        self.weights = torch.ones(196)
        
        # [QUAN TRỌNG] TĂNG TRỌNG SỐ VÙNG MẮT LÊN 5 LẦN
        # Mắt trái: 60-67 | Mắt phải: 68-75
        # Trong mảng 1 chiều (196), toạ độ mắt bắt đầu từ 60*2=120 đến 75*2+1=151
        self.weights[120:152] = 5.0 
        
        # (Tuỳ chọn) Tăng nhẹ trọng số vùng miệng để bù góc nghiêng
        # Miệng: 76-97 -> Toạ độ: 152-195
        self.weights[152:196] = 2.0

    def forward(self, preds, targets):
        diff = torch.abs(preds - targets)
        
        # Tính Wing Loss nguyên gốc
        loss = torch.where(diff < self.w, 
                           self.w * torch.log(1 + diff / self.epsilon), 
                           diff - self.c)
        
        # Áp dụng trọng số lên từng điểm tương ứng
        weight_tensor = self.weights.to(preds.device)
        weighted_loss = loss * weight_tensor
        
        return weighted_loss.mean()


# 🔥 FIX PATH (QUAN TRỌNG) ĐÃ CHUYỂN SANG ĐƯỜNG DẪN ĐỘNG THEO NƠI CHỨA
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATION_PATH = os.path.join(BASE_DIR, "WFLW", "WFLW_annotations", "list_98pt_rect_attr_train_test", "list_98pt_rect_attr_train.txt")
IMAGE_DIR = os.path.join(BASE_DIR, "WFLW", "WFLW_images")

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "landmark.pth")

# ====== 2. DEVICE ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ====== 3. DATASET ======
dataset = WFLWDataset(ANNOTATION_PATH, IMAGE_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Dataset size:", len(dataset))

# ====== 4. MODEL ======
model = LandmarkModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
# Sử dụng WingLoss để xử lý siêu chi tiết các chấm mắt (khắc phục điểm mắt không dính sát)
criterion = WingLoss()

# Parameters for Early Stopping
best_loss = float('inf')
patience = 10
patience_counter = 0

# ====== 5. TRAIN ======
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, (imgs, labels) in enumerate(loader):

        imgs = imgs.to(device)
        labels = labels.to(device)

        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"🔥 Epoch {epoch} DONE | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Bước scheduler sau mỗi epoch
    scheduler.step()

    # Early stopping logic & model saving
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"   [+] Model improved! Saved to {MODEL_SAVE_PATH}")
    else:
        patience_counter += 1
        print(f"   [-] No improvement. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered tại Epoch {epoch}")
            break

print("✅ TRAINING DONE!")