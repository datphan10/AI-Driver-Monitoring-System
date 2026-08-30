import torch
import numpy as np
import os
import time
from torch.utils.data import DataLoader

from dataset import WFLWDataset
from model import LandmarkModel

# ====== CONFIG ======
BATCH_SIZE = 1
# Đường dẫn (đã cố định theo cấu trúc dự án)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WFLW_DIR = os.path.join(BASE_DIR, "WFLW")
ANNOTATION_PATH = os.path.join(WFLW_DIR, "WFLW_annotations", "list_98pt_rect_attr_train_test", "list_98pt_rect_attr_test.txt")
IMAGE_DIR = os.path.join(WFLW_DIR, "WFLW_images")

MODEL_PATH = os.path.join(BASE_DIR, "landmark.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print("\n" + "="*60)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH (EVALUATION REPORT) 🚀")
    print("="*60)
    print(f"👉 Nền tảng phần cứng (Hardware) : {device}")
    
    # ====== LOAD MODEL ======
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy model tại {MODEL_PATH}")
        return
        
    model = LandmarkModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Tính toán Parameters và Model Size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print("\n" + "-"*60)
    print("📊 1. THÔNG SỐ KIẾN TRÚC MÔ HÌNH")
    print("-"*60)
    print(f"   • Số lượng tham số (Parameters) : {total_params:,}")
    print(f"   • Kích thước lưu trữ (Size)     : {model_size_mb:.2f} MB")
    
    # ====== LOAD DATASET ======
    if not os.path.exists(ANNOTATION_PATH):
         print(f"❌ Không tìm thấy file Nhãn (Annotation test): {ANNOTATION_PATH}")
         print("Bạn hãy kiểm tra lại dataset WFLW đã có file list_98pt_rect_attr_test.txt chưa.")
         return

    dataset = WFLWDataset(ANNOTATION_PATH, IMAGE_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("\n" + "-"*60)
    print("📂 2. THÔNG TIN DỮ LIỆU ĐÁNH GIÁ (DATASET)")
    print("-"*60)
    print(f"   • Tập dữ liệu kiểm thử (Test Set): {len(dataset):,} ảnh (WFLW)")

    # ====== EVALUATION LOOP ======
    nme_list = []
    inference_times = []

    print("\n⏳ Đang tiến hành suy luận (Inference) trên Test Set...")
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            labels = labels.cpu().numpy()
            
            # Measure inference time
            start_time = time.time()
            preds = model(imgs).cpu().numpy()
            inference_times.append(time.time() - start_time)
            
            # Predict
            pred = preds[0]
            gt = labels[0]
            
            # Reshape về dạng 98 điểm (x, y)
            pred_pts = pred.reshape(98, 2)
            gt_pts = gt.reshape(98, 2)
            
            # Norm theo tiêu chuẩn: Khoảng cách đuôi mắt 2 bên (Outer corners of the eyes: pt 60 vs pt 72)
            interocular_dist = np.linalg.norm(gt_pts[60] - gt_pts[72])
            
            # Tránh chia cho 0 nếu bị lỗi
            if interocular_dist < 1e-6:
                continue
                
            # Tính Mean Error (L2 norm) cho tất cả 98 điểm trên Ground Truth
            error_per_point = np.linalg.norm(pred_pts - gt_pts, axis=1)
            mean_error = np.mean(error_per_point)
            
            # NME (Normalized Mean Error) của 1 ảnh
            nme = mean_error / interocular_dist
            nme_list.append(nme)

            if batch_idx % 500 == 0 and batch_idx > 0:
                print(f"   [+] Đã xử lý: {batch_idx}/{len(dataset)} ảnh...")

    # ====== CALCULATE FULL METRICS ======
    nme_array = np.array(nme_list)
    average_nme = np.mean(nme_array) * 100 # Chuyển sang %
    
    # Khái niệm Accuracy ở Landmark: Tỉ lệ số ảnh có tỷ lệ NME < 0.1 (Tức sai số lệch nhỏ hơn 10% mắt)
    threshold = 0.1
    accuracy = np.mean(nme_array < threshold) * 100
    
    # Tốc độ FPS (Chỉ tính riêng phần Model xử lý - Không tính Camera & UI)
    # Loại bỏ 10 frame đầu bị overhead (Cold Start)
    average_time = np.mean(inference_times[10:]) if len(inference_times) > 10 else np.mean(inference_times)
    fps = 1.0 / average_time if average_time > 0 else 0

    print("\n" + "="*60)
    print("🏆 BÁO CÁO KẾT QUẢ HIỆU NĂNG (PERFORMANCE METRICS)")
    print("="*60)
    print(f" 🎯 Độ chính xác tổng thể (Accuracy) : {accuracy:.2f} % (NME < {threshold})")
    print(f" 📉 Sai số chuẩn hóa trung bình (NME): {average_nme:.2f} %")
    print(f" ⚡ Tốc độ xử lý khung hình (FPS)    : {fps:.1f} FPS")
    print("="*60)
    print(f"\n💡 HƯỚNG DẪN: Cập nhật biến MODEL_ACCURACY = {accuracy:.2f} trong file realtime.py")

if __name__ == "__main__":
    evaluate()
