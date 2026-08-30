import numpy as np
import cv2

# ================= KHOẢNG CÁCH EUCLIDE =================
def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

# ================= GET EYE POINTS =================
def get_points(landmarks, indices):
    """
    Trích xuất mảng toạ độ từ list mục tiêu.
    """
    return landmarks[indices]

# ================= COMPUTE EAR (EYE ASPECT RATIO) =================
def compute_EAR(eye):
    """
    Hệ số EAR (Eye Aspect Ratio) theo chuẩn công bố khoa học của Soukupova và Cech (2016),
    đã được tinh chỉnh cho hệ 8 điểm quanh mắt của dataset WFLW.
    
    Trong WFLW (8 điểm):
    - eye[0]: Khoé ngoài (p1)
    - eye[1], eye[2], eye[3]: Ba điểm mí trên
    - eye[4]: Khoé trong (p4)
    - eye[5], eye[6], eye[7]: Ba điểm mí dưới
    """
    # 3 Khoảng cách dọc (Top to Bottom) để đảm bảo độ đo chính xác nhất
    A = np.linalg.norm(eye[1] - eye[7]) 
    B = np.linalg.norm(eye[2] - eye[6]) 
    C = np.linalg.norm(eye[3] - eye[5]) 
    
    # Khoảng cách ngang (Left to Right)
    D = np.linalg.norm(eye[0] - eye[4]) 

    # Trả về công thức kinh điển (Trung bình khoảng cách dọc chia cho khoảng cách ngang)
    ear = (A + B + C) / (3.0 * D + 1e-6)
    return ear

def calculate_ear_from_landmarks(landmarks):
    """
    Hàm đóng gói trả về trung bình độ mở 2 mắt.
    """
    LEFT_EYE_IDX = list(range(60, 68))
    RIGHT_EYE_IDX = list(range(68, 76))

    left_eye = get_points(landmarks, LEFT_EYE_IDX)
    right_eye = get_points(landmarks, RIGHT_EYE_IDX)

    ear_left = compute_EAR(left_eye)
    ear_right = compute_EAR(right_eye)
    
    ear_avg = (ear_left + ear_right) / 2.0
    return ear_left, ear_right, ear_avg

# ================= COMPUTE MAR (MOUTH ASPECT RATIO) =================
def compute_MAR(mouth):
    """
    Hệ số MAR (Tỉ lệ co mở miệng) - Phục vụ phát hiện NGÁP (Yawning).
    [Phục vụ luận văn] Trong dataset WFLW, sử dụng các điểm môi ngoài (76-87).
    """
    # Các khoảng cách dọc môi trên xuống môi dưới (dựa theo thứ tự 12 điểm outer lip)
    # 12 điểm Outer: 76 (Trái), 77, 78, 79(Tâm trên), 80, 81, 82 (Phải)
    # Dưới: 83, 84, 85(Tâm dưới), 86, 87
    
    # Chỉ đo 3 khoảng cách dọc đặc trưng để chuẩn hoá nhiễu:
    A = np.linalg.norm(mouth[2] - mouth[10]) # Điểm 78 đến 86
    B = np.linalg.norm(mouth[3] - mouth[9])  # Điểm 79 (chóp) đến 85 (đáy)
    C = np.linalg.norm(mouth[4] - mouth[8])  # Điểm 80 đến 84
    
    # Khoảng cách ngang miệng
    D = np.linalg.norm(mouth[0] - mouth[6]) # Điểm 76 đến 82
    
    mar = (A + B + C) / (3.0 * D + 1e-6)
    return mar

def calculate_mar_from_landmarks(landmarks):
    # Bộ điểm viền môi ngoài (12 điểm)
    OUTER_LIP_IDX = list(range(76, 88))
    mouth = get_points(landmarks, OUTER_LIP_IDX)
    mar = compute_MAR(mouth)
    return mar

# ================= SOLVE PNP HEAD POSE =================
def solve_pnp_head_pose(landmarks, frame_shape):
    """
    Sử dụng 6 điểm tiêu chuẩn để xoay không gian 3D, lấy ra 3 góc Euler (Pitch, Yaw, Roll).
    [Phục vụ luận văn] Thuật toán SolvePnP của OpenCV là chuẩn công nghiệp để 
    ánh xạ từ 2D Landmarks vào 3D Face Model nhằm phát hiện Gật Đầu (Nodding) 
    và Lơ đễnh nhìn sang hai bên (Yawning).
    """
    size = frame_shape # (h, w, c)
    focal_length = size[1]
    center = (size[1]/2.0, size[0]/2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4,1))
    
    # Khuôn mẫu 3D trung bình (Mean 3D Face) tính theo milimet
    model_points = np.array([
        (0.0, 0.0, 0.0),             # 54: Chóp mũi
        (0.0, 330.0, -65.0),         # 16: Cằm 
        (-225.0, -170.0, -135.0),    # 60: Góc mắt trái (viền ngoài)
        (225.0, -170.0, -135.0),     # 72: Góc mắt phải (viền ngoài)
        (-150.0, 150.0, -125.0),     # 76: Mép miệng trái 
        (150.0, 150.0, -125.0)       # 82: Mép miệng phải 
    ], dtype="double")
    
    # Mapping 6 điểm đó từ model thực tế
    image_points = np.array([
        landmarks[54],
        landmarks[16],
        landmarks[60],
        landmarks[72],
        landmarks[76],
        landmarks[82]
    ], dtype="double")
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 0, 0, 0
        
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    pitch = angles[0] # Lên / Xuống (Gật đầu)
    yaw = angles[1]   # Trái / Phải (Lơ đễnh nhìn ngang)
    roll = angles[2]  # Trục xiên
    
    return pitch, yaw, roll