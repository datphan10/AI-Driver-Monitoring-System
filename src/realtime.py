import os
import time
import wave
import struct
import threading
import warnings
import platform
from collections import deque

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import cv2
import numpy as np
import torch
import mediapipe as mp

from model import LandmarkModel
from ear_utils import calculate_ear_from_landmarks, calculate_mar_from_landmarks, solve_pnp_head_pose

# ==========================================================
# CONFIG CHO RASPBERRY PI 4 + MAN HINH HDMI 7 INCH 1024x600
# ==========================================================
SCREEN_W = 1024
SCREEN_H = 600

# Nut KET THUC tren dashboard - goc trai ben tren
EXIT_BTN_X1 = 20
EXIT_BTN_Y1 = 20
EXIT_BTN_X2 = 170
EXIT_BTN_Y2 = 75

# Khung camera ngay duoi cong truong
CAM_W = 520
CAM_H = 300
CAM_X = 252
CAM_Y = 245

# Camera de nhe cho Pi 4
CAM_ID = 0
CAP_W = 320
CAP_H = 240
CAP_FPS = 15
PROCESS_EVERY = 2          # Xu ly AI moi 2 frame, nhung van ve lai landmark moi frame => khong chop chop
DETECTION_CONF = 0.55

# Calib + smoothing
ALERT_FRAMES = 5
CALIBRATION_FRAMES = 20
SMOOTHING_FRAMES = 8
POSE_SMOOTHING_FRAMES = 5
LANDMARK_ALPHA = 0.65      # lam min landmark, giam rung
MAX_LOST_FRAMES = 8        # mat mat qua so frame nay moi xoa landmark cu

# Am thanh: pip pip pip roi 1 pip dai
BEEP_FREQ = 1200
SHORT_BEEP = 0.18
LONG_BEEP = 0.90
SILENCE = 0.12
ALARM_COOLDOWN = 3.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_PATHS = [
    os.path.join(BASE_DIR, "Hinhnen_1024x600.jpg"),
    os.path.join(BASE_DIR, "Hinhnen.jpg"),
    "/mnt/data/Hinhnen_1024x600.jpg",
    "/mnt/data/Hinhnen.jpg",
]
MODEL_PATHS = [
    os.path.join(BASE_DIR, "landmark.pth"),
    os.path.join(BASE_DIR, "..", "landmark.pth"),
    "landmark.pth",
    "../landmark.pth",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    torch.set_num_threads(2)
except Exception:
    pass

# ==========================================================
# HAM PHU TRO
# ==========================================================
def find_existing_path(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def resize_crop_center(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x1 = max(0, (new_w - target_w) // 2)
    y1 = max(0, (new_h - target_h) // 2)
    return resized[y1:y1 + target_h, x1:x1 + target_w].copy()


def load_background():
    bg_path = find_existing_path(BACKGROUND_PATHS)
    if bg_path is None:
        bg = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    else:
        bg = cv2.imread(bg_path)
        if bg is None:
            bg = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        else:
            bg = resize_crop_center(bg, SCREEN_W, SCREEN_H)

    # Lam toi nhe nen de khung camera va HUD de nhin
    dark = np.zeros_like(bg)
    bg = cv2.addWeighted(bg, 0.78, dark, 0.22, 0)
    return bg


def draw_panel(img, x1, y1, x2, y2, color=(25, 25, 25), alpha=0.60):
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    panel = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(panel, alpha, roi, 1 - alpha, 0, roi)


def put_text(img, text, pos, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_exit_button(img):
    """Ve nut KET THUC o goc trai tren dashboard."""
    # Nen nut mau do dam
    cv2.rectangle(
        img,
        (EXIT_BTN_X1, EXIT_BTN_Y1),
        (EXIT_BTN_X2, EXIT_BTN_Y2),
        (0, 0, 180),
        -1
    )

    # Vien trang cho nut
    cv2.rectangle(
        img,
        (EXIT_BTN_X1, EXIT_BTN_Y1),
        (EXIT_BTN_X2, EXIT_BTN_Y2),
        (255, 255, 255),
        2
    )

    # Chu KET THUC
    put_text(
        img,
        "KET THUC",
        (EXIT_BTN_X1 + 20, EXIT_BTN_Y1 + 36),
        0.65,
        (255, 255, 255),
        2
    )


def create_alarm_wav(path):
    sample_rate = 44100
    samples = []

    def add_tone(duration):
        n = int(sample_rate * duration)
        fade_n = int(sample_rate * 0.02)  # fade 20ms de tieng pip mem hon

        for i in range(n):
            amp = 1.0

            # Fade in: dau tieng pip nho len tu tu
            if i < fade_n:
                amp = i / fade_n

            # Fade out: cuoi tieng pip nho dan
            if i > n - fade_n:
                amp = max(0.0, (n - i) / fade_n)

            val = int(22000 * amp * np.sin(2 * np.pi * BEEP_FREQ * i / sample_rate))
            samples.append(val)

    def add_silence(duration):
        samples.extend([0] * int(sample_rate * duration))

    # pip pip pip
    for _ in range(3):
        add_tone(SHORT_BEEP)
        add_silence(SILENCE)

    # pip dai
    add_tone(LONG_BEEP)

    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for s in samples:
            f.writeframesraw(struct.pack("<h", s))

    return path


class AlarmPlayer:
    def __init__(self):
        self.is_playing = False
        self.last_time = 0.0
        self.wav_path = os.path.join(BASE_DIR, "alarm_pip.wav")
        if not os.path.exists(self.wav_path):
            create_alarm_wav(self.wav_path)

    def trigger(self):
        now = time.time()
        if self.is_playing or (now - self.last_time) < ALARM_COOLDOWN:
            return
        self.last_time = now
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self):
        self.is_playing = True
        try:
            if os.name == "nt":
                import winsound
                for _ in range(3):
                    winsound.Beep(BEEP_FREQ, int(SHORT_BEEP * 1000))
                    time.sleep(SILENCE)
                winsound.Beep(BEEP_FREQ, int(LONG_BEEP * 1000))
            else:
                cmd = (
                    f"paplay '{self.wav_path}' >/dev/null 2>&1 || "
                    f"pw-play '{self.wav_path}' >/dev/null 2>&1 || "
                    f"aplay -q '{self.wav_path}' >/dev/null 2>&1"
                )
                os.system(cmd)
        finally:
            self.is_playing = False


def smooth_points(prev_pts, new_pts, alpha=0.65):
    if prev_pts is None:
        return new_pts.astype(np.float32)
    return alpha * new_pts.astype(np.float32) + (1.0 - alpha) * prev_pts.astype(np.float32)


# ==========================================================
# LOAD MODEL + FACE DETECTOR
# ==========================================================
model_path = find_existing_path(MODEL_PATHS)
if model_path is None:
    print("CANH BAO: Khong tim thay landmark.pth")

model = LandmarkModel().to(DEVICE)
if model_path is not None:
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
model.eval()

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=DETECTION_CONF,
)


# ==========================================================
# MAIN
# ==========================================================
def main():
    background = load_background()
    alarm = AlarmPlayer()

    # Bien thoat chuong trinh khi bam nut KET THUC tren dashboard
    should_exit = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal should_exit
        if event == cv2.EVENT_LBUTTONDOWN:
            if EXIT_BTN_X1 <= x <= EXIT_BTN_X2 and EXIT_BTN_Y1 <= y <= EXIT_BTN_Y2:
                print("Da bam nut KET THUC tren dashboard.")
                should_exit = True

    backend = cv2.CAP_V4L2 if platform.system() != "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAM_ID, backend)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Khong mo duoc camera.")
        return

    frame_id = 0
    lost_frames = 0

    frame_count_eye = 0
    frame_count_pose = 0
    frame_count_mouth = 0

    ear_calib, mar_calib, pitch_calib, yaw_calib = [], [], [], []
    ear_q = deque(maxlen=SMOOTHING_FRAMES)
    mar_q = deque(maxlen=SMOOTHING_FRAMES)
    pitch_q = deque(maxlen=POSE_SMOOTHING_FRAMES)
    yaw_q = deque(maxlen=POSE_SMOOTHING_FRAMES)

    calibration_done = False
    EAR_THRESHOLD = None
    MAR_THRESHOLD = None
    PITCH_NORM = None
    YAW_NORM = None

    status_text = "Dang khoi dong"
    status_color = (0, 180, 255)
    ear_smooth = mar_smooth = pitch_smooth = yaw_smooth = 0.0

    # Luu ket qua cu de ve lai tren frame bi skip => khong chop chop
    last_bbox = None                # raw coords: x1,y1,x2,y2
    last_landmarks = None           # raw coords 98x2 float32
    last_face_detected = False

    prev_time = time.time()
    fps_q = deque(maxlen=20)
    avg_fps = 0.0

    cv2.namedWindow("DMS", cv2.WINDOW_NORMAL)

    # Tren Windows: hien cua so 1024x600 de de test
    # Tren Raspberry Pi/Linux: full man hinh cho HDMI 7 inch
    if platform.system() == "Windows":
        cv2.resizeWindow("DMS", SCREEN_W, SCREEN_H)
    else:
        cv2.setWindowProperty("DMS", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Bat su kien click/cham vao nut KET THUC
    cv2.setMouseCallback("DMS", mouse_callback)

    while True:
        ret, raw = cap.read()
        if not ret:
            print("Khong doc duoc frame camera.")
            break

        now = time.time()
        fps_q.append(1.0 / (now - prev_time + 1e-6))
        prev_time = now
        avg_fps = sum(fps_q) / len(fps_q)

        frame_id += 1
        raw_h, raw_w = raw.shape[:2]
        scale_x = CAM_W / raw_w
        scale_y = CAM_H / raw_h

        # Camera hien thi tren nen
        cam_view = cv2.resize(raw, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)

        # ---------------- AI PROCESS ----------------
        if frame_id % PROCESS_EVERY == 0:
            found_face_this_round = False
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)

            if results.detections:
                detection = max(
                    results.detections,
                    key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height,
                )
                box = detection.location_data.relative_bounding_box
                x = int(box.xmin * raw_w)
                y = int(box.ymin * raw_h)
                w = int(box.width * raw_w)
                h = int(box.height * raw_h)

                if w >= 35 and h >= 35:
                    found_face_this_round = True
                    cx, cy = x + w // 2, y + h // 2
                    side = int(max(w, h) * 1.15)
                    x1 = max(0, cx - side // 2)
                    y1 = max(0, cy - side // 2)
                    x2 = min(raw_w - 1, cx + side // 2)
                    y2 = min(raw_h - 1, cy + side // 2)

                    if (x2 - x1) >= 35 and (y2 - y1) >= 35:
                        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                        face_crop = gray[y1:y2, x1:x2]
                        face_crop = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_AREA)

                        inp = torch.from_numpy(face_crop).float().to(DEVICE)
                        inp = inp.unsqueeze(0).unsqueeze(0) / 255.0

                        with torch.no_grad():
                            preds = model(inp).squeeze().cpu().numpy()

                        pts = preds.reshape(98, 2)
                        new_landmarks = np.zeros((98, 2), dtype=np.float32)
                        new_landmarks[:, 0] = x1 + pts[:, 0] * (x2 - x1)
                        new_landmarks[:, 1] = y1 + pts[:, 1] * (y2 - y1)

                        # Lam min landmark
                        last_landmarks = smooth_points(last_landmarks, new_landmarks, LANDMARK_ALPHA)
                        last_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                        last_face_detected = True
                        lost_frames = 0

                        lm_int = np.round(last_landmarks).astype(np.int32)

                        ear_left, ear_right, ear = calculate_ear_from_landmarks(lm_int)
                        mar = calculate_mar_from_landmarks(lm_int)
                        pitch, yaw, roll = solve_pnp_head_pose(lm_int, (raw_h, raw_w))

                        if not calibration_done:
                            ear_calib.append(ear)
                            mar_calib.append(mar)
                            pitch_calib.append(pitch)
                            yaw_calib.append(yaw)
                            status_text = f"Hieu chinh {len(ear_calib)}/{CALIBRATION_FRAMES}"
                            status_color = (0, 180, 255)

                            if len(ear_calib) >= CALIBRATION_FRAMES:
                                sorted_ears = sorted(ear_calib)
                                trim = int(CALIBRATION_FRAMES * 0.1)
                                valid_ears = sorted_ears[trim:-trim] if trim > 0 else sorted_ears
                                EAR_THRESHOLD = float(np.mean(valid_ears) * 0.75)

                                valid_mars = sorted(mar_calib)[:max(1, int(CALIBRATION_FRAMES * 0.5))]
                                MAR_THRESHOLD = max(0.20, float(np.mean(valid_mars) * 1.5))

                                PITCH_NORM = float(np.mean(pitch_calib))
                                YAW_NORM = float(np.mean(yaw_calib))
                                calibration_done = True
                                status_text = "Binh thuong"
                                status_color = (0, 200, 0)
                        else:
                            ear_q.append(ear)
                            mar_q.append(mar)
                            pitch_q.append(pitch)
                            yaw_q.append(yaw)

                            ear_smooth = float(np.mean(ear_q))
                            mar_smooth = float(np.mean(mar_q))
                            pitch_smooth = float(np.mean(pitch_q))
                            yaw_smooth = float(np.mean(yaw_q))

                            yaw_diff_raw = abs(yaw - YAW_NORM)
                            pitch_diff_raw = pitch - PITCH_NORM
                            is_head_rotated = (yaw_diff_raw > 15) or (pitch_diff_raw > 12) or (pitch_diff_raw < -10)
                            rotation_factor = 1.35 if is_head_rotated else 1.0
                            current_mar_threshold = MAR_THRESHOLD * rotation_factor

                            if ear_smooth < EAR_THRESHOLD:
                                frame_count_eye += 1
                            else:
                                frame_count_eye = 0

                            if mar_smooth > current_mar_threshold:
                                frame_count_mouth += 1
                            else:
                                frame_count_mouth = 0

                            pose_alert = False
                            if (pitch_smooth - PITCH_NORM) > 18 or (pitch_smooth - PITCH_NORM) < -15:
                                pose_alert = True
                            if (yaw_smooth - YAW_NORM) > 20 or (yaw_smooth - YAW_NORM) < -20:
                                pose_alert = True

                            if pose_alert:
                                frame_count_pose += 1
                            else:
                                frame_count_pose = 0

                            alerts = []
                            if frame_count_eye >= ALERT_FRAMES or frame_count_mouth >= ALERT_FRAMES:
                                alerts.append("Buon ngu")
                            if frame_count_pose >= ALERT_FRAMES:
                                alerts.append("Mat tap trung")

                            if alerts:
                                alarm.trigger()
                                status_text = " + ".join(alerts)
                                status_color = (0, 0, 255)
                            else:
                                status_text = "Binh thuong"
                                status_color = (0, 200, 0)

            if not found_face_this_round:
                lost_frames += 1
                if lost_frames >= MAX_LOST_FRAMES:
                    last_face_detected = False
                    last_bbox = None
                    last_landmarks = None
                    if calibration_done:
                        status_text = "Khong co mat"
                        status_color = (0, 120, 255)

        # ---------------- DRAW SAVED RESULT ----------------
        # Ve lai landmark cu moi frame => khong chop chop
        if last_face_detected and last_bbox is not None and last_landmarks is not None:
            x1, y1, x2, y2 = last_bbox.astype(np.int32)
            x1d, y1d = int(x1 * scale_x), int(y1 * scale_y)
            x2d, y2d = int(x2 * scale_x), int(y2 * scale_y)
            cv2.rectangle(cam_view, (x1d, y1d), (x2d, y2d), (255, 180, 0), 2)

            lm_draw = np.round(last_landmarks).astype(np.int32)
            for lx, ly in lm_draw:
                px = int(lx * scale_x)
                py = int(ly * scale_y)
                if 0 <= px < CAM_W and 0 <= py < CAM_H:
                    cv2.circle(cam_view, (px, py), 1, (0, 255, 0), -1)

        # ---------------- COMPOSE DISPLAY ----------------
        display = background.copy()
        display[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = cam_view
        cv2.rectangle(display, (CAM_X, CAM_Y), (CAM_X + CAM_W, CAM_Y + CAM_H), (0, 255, 0), 3)

        # Nut ket thuc tren dashboard
        draw_exit_button(display)

        # Status
        draw_panel(display, 20, 530, 660, 595, alpha=0.67)
        cv2.circle(display, (50, 565), 12, status_color, -1)
        put_text(display, f"Trang thai: {status_text}", (75, 573), 0.72, (255, 255, 255), 2)

        # FPS
        draw_panel(display, 780, 510, 1000, 580, alpha=0.67)
        put_text(display, f"FPS: {avg_fps:.1f}", (806, 548), 0.78, (0, 255, 255), 2)

        # Metric box
        if calibration_done:
            draw_panel(display, 680, 20, 1004, 126, alpha=0.62)
            put_text(display, f"EAR: {ear_smooth:.2f}/{EAR_THRESHOLD:.2f}", (700, 55), 0.58,
                     (0, 255, 0) if ear_smooth >= EAR_THRESHOLD else (0, 0, 255), 1)
            put_text(display, f"MAR: {mar_smooth:.2f}/{MAR_THRESHOLD:.2f}", (700, 86), 0.58,
                     (255, 200, 0) if mar_smooth <= MAR_THRESHOLD else (0, 0, 255), 1)
            put_text(display, f"P/Y: {pitch_smooth - PITCH_NORM:+.1f}/{yaw_smooth - YAW_NORM:+.1f}",
                     (700, 116), 0.58, (255, 255, 255), 1)

        cv2.imshow("DMS", display)
        key = cv2.waitKey(1) & 0xFF
        if should_exit or key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
