import cv2
import pickle
import numpy as np
import os

# —————————————— 全域參數 ——————————————
MODEL_PATH        = 'clf.pkl'
IM_SIZE           = (200, 300)         # (高, 寬)
THRESHOLD_BG      = 17
KERNEL            = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
CONFIDENCE_THRESH = 0.6
DEFECT_DEPTH_TH   = 1000
SOLIDITY_TH       = 0.9                # Solidity > 0.9 → rock
EXTENT_TH         = 0.65               # Extent  > 0.65 → rock

# —————————————— rpscv.imgproc.getGray ——————————————
def getGray(img, hueValue=63, threshold=0):
    """Returns the grayscale of the source image with its background
    removed as a 1D feature vector."""
    img = removeBackground(img, hueValue, threshold)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    return img.ravel()

def removeBackground(img, hueValue, threshold=0):
    dist = hueDistance(img, hueValue)
    masked = img.copy()
    th = float(dist.mean()) if threshold == 0 else threshold
    masked[dist < th] = 0
    return masked

def hueDistance(img, hueValue):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hChannel = hsv[:, :, 0].astype(int)
    hueOffset = 180 if hueValue < 90 else -180
    return np.minimum(
        np.abs(hChannel - hueValue),
        np.abs(hChannel - (hueValue + hueOffset))
    )

# —————————————— rpscv.utils.gestureTxt ——————————————
ROCK = 0
PAPER = 1
SCISSORS = 2
gestureTxt = {ROCK: 'rock', PAPER: 'paper', SCISSORS: 'scissors'}

# 預先載入模型
MODEL_PATH = os.path.join(os.path.dirname(__file__), "clf.pkl")
_model = None
def _load_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
    return _model

# —————————————— ROI 與特徵函式 ——————————————
def _get_skin_mask(hsv):
    lower = np.array([0, 30, 60])
    upper = np.array([20, 150, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL, iterations=2)
    return mask

def _extract_hand_roi(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = _get_skin_mask(hsv)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w * h < 2000:
        return None, None
    return frame[y:y+h, x:x+w], (x, y, w, h)

def _analyze_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, 0, 0
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = max(w * h, 1)
    hull = cv2.convexHull(cnt)
    hull_area = max(cv2.contourArea(hull), 1)
    solidity = area / hull_area
    extent = area / bbox_area
    defects = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))
    dcount = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            _, _, _, depth = defects[i, 0]
            if depth > DEFECT_DEPTH_TH:
                dcount += 1
    fingers = dcount + 1
    return solidity, extent, fingers

# SVM + 規則判斷

def _svm_predict(model, roi):
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IM_SIZE[1], IM_SIZE[0]))
    feat = getGray(resized, threshold=THRESHOLD_BG).reshape(1,-1).astype(np.float32)
    scores = np.array(model.decision_function(feat)).ravel()
    conf   = float(scores.max() - scores.min())
    pred   = int(model.predict(feat)[0])
    return gestureTxt[pred], conf

def _rule_predict(fingers):
    if fingers == 2:
        return 'scissors'
    elif fingers >= 4:
        return 'paper'
    return None

def _fuse_prediction(solidity, extent, fingers, svm_label, conf):
    # 根據 fingers 數量分層：
    # fingers >= 9 → paper
    if fingers >= 9:
        return 'paper'
    # fingers 7~8 → scissors
    if fingers >= 7:
        return 'scissors'
    # fingers 5~6 → rock
    if fingers >= 5:
        return 'rock'
    # 其餘交由 SVM
    return svm_label

# 繪圖覆蓋
def _draw_overlay(frame, rect, solidity, extent, fingers, label):
    disp = frame.copy()
    if rect is not None:
        x, y, w, h = rect
        cv2.rectangle(disp, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(disp,
                    f'sol:{solidity:.2f} ext:{extent:.2f} fin:{fingers}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(disp,
                f'Gesture: {label}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    return disp

# —————————————— Android 呼叫入口 ——————————————
def process_nv21(nv21_bytes: bytes, w: int, h: int) -> bytes:
    yuv = np.frombuffer(nv21_bytes, dtype=np.uint8).reshape((h*3//2, w))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)

    roi, rect = _extract_hand_roi(bgr)
    solidity, extent, fingers = 0, 0, 0
    label = 'No hand'
    if roi is not None:
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = _get_skin_mask(hsv_roi)
        solidity, extent, fingers = _analyze_contour(mask)
        svm_label, conf = _svm_predict(_load_model(), roi)
        label = _fuse_prediction(solidity, extent, fingers, svm_label, conf)

    disp = _draw_overlay(bgr, rect, solidity, extent, fingers, label)
    # 旋轉 90°
    disp = cv2.rotate(disp, cv2.ROTATE_90_CLOCKWISE)

    ok, buf = cv2.imencode('.png', disp)
    if not ok:
        raise RuntimeError('encode failed')
    return buf.tobytes()
