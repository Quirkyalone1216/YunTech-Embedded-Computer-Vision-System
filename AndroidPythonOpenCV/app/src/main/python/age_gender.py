import numpy as np
import cv2
import tensorflow as tf           # for Keras model and gradients
from tensorflow.keras import backend as K
from PIL import Image, ImageDraw, ImageFont
import os

SCALE = 4

# 全域預設 backbone，可設為 'vggface' 或 'resnet'
BACKBONE_TYPE = 'vggface'  # 如使用 ResNet backbone，可改成 'resnet'

# ------------------------------------------------------------------
# 新增：根據 BACKBONE_TYPE 選擇預處理函式
# ------------------------------------------------------------------
def preprocess_backbone(x: np.ndarray) -> np.ndarray:
    """
    根據全域 BACKBONE_TYPE，對輸入影像 x 進行相應的 preprocess_input 處理。
    """
    if BACKBONE_TYPE.lower().startswith('resnet'):
        pass
    else:
        return vggface_preprocess_input(x)


def vggface_preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp


# draw text on screen
def putText(frame, text, color, location, size=20):
    """
    在 frame 上使用 OpenCV 畫文字，以避免 PIL font 大小無法調整問題。
    location: (x, y) 左下角起點 (OpenCV 座標系)
    size: 文字高度近似像素值
    """
    # 計算 fontScale: OpenCV 字體高度約與 fontScale*30 相關，需視情況微調
    # 這裡假設 size 對應約字體高度，fontScale = size / 30
    font_scale = size / 30.0
    # 設定字型與粗細
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(size / 20))
    # OpenCV putText 起點是 baseline 底部位置，因此 location 給定的 y 需為 baseline 座標
    x, y = location
    # 在部分情況下，若 location 是相對於頂部可微調：此處假設 location 為 baseline 座標
    cv2.putText(frame, text, (int(x), int(y)), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return frame


# ------------------------------------------------------------------
# 0. Grad-CAM: 產生熱力圖
# ------------------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=0):
    # 建立 grad model，輸出最後 conv 層特徵圖與目標輸出
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output[pred_index]]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

# ------------------------------------------------------------------
# 1. 載入 Keras 模型
# ------------------------------------------------------------------
def load_keras_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# ------------------------------------------------------------------
# 新增：載入 TFLite 模型
# ------------------------------------------------------------------
def load_tflite_model(model_path):
    """
    載入 TFLite 模型，並回傳 Interpreter 實例。
    注意：TFLite Interpreter 不支援梯度計算，Grad-CAM 無法使用。
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# ------------------------------------------------------------------
# 2. 初始化 Haar Cascade（人臉偵測器）
# ------------------------------------------------------------------
def init_face_detector():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

# ------------------------------------------------------------------
# 3. 處理單張人臉 Region，做推論 + Grad-CAM
# ------------------------------------------------------------------
def predict_and_cam(model, last_conv_layer, face_img):
    # 選擇預處理：根據 BACKBONE_TYPE 動態使用 VGGFace 或 ResNet 的 preprocess_input
    x = face_img.astype(np.float32)
    x = preprocess_backbone(x)
    inp = np.expand_dims(x, axis=0)
    # 如果是 TFLite Interpreter，使用 Interpreter 進行推論；否則用 Keras model
    if isinstance(model, tf.lite.Interpreter):
        # TFLite 不支援梯度，無法產生 Grad-CAM
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        # 假設只有一個輸入 tensor
        model.set_tensor(input_details[0]['index'], inp)
        model.invoke()
        # 假設輸出有兩個 tensors: gender output, age output (依模型順序)
        # 依 output_details 順序取值
        out_tensors = [model.get_tensor(d['index']) for d in output_details]
        # 取前兩個輸出，順序需與 TFLite 模型匯出時一致
        preds_gender = out_tensors[0]
        preds_age = out_tensors[1]
        heatmap = None
        return float(preds_age[0][0]), preds_gender[0][0], heatmap
    else:
        # 取得預測（直接呼叫 model，避免 DataAdapter 匯入 pandas）
        outputs = model(inp, training=False)
        # 若模型回傳多個輸出，取前兩作為 gender 和 age
        preds_gender, preds_age = outputs[0], outputs[1]
        # 產生 Grad-CAM 熱力圖 (性別分類, 索引 0)
        heatmap = make_gradcam_heatmap(inp, model, last_conv_layer, pred_index=0)
        return float(preds_age[0][0]), preds_gender[0][0], heatmap

# ------------------------------------------------------------------
# 4. 繪製並疊加 Grad-CAM
# ------------------------------------------------------------------
def overlay_heatmap(frame, heatmap, x, y, w, h, alpha=0.4):
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame[y:y+h, x:x+w], 1 - alpha, heatmap, alpha, 0)
    frame[y:y+h, x:x+w] = overlay
    return frame

# ------------------------------------------------------------------
# 5. 繪製人臉框與屬性文字
# ------------------------------------------------------------------
def annotate_frame(frame, faces, preds_list, labels):
    gender_labels = ['Male', 'Female']

    for (x, y, w, h), (age_val, gender_prob, heatmap) in zip(faces, preds_list):
        # CAM 疊加：若 heatmap 為 None，表示 TFLite 推論無法產生 Grad-CAM，跳過
        if heatmap is not None:
            # 若要顯示，可解除下行註解
            # frame = overlay_heatmap(frame, heatmap, x, y, w, h)
            pass
        # 文字
        gen_idx = 1 if gender_prob > 0.5 else 0
        age_text = f'Age: {age_val:.1f}'
        gender_text = f'Gender: {gender_labels[gen_idx]} ({gender_prob:.2f})'
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 調整文字顯示位置：畫在框下方，避免過度往上偏移導致文字出界
        int_x = int(x)
        int_y = int(y)
        int_w = int(w)
        int_h = int(h)
        # 在人臉框下方顯示：第一行 Age，第二行 Gender
        # 調整文字間隔與大小：將 text_size 調小
        text_size = 10 * SCALE  # 原先 20*SCALE 太大，改為 10*SCALE，可再微調
        margin = 5  # 減少行間距
        base_y = int_y + int_h + margin
        # Age 文字
        frame = putText(frame, age_text, (0, 255, 0), (int_x, base_y), size=text_size)
        # Gender 文字：下一行
        frame = putText(frame, gender_text, (0, 255, 0), (int_x, base_y + text_size + margin), size=text_size)
    return frame

# ------------------------------------------------------------------
# 6. 處理影格：偵測 → 裁切 → 推論+CAM → 標註
# ------------------------------------------------------------------
def process_frame(frame, face_cascade, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(100, 100))
    preds_list = []
    for (x, y, w, h) in faces:
        # 裁切並 resize，保留三通道彩色輸入 (128,128,3)
        face = cv2.resize(frame[y:y+h, x:x+w], (128, 128))
        # 如模型在 RGB 空間訓練，可加這行進行 BGR→RGB 轉換
        face_img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # 動態決定最後 conv 層名稱：VGGFace 常為 'conv5_3'，ResNet50 常為 'conv5_block3_out'
        if BACKBONE_TYPE.lower().startswith('resnet'):
            last_conv = 'conv5_block3_out'
        else:
            last_conv = 'conv5_3'
        age, gender_prob, heatmap = predict_and_cam(model, last_conv, face_img)
        preds_list.append((age, gender_prob, heatmap))
    if faces is not None and len(faces) > 0:
        frame = annotate_frame(frame, faces, preds_list, None)
    return frame

# ------------------------------------------------------------------
# 全域 Interpreter 與 Cascade
# ------------------------------------------------------------------
_interpreter = None
_face_cascade = None

def init(model_path):
    """
    初始化 TFLite Interpreter 與人臉偵測器，請在 Android 端啟動後呼叫一次，之後重複呼 process_realtime 即可。
    """
    global _interpreter, _face_cascade
    # 確認模型檔存在
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found at {model_path}")
    # 載入 TFLite Interpreter
    _interpreter = load_tflite_model(model_path)
    # 初始化人臉偵測
    _face_cascade = init_face_detector()
    return True

def process_realtime(nv21_bytes: bytes, w: int, h: int) -> bytes:
    """
    使用 NV21 bytes (寬 w, 高 h) 作為輸入，進行人臉偵測與推論，回傳 PNG bytes。
    要先呼 init(model_path) 完成初始化。
    """
    if _interpreter is None or _face_cascade is None:
        raise RuntimeError("Interpreter or face detector not initialized. Call init(model_path) first.")
    try:
        # NV21 -> BGR
        print(f"[age_gender] process_realtime: w={w}, h={h}, nv21_bytes_len={len(nv21_bytes)}")
        yuv = np.frombuffer(nv21_bytes, dtype=np.uint8)
        expected = h * 3 // 2 * w
        if yuv.size != expected:
            print(f"[age_gender] Warning: NV21 size mismatch: got {yuv.size}, expected {expected}")
        yuv = yuv.reshape((h * 3 // 2, w))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        # 偵測與推論
        out_frame = process_frame(bgr, _face_cascade, _interpreter)
        # 旋轉顯示（如有需要）
        disp = cv2.rotate(out_frame, cv2.ROTATE_90_CLOCKWISE)
        ok, buf = cv2.imencode('.png', disp)
        if not ok:
            print("[age_gender] encode failed")
            # 回傳空白 PNG
            blank = np.zeros((1,1,3), dtype=np.uint8)
            _, b2 = cv2.imencode('.png', blank)
            return b2.tobytes()
        return buf.tobytes()
    except Exception as e:
        import traceback
        print("[age_gender] Error in process_realtime:", e)
        traceback.print_exc()
        # 返回最小空白 PNG 避免 Java 端崩潰
        blank = np.zeros((1,1,3), dtype=np.uint8)
        _, b2 = cv2.imencode('.png', blank)
        return b2.tobytes()


# ------------------------------------------------------------------
# 8. 靜態圖片辨識
# ------------------------------------------------------------------
def process_image(image_path, face_cascade, model, save_path=None):
    """對單張圖片做人臉偵測、推論 + Grad-CAM，並顯示或儲存結果。"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"無法讀取圖片：{image_path}")
        return
    out = process_frame(img, face_cascade, model)
    if save_path:
        cv2.imwrite(save_path, out)
        print(f"結果已儲存至 {save_path}")
    else:
        # 縮小視窗為原尺寸 1/scale
        h, w = out.shape[:2]
        small = cv2.resize(out, (w // SCALE, h // SCALE))
        print(f"[age_gender] display static image size {small.shape}")
        cv2.imshow('Static Image Inference', small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ------------------------------------------------------------------
# 9. 主程式
# ------------------------------------------------------------------
def main():
    # 初始化偵測器
    face_cascade = init_face_detector()
    # 靜態圖片路徑
    img_path = r'tmp\archive\train\21-30\8.jpg'
    os.makedirs('results', exist_ok=True)
    # 定義多個模型與對應 backbone
    model_configs = [
        (r'face_model.tflite', 'vggface'),
        # (r'model\best_model_resnet.h5', 'resnet'),
    ]
    for model_path, backbone in model_configs:
        # 設定全域 BACKBONE_TYPE
        global BACKBONE_TYPE
        BACKBONE_TYPE = backbone
        # 載入模型：依副檔名決定 Keras 還是 TFLite
        if model_path.lower().endswith('.tflite'):
            model = load_tflite_model(model_path)
        else:
            model = load_keras_model(model_path)
        # process_image(img_path, face_cascade, model)
        # 啟動即時辨識
        process_realtime(face_cascade, model)

