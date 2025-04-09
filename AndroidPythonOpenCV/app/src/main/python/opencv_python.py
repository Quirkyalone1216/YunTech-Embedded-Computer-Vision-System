import numpy as np
import cv2

def opencv_process_image(data):
    # 读取图片数据
    image = cv2.imdecode(np.asarray(data),cv2.IMREAD_COLOR)
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 将处理后的图像转换为png格式并转换为byte数组
    is_success, im_buf_arr = cv2.imencode(".png", gray_image)
    byte_im = im_buf_arr.tobytes()

    # 返回处理后的图像数据
    return byte_im

def CalTiltAngle(data):
    """
    傾斜角度計算函數:
        輸入: data -- 影像二進位資料 (bytes)
        輸出: 傾斜角度 (度數)
    """
    # 將資料轉換成 numpy 陣列，然後用 cv2.imdecode 解碼成影像
    image = cv2.imdecode(np.asarray(bytearray(data), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("無法解碼影像資料。")
    # 轉換為灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化影像 (閾值128，可視情況調整)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # 尋找外部輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("影像中未找到任何輪廓。")
    # 取得面積最大的輪廓
    c = max(contours, key=cv2.contourArea)
    # 取得最小外接矩形及其相關參數
    rect = cv2.minAreaRect(c)
    (cx, cy), (w, h), angle = rect
    # OpenCV 返回的 angle 範圍通常為 [-90, 0)
    # 若寬度小於高度，校正角度為 angle+90
    if w < h:
        angle += 90
    return angle
