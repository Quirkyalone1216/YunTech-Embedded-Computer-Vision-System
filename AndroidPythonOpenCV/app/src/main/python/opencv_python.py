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
    # 1. 解碼影像
    image = cv2.imdecode(np.asarray(bytearray(data), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print("無法解析影像")
        return None
    # 2. 灰階處理與二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # 3. 輪廓偵測
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("找不到輪廓")
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
    if len(approx) != 4:
        print("無法偵測四邊形")
        return None
    # 4. 整理角點順序：左上、右上、右下、左下
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # 左上
    rect[2] = pts[np.argmax(s)]        # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]     # 右上
    rect[3] = pts[np.argmax(diff)]     # 左下
    # 計算傾斜角度 (以左下與右下為基準)
    # 計算底邊從左下到右下的向量
    dx = rect[2][0] - rect[3][0]
    dy = rect[2][1] - rect[3][1]
    # 使用 arctan2 計算角度，再轉換成度數 (結果為角度值，水平為 0 度)
    tiltAngle = np.degrees(np.arctan2(dy, dx))
    # 5. 建立透視校正目標點
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst_rect = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # 6. 透視變換
    M_horiz = cv2.getPerspectiveTransform(rect, dst_rect)
    horizontal_image = cv2.warpPerspective(image, M_horiz, (maxWidth, maxHeight))

    # 7. 貼回原始大小畫布中央（可選）
    canvas = np.zeros_like(image)
    h_warped, w_warped = horizontal_image.shape[:2]
    start_y = (canvas.shape[0] - h_warped) // 2
    start_x = (canvas.shape[1] - w_warped) // 2
    canvas[start_y:start_y+h_warped, start_x:start_x+w_warped] = horizontal_image

    # 8. 編碼成 JPEG byte 並回傳，同時回傳傾斜角度 (轉為 float)
    success, img_bytes = cv2.imencode('.jpeg', canvas)
    if not success:
        print("JPEG 編碼失敗")
        return None

    return img_bytes.tobytes(), float(tiltAngle)