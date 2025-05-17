import cv2
import mediapipe as mp
import time

# 导入 MediaPipe Tasks 模块
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
MPImage = mp.Image

# 打印识别结果
def print_result(result: HandLandmarkerResult, output_image: MPImage, timestamp_ms: int):
    global latest_result
    latest_result = result  # 将结果缓存到全局变量中，供后续绘图使用

# 设置模型路径
model_path = 'google_mediapipe\model.task'

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# 初始化识别器
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    timestamp = 0
    latest_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        # 转换颜色顺序 BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 获取当前时间戳（毫秒）
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        # 如果有结果就绘制
        if latest_result:
            for hand_landmarks in latest_result.hand_landmarks:
                for lm in hand_landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        cv2.imshow('Hand Landmarks - Real Time', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
