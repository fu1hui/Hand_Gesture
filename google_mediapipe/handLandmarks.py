import cv2
import mediapipe as mp
import time
import os

# 导入 MediaPipe Tasks 模块
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
MPImage = mp.Image


# 设置模型路径
model_path = 'google_mediapipe/model.task'

# 设置摄像头索引, 默认为0
canmera_index = 1

gesture_label = "one"  # 手势标签名
save_dir = "gesture_dataset"

# 录制相关变量初始化
record_duration = 3  # 录制时长（秒）
skip_frame = 3       # 每隔多少帧采样一次
os.makedirs(save_dir, exist_ok=True)

hand_sides = ['right', 'left']
current_hand_index = 0

# 打印识别结果
def print_result(result: HandLandmarkerResult, output_image: MPImage, timestamp_ms: int):
    global latest_result
    latest_result = result  # 将结果缓存到全局变量中，供后续使用



options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# 初始化识别器
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(canmera_index)  # 使用默认摄像头
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    timestamp = 0
    latest_result = None

    recording = False
    start_time = 0
    frame_count = 0
    recorded_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break
        frame = cv2.flip(frame, 1)  # 镜像翻转摄像头图像

        # 转换颜色顺序 BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 获取当前时间戳（毫秒）
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        key = cv2.waitKey(1) & 0xFF
        # 开始录制逻辑
        if not recording and key == ord('s'):
            recording = True
            start_time = time.time()
            frame_count = 0
            recorded_data = []
            print("Start recording")

        # 如果有结果就绘制
        if latest_result:
            for hand_landmarks in latest_result.hand_landmarks:
                # 定义关键点之间的连接关系（基于 MediaPipe 的21个手部关键点）
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
                    (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
                    (5, 9), (9,10), (10,11), (11,12),      # 中指
                    (9,13), (13,14), (14,15), (15,16),     # 无名指
                    (13,17), (17,18), (18,19), (19,20),    # 小指
                    (0,17)                                 # 掌根连接小指底部
                ]
                # 绘制连接线
                for start_idx, end_idx in connections:
                    x0 = int(hand_landmarks[start_idx].x * frame.shape[1])
                    y0 = int(hand_landmarks[start_idx].y * frame.shape[0])
                    x1 = int(hand_landmarks[end_idx].x * frame.shape[1])
                    y1 = int(hand_landmarks[end_idx].y * frame.shape[0])
                    cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                for lm in hand_landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # 显示提示文本
        elapsed = time.time() - start_time if recording else 0
        if not recording and current_hand_index < len(hand_sides):
            cv2.putText(frame, f"Press 's' to record {hand_sides[current_hand_index]} hand", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif recording:
            cv2.putText(frame, "Recording... {:.1f}s".format(elapsed), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif current_hand_index >= len(hand_sides):
            cv2.putText(frame, "Both hands recorded. Press 'ESC' to exit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 录制数据逻辑
        if recording:
            elapsed = time.time() - start_time
            if elapsed <= record_duration:
                if frame_count % skip_frame == 0 and latest_result and latest_result.hand_landmarks:
                    data = []
                    for lm in latest_result.hand_landmarks[0]:
                        data.extend([lm.x, lm.y, lm.z])
                    recorded_data.append(data)
                frame_count += 1
            else:
                print(f"录制完成，共记录 {len(recorded_data)} 帧")
                hand_side = hand_sides[current_hand_index]
                file_path = os.path.join(save_dir, f"{gesture_label}_{hand_side}_gesture.csv")
                with open(file_path, 'w', newline='') as f:
                    import csv
                    writer = csv.writer(f)
                    writer.writerows(recorded_data)
                print(f"Saved: {file_path}")
                recording = False
                current_hand_index += 1
                cv2.putText(frame, "Recording completed. Press 'ESC' to exit", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Hand Landmarks - Real Time', frame)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
