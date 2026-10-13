import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
import time

model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

webcam = cv2.VideoCapture(0)
last_action_time = 0
COOLDOWN = 0.2  # saniye

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

with vision.HandLandmarker.create_from_options(options) as detector:
    while True:
        success, image = webcam.read()
        if not success:
            print("Camera could not be read!")
            break

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                h, w, _ = image.shape

                # Landmarks
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

                # Skeleton lines
                for connection in HAND_CONNECTIONS:
                    start = hand_landmarks[connection[0]]
                    end = hand_landmarks[connection[1]]
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(image, start_point, end_point, (255, 0, 0), 2)

                # Volume control
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
                distance = math.hypot(
                    thumb_tip.x - index_tip.x,
                    thumb_tip.y - index_tip.y
                )

                current_time = time.time()
                if current_time - last_action_time > COOLDOWN:
                    if distance > 0.1:
                        pyautogui.press('volumeup')
                        last_action_time = current_time
                    elif distance < 0.05:
                        pyautogui.press('volumedown')
                        last_action_time = current_time

                # Show distance on screen (for debugging)
                cv2.putText(image, f"Distance: {distance:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Hand volume control using python", image)
        key = cv2.waitKey(10)
        if key == 27:
            break

webcam.release()
cv2.destroyAllWindows()