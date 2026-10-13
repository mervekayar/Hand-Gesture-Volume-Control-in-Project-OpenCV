import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math

# Model file (must be in the same folder as this script)
model_path = 'hand_landmarker.task'

# Prepare the detector
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

# Camera and detector
webcam = cv2.VideoCapture(0)
detector = vision.HandLandmarker.create_from_options(options)

while True:
    # Read a frame from the camera
    success, image = webcam.read()

    # OpenCV reads in BGR, but MediaPipe expects RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Detect hands
    detection_result = detector.detect(mp_image)

    # If a hand is found
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:

            # Draw each landmark point on the screen
            h, w, _ = image.shape
            for landmark in hand_landmarks:
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

            # Thumb tip (4) and index finger tip (8)
            thumb = hand_landmarks[4]
            index = hand_landmarks[8]

            # Calculate the distance between the two fingers
            distance = math.hypot(thumb.x - index.x, thumb.y - index.y)

            # Adjust volume based on distance
            if distance > 0.1:
                pyautogui.press('volumeup')
            elif distance < 0.05:
                pyautogui.press('volumedown')

    # Show the frame
    cv2.imshow("Hand volume control using python", image)

    # Exit if ESC is pressed
    key = cv2.waitKey(10)
    if key == 27:
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()