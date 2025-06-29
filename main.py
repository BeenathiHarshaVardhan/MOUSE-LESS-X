import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize screen size
screen_w, screen_h = pyautogui.size()

# Initialize MediaPipe and Webcam
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Smoothing parameters
prev_x, prev_y = 0, 0
smoothening = 5

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)  # Flip image for natural movement
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Index finger tip (id=8), Thumb tip (id=4)
            x_index, y_index = lm_list[8][1:]
            x_thumb, y_thumb = lm_list[4][1:]

            # Move mouse based on index finger
            screen_x = screen_w * (x_index / w)
            screen_y = screen_h * (y_index / h)

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Distance between thumb and index finger
            distance = math.hypot(x_thumb - x_index, y_thumb - y_index)

            # Click if fingers are close
            if distance < 30:
                pyautogui.click()
                cv2.putText(img, "Click", (x_index + 10, y_index - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the webcam feed
    cv2.imshow("Hand Gesture Mouse Control", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()