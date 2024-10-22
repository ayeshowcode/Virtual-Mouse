import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize webcam and hand detection
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            # Get coordinates of index tip (id = 8) and thumb tip (id = 4)
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]

            # Convert to pixel coordinates
            index_x = int(index_tip.x * frame_width)
            index_y = int(index_tip.y * frame_height)
            thumb_x = int(thumb_tip.x * frame_width)
            thumb_y = int(thumb_tip.y * frame_height)

            # Draw circles on landmarks
            cv2.circle(frame, (index_x, index_y), 10, (0, 0, 255), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 0, 255), -1)

            # Move mouse to index finger position
            pyautogui.moveTo(screen_width / frame_width * index_x, screen_height / frame_height * index_y)

            # Check distance between thumb and index finger tips
            distance = calculate_distance(index_x, index_y, thumb_x, thumb_y)
            print(f"Distance: {distance}")  # Debugging print

            if distance < 20:  # Adjust threshold if needed
                print("Click detected")
                pyautogui.click()
                pyautogui.sleep(1)

    # Display the webcam feed
    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
