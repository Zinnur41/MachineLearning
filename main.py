import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

MY_FACE_ID = None

def determine_name_fingers(num_fingers):
    if num_fingers == 1:
        return "Zinnur"
    elif num_fingers == 2:
        return "Zagidullin"
    return "Unknown"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_detection.process(frame_rgb)

    is_my_face = False
    x, y = 0, 0

    if results_face.detections:
        for i, detection in enumerate(results_face.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * iw)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if MY_FACE_ID is None:
                MY_FACE_ID = (x, y, w, h)
            elif abs(x - MY_FACE_ID[0]) < 50 and abs(y - MY_FACE_ID[1]) < 50:
                is_my_face = True

    results_hands = hands.process(frame_rgb)
    num_fingers = 0

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            wrist = landmarks[0]
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[2]

            thumb_up = (thumb_tip.y < thumb_ip.y) and (abs(thumb_tip.x - wrist.x) > 0.05)

            fingers_up = [
                thumb_up,  # Большой палец
                landmarks[8].y < landmarks[6].y,  # Указательный
                landmarks[12].y < landmarks[10].y,  # Средний
                landmarks[16].y < landmarks[14].y,  # Безымянный
                landmarks[20].y < landmarks[18].y  # Мизинец
            ]

            num_fingers = fingers_up.count(True)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for i, tip in enumerate([4, 8, 12, 16, 20]):
                color = (0, 255, 0) if fingers_up[i] else (0, 0, 255)
                cx, cy = int(landmarks[tip].x * iw), int(landmarks[tip].y * ih)
                cv2.circle(frame, (cx, cy), 10, color, -1)

    if is_my_face and num_fingers == 0:
        name_fingers = "Zinnur Zagidullin"
    else:
        name_fingers = determine_name_fingers(num_fingers)

    if results_face.detections:
        cv2.putText(frame, name_fingers, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
