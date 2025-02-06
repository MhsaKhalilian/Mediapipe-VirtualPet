import cv2
import mediapipe as mp

# Load videos
default_video = cv2.VideoCapture('4_5834757945434315291.mp4')
palm_video = cv2.VideoCapture('4_5834757945434315288.mp4')
korean_heart_video = cv2.VideoCapture('backup_1738515498100_.mp4')

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
playing_palm_video = False
playing_korean_heart_video = False

# Initialize detection flags
palm_detected = False
korean_heart_detected = False
gesture_in_progress = False  # To prevent re-triggering video while hand is still up

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frameRGB)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = {}
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                landmarks[id] = (int(lm.x * w), int(lm.y * h))
            
            # Debug: Print landmarks
            print("Landmarks:", landmarks)
            
            # Detect open palm (all fingers extended)
            if all(
                id in landmarks and landmarks[id][1] < landmarks[id - 2][1] 
                for id in [8, 12, 16, 20]
            ):
                if not gesture_in_progress:  # Check if video is already playing
                    print("Palm detected!")
                    palm_detected = True
                    gesture_in_progress = True  # Prevent re-triggering
                    break  # Stop checking other gestures once a gesture is detected
            
            # Detect Korean heart gesture (Thumb and index finger tips close together)
            if (8 in landmarks and 4 in landmarks and 
                abs(landmarks[8][0] - landmarks[4][0]) < 20 and 
                abs(landmarks[8][1] - landmarks[4][1]) < 20):
                if not gesture_in_progress:  # Check if video is already playing
                    print("Korean heart detected!")
                    korean_heart_detected = True
                    gesture_in_progress = True  # Prevent re-triggering
                    break  # Stop checking other gestures once a gesture is detected
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    if palm_detected and not playing_palm_video and not playing_korean_heart_video:
        print("Playing palm video...")
        playing_palm_video = True
        palm_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
        while palm_video.isOpened():
            for _ in range(2):  # Skip frames to speed up
                ret, palm_frame = palm_video.read()
            if not ret:
                break
            palm_frame = cv2.resize(palm_frame, (frame.shape[1], frame.shape[0]))  # Resize video to match camera frame
            cv2.imshow('Webcam', palm_frame)
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break
        playing_palm_video = False  # Reset after palm video ends
        palm_detected = False  # Reset after video plays
        gesture_in_progress = False  # Allow new gesture detection
    
    if korean_heart_detected and not playing_korean_heart_video:
        print("Playing Korean heart video...")
        playing_korean_heart_video = True
        korean_heart_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
        while korean_heart_video.isOpened():
            for _ in range(2):  # Skip frames to speed up
                ret, korean_heart_frame = korean_heart_video.read()
            if not ret:
                break
            korean_heart_frame = cv2.resize(korean_heart_frame, (frame.shape[1], frame.shape[0]))  # Resize video
            cv2.imshow('Webcam', korean_heart_frame)
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break
        playing_korean_heart_video = False  # Reset after video ends
        korean_heart_detected = False  # Reset after video plays
        gesture_in_progress = False  # Allow new gesture detection

    # Display default video or webcam feed when no gesture is detected
    if not playing_palm_video and not playing_korean_heart_video:
        for _ in range(2):  # Skip frames to speed up
            ret, video_frame = default_video.read()
        if not ret:
            default_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop default video
            ret, video_frame = default_video.read()
        video_frame = cv2.resize(video_frame, (frame.shape[1], frame.shape[0]))  # Resize video to match camera frame
        cv2.imshow('Webcam', video_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
default_video.release()
palm_video.release()
korean_heart_video.release()
cv2.destroyAllWindows()
