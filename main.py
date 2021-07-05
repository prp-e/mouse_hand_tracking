import autopy 
import cv2 
import mediapipe as mp 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

camera = cv2.VideoCapture(1)
mouse = Controller()

def find_landmarks(image, results, draw=False):
    landmark_list = []
    tip_ids = [8, 12, 16, 20]
    for id, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmark_list.append([id, cx, cy])
                if id in tip_ids and draw:
                    cv2.circle(image, (cx, cy), 15, (255, 0, 200), cv2.FILLED)
    
    return landmark_list

def fingers_up(landmark_list):
    tip_ids = [8, 12, 16, 20]
    fingers = []
    for i in range(4):
        if landmark_list[tip_ids[i]][2] < landmark_list[tip_ids[i] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers
            


with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands = 1) as hands:
    while camera.isOpened():
        _, image = camera.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False 
        results = hands.process(image)

        image.flags.writeable = True
        landmark_list = []
        if results.multi_hand_landmarks:
            landmark_list = find_landmarks(image=image, results=results, draw=False)
            fingers = fingers_up(landmark_list)
            
            if fingers[0] == 1 and fingers[1:4] == [0, 0, 0]:
                cv2.circle(image, (landmark_list[8][1], landmark_list[8][2]), 25, (0, 255, 0), cv2.FILLED)
                print("Moving mode...")
                print(f'Dimensions - X : {landmark_list[8][1]}, Y: {landmark_list[8][2]}')
                autopy.mouse.move(landmark_list[8][1], landmark_list[8][2])
                
            
        cv2.imshow("Camera No. 1", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xff == ord('q'):
            break 

camera.release()
cv2.destroyAllWindows()