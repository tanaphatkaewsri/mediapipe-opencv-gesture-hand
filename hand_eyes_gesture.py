import time, cv2, threading, math
import mediapipe as mp, numpy as np
from mediapipe.tasks import python
# from scipy.spatial import distance as dist
# from imutils import face_utils

class GestureRecognizer:
    def main(self):
        num_hands = 2

        model_path1 = 'D:/Proj/HandReg/env/project/keep/model/gesture_recognizer.task'
        model_path2 = 'D:/Proj/HandReg/env/project/keep/model/gesture_recognizer_ok.task'

        GestureRecognizer1 = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions1 = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode1 = mp.tasks.vision.RunningMode

        self.lock1 = threading.Lock()
        self.current_gestures1 = []

        self.lock2 = threading.Lock()
        self.current_gestures2 = []

        options1 = GestureRecognizerOptions1(base_options=python.BaseOptions(model_asset_path=model_path1), running_mode=VisionRunningMode1.LIVE_STREAM, num_hands = num_hands, result_callback=self.__result_callback1)

        options2 = GestureRecognizerOptions1(base_options=python.BaseOptions(model_asset_path=model_path2), running_mode=VisionRunningMode1.LIVE_STREAM, num_hands = num_hands, result_callback=self.__result_callback2)


        recognizer1 = GestureRecognizer1.create_from_options(options1)

        recognizer2 = GestureRecognizer1.create_from_options(options2)


        timestamp = 0
        timestamp2 = 0
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=num_hands, min_detection_confidence=0.65, min_tracking_confidence=0.65)

        self.count = 0
        self.count2 = 0

        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture("rtsp://admin:admin@192.168.7.160:554/2/1")
        # cap = cv2.VideoCapture("rtsp://admin:$Gets12345@172.24.1.9:30180/live")

        self.isCount = False

        # self.fourcc = cv2.VideoWriter_fourcc(*"MP4V") # screen recoder*********
        # self.writer = cv2.VideoWriter(r'D:/Proj/HandReg/env/project/keep/video/eyeshand_gesture250966.mp4', self.fourcc, 20, (int(cap.get(3)), int(cap.get(4)))) # screen recoder*********

        COUNTER = 0
        TOTAL_BLINKS = 0
        WITHIN = 1
        CLOSED_EYES_FRAME = 1
        FONT_SIMP = cv2.FONT_HERSHEY_SIMPLEX
        FONT_COMP = cv2.FONT_HERSHEY_COMPLEX

        LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
        RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 

        iscount = False
        c_time = None
        oldblink = 0

        map_face_mesh = mp.solutions.face_mesh
        face_mesh = map_face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            np_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if result.multi_hand_landmarks:
                # list_hands = result.multi_hand_landmarks
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_array)
                    recognizer1.recognize_async(mp_image, timestamp)
                    timestamp = timestamp + 1
                    recognizer2.recognize_async(mp_image, timestamp2)
                    timestamp2 = timestamp2 + 1
                
                self.put_gestures(frame)

            #eye blinks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_coords = self.landmarksDetection(frame, results, False)
                ratio = self.blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

                if ratio > 4: # close eye
                    COUNTER += 1
                    # print(COUNTER)
                else:
                    if COUNTER > CLOSED_EYES_FRAME: #For blink and count
                        TOTAL_BLINKS += 1
                        COUNTER = 0

                if TOTAL_BLINKS > 0:
                    if not iscount:
                        s_time = time.time()
                        oldblink += 1
                        iscount = True
                    else:
                        c_time = time.time()
                        if c_time-s_time < WITHIN and TOTAL_BLINKS > oldblink:
                            s_time = time.time()
                            iscount = False
                        elif c_time-s_time > WITHIN:
                            s_time = time.time()
                            iscount = False
                            TOTAL_BLINKS = oldblink = 0

                if TOTAL_BLINKS >= 3:
                    cv2.rectangle(frame, (5, 70), (220, 100), (255, 255, 255), -1)
                    cv2.putText(frame, f"Eyes detected", (10, 90), FONT_SIMP, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame, (50, 210), (410, 150), (255, 255, 255), -1)
                    cv2.putText(frame, "HELP ME!!!", (50, 200), FONT_COMP, 2, (0,0,255), 2, cv2.LINE_AA)

                if iscount and c_time is not None:
                    cv2.putText(frame, f"time(s): {c_time-s_time:.2f}", (10, 60), FONT_SIMP, 0.7, (0, 0, 255), 2)

                cv2.putText(frame, "Blinks: {}".format(TOTAL_BLINKS), (10, 30), FONT_SIMP, 0.7, (0, 0, 255), 2)

                cv2.rectangle(frame, (3, frame.shape[0]-50), (frame.shape[1]-5, frame.shape[0]-10), (255, 255, 255), -1)
                cv2.putText(frame, "blink 3 times continuously = need help", (6, frame.shape[0]-20), FONT_SIMP, 1, (0, 0, 255), 2)

            # self.writer.write(frame) # screen recoder*********
            cv2.imshow('MediaPipe Hands', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            elif cv2.waitKey(1) == 32:
                help = False
        
        # self.writer.release() # screen recoder*********
        cap.release()
    
    def landmarksDetection(self, img, results, draw=False):
        img_height, img_width = img.shape[:2]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
        if draw:
            [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
        return mesh_coord

    # Euclaidean distance 
    def euclaideanDistance(self, point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
        return distance

    # Blinking Ratio
    def blinkRatio(self, img, landmarks, right_indices, left_indices):
        # Right eyes 
        # horizontal line 
        rh_right = landmarks[right_indices[0]]
        rh_left = landmarks[right_indices[8]]
        # vertical line 
        rv_top = landmarks[right_indices[12]]
        rv_bottom = landmarks[right_indices[4]]
        # draw lines on right eyes 
        # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
        # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

        # LEFT_EYE 
        # horizontal line 
        lh_right = landmarks[left_indices[0]]
        lh_left = landmarks[left_indices[8]]

        # vertical line 
        lv_top = landmarks[left_indices[12]]
        lv_bottom = landmarks[left_indices[4]]

        rhDistance = self.euclaideanDistance(rh_right, rh_left)
        rvDistance = self.euclaideanDistance(rv_top, rv_bottom)

        lvDistance = self.euclaideanDistance(lv_top, lv_bottom)
        lhDistance = self.euclaideanDistance(lh_right, lh_left)

        reRatio = rhDistance/rvDistance
        leRatio = lhDistance/lvDistance

        ratio = (reRatio+leRatio)/2
        return ratio

    def put_gestures(self, frame):
        self.lock1.acquire()
        gestures = self.current_gestures1
        self.lock1.release()
        y_pos = 200

        self.lock2.acquire()
        gestures2 = self.current_gestures2
        self.lock2.release()

        # print(gestures2)

        for hand_gesture_name in gestures2:
            if (hand_gesture_name == 'ok'):
                self.count += 1
                if self.count > 15:
                    cv2.rectangle(frame, (50, 210), (160, 150), (255, 255, 255), -1)
                    cv2.putText(frame, "OK", (50, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
            else:
                self.count = 0
        
        for hand_gesture_name in gestures:

            # print(hand_gesture_name)

            if (hand_gesture_name == 'Thumb_Up'):
                # print('count: ', self.count2)
                self.count2 += 1
                if self.count2 > 15:
                    cv2.rectangle(frame, (50, 210), (210, 150), (255, 255, 255), -1)
                    cv2.putText(frame, "Good", (50, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
            else:
                self.count2 = 0

            if (hand_gesture_name == 'Open_Palm'):
                # print('hello')
                self.start_time = time.time()
                self.isCount = True
                # print('Open_Palm: ', self.start_time)

            if (hand_gesture_name == 'Closed_Fist' and self.isCount):
                self.end_time = time.time()
                self.total_time = self.end_time - self.start_time
            
                if self.total_time < 3:
                    cv2.rectangle(frame, (5, 70), (220, 100), (255, 255, 255), -1)
                    cv2.putText(frame, f"Hand detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.rectangle(frame, (50, 210), (410, 150), (255, 255, 255), -1)
                    cv2.putText(frame, "HELP ME!!!", (50, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

            y_pos += 100
    
    def __result_callback1(self, result, output_image, timestamp_ms):
        self.lock1.acquire()
        self.current_gestures1 = []
        if result is not None and any(result.gestures):
            # print("Recognized gestures: ")
            for single_hand_gesture_data in result.gestures:
                gesture_name = single_hand_gesture_data[0].category_name
                # print(gesture_name)                
                            
                self.current_gestures1.append(gesture_name)
        self.lock1.release()

    def __result_callback2(self, result, output_image, timestamp_ms):
        self.lock2.acquire()
        self.current_gestures2 = []
        if result is not None and any(result.gestures):
            # print("Recognized gestures: ")
            for single_hand_gesture_data in result.gestures:
                gesture_name = single_hand_gesture_data[0].category_name
                # print(gesture_name)                
                            
                self.current_gestures2.append(gesture_name)
        self.lock2.release()

if __name__ == "__main__":
    rec = GestureRecognizer()
    rec.main()