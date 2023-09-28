import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import time, math
class GestureRecognizer:
    def main(self):
        num_hands = 2
        # model_path = 'keep/model/gesture_recognizer.task'
        # model_path = 'keep/model/gesture_recognizer_train.task'
        model_path = 'keep/model/gesture_recognizer_ok.task'
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.lock = threading.Lock()
        self.current_gestures = []
        options = GestureRecognizerOptions(base_options=python.BaseOptions(model_asset_path=model_path), running_mode=VisionRunningMode.LIVE_STREAM, num_hands = num_hands, result_callback=self.__result_callback)
        recognizer = GestureRecognizer.create_from_options(options)

        timestamp = 0
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=num_hands, min_detection_confidence=0.65, min_tracking_confidence=0.65)

        self.count = 0

        cap = cv2.VideoCapture(0)

        self.isCount = False

        # self.fourcc = cv2.VideoWriter_fourcc(*"MP4V") # screen recoder*********
        # self.writer = cv2.VideoWriter(r'D:\Proj\HandReg\env\project\keep\video\hand_gesture_ok.mp4', self.fourcc, 20, (int(cap.get(3)), int(cap.get(4)))) # screen recoder*********

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            np_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_array)
                    recognizer.recognize_async(mp_image, timestamp)
                    timestamp = timestamp + 1
                    
                list_hands = result.multi_hand_landmarks

                # print('hand point: ', list_hands)                
                self.put_gestures(frame, list_hands)

            # self.writer.write(frame) # screen recoder*********

            cv2.imshow('MediaPipe Hands', frame)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        
        # self.writer.release() # screen recoder*********
        cap.release()

    def put_gestures(self, frame, list_hands):

        self.lock.acquire()
        gestures = self.current_gestures
        self.lock.release()
        y_pos = 200

        # print(type(list_hands))

        # print(landmark)

        frame_width, frame_height = frame.shape[1], frame.shape[0]

        pixel_landmark = []

        for i, item in enumerate(list_hands):
            # print(landmark_string)
            # print(item)
            # print(item.landmark)
            for i , landmark in enumerate(item.landmark):
                # print(landmark)
                x_pixel = int(landmark.x * frame_width)
                y_pixel = int(landmark.y * frame_height)
                pixel_landmark.append((x_pixel, y_pixel))

        # print(pixel_landmark)

        for hand_gesture_name in gestures:

            # cv2.putText(frame, "{}".format(hand_gesture_name), (50, y_pos), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

            # Point on hands #############
            x4, y4 = pixel_landmark[4][0], pixel_landmark[4][1]
            x8, y8 = pixel_landmark[8][0], pixel_landmark[8][1] 
            distance_ok = math.sqrt((x8-x4)**2 + (y8-y4)**2)
            # print('dist: ',distance_ok)
            # cv2.putText(frame, "4", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
            # cv2.putText(frame, "8", (x8, y8), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

            if (hand_gesture_name == 'ok' and distance_ok <= 20):
                self.count += 1
                if self.count > 15:
                    cv2.putText(frame, "OK", (50, y_pos), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
            else:
                self.count = 0

            if (hand_gesture_name == 'Open_Palm'):
                # print('hello')
                self.start_time = time.time()
                self.isCount = True
                # print('Open_Palm: ', self.start_time)

            if (hand_gesture_name == 'Closed_Fist' and self.isCount):
                self.end_time = time.time()
                self.total_time = self.end_time - self.start_time
            
                if self.total_time < 5:
                    cv2.putText(frame, "HELP MEEEEEE!!!", (50, y_pos), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
                    
            y_pos += 100
    
    def __result_callback(self, result, output_image, timestamp_ms):
    # def __result_callback(self, result):
        self.lock.acquire()
        self.current_gestures = []
        if result is not None and any(result.gestures):
            # print("Recognized gestures: ")
            for single_hand_gesture_data in result.gestures:
                gesture_name = single_hand_gesture_data[0].category_name
                # print(gesture_name)
                                            
                self.current_gestures.append(gesture_name)
        self.lock.release()

if __name__ == "__main__":
    rec = GestureRecognizer()
    rec.main()