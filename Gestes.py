import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras


class Gestes:
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
        self.actions = np.array(['hello', 'thanks'])
        self.model = Sequential()
        self.load_model = keras.models.load_model("action1.h5")
        self.res = [0.7, 0.3]
        self.cap = cv2.VideoCapture(0)
        self.Gesture_detection()

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def extract_key_points(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
        ) if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
        ) if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
        ) if results.right_hand_landmarks else np.zeros(21*3)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
        ) if results.face_landmarks else np.zeros(468*3)
        return np.concatenate([pose, face, lh, rh])

    def prob_viz(self, input_frame):
        output_frame = input_frame.copy()
        for num, prob in enumerate(self.res):
            cv2.rectangle(output_frame, (0, 60+num*40),
                          (int(prob*100), 90+num*40), self.colors[num], -1)
            cv2.putText(output_frame, self.actions[num], (0, 85+num*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return output_frame

    def Gesture_detection(self):
        sequence = []
        sentence = []
        threshold = .4

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                image, results = self.mediapipe_detection(frame, holistic)
                key_points = self.extract_key_points(results)
                sequence.insert(0, key_points)
                sequence = sequence[:30]
                self.res = [0.5, 0.5]
                if len(sequence) == 30:
                    self.res = self.load_model.predict(
                        np.expand_dims(sequence, axis=0))[0]
                # print(f"Action Probabilities: {self.res}")
                if self.res[np.argmax(self.res)] > threshold:
                    if len(sentence) > 0:
                        if self.actions[np.argmax(self.res)] != sentence[-1]:
                            sentence.append(self.actions[np.argmax(self.res)])
                    else:
                        sentence.append(self.actions[np.argmax(self.res)])

                if len(sentence) > 5:
                    sentence = sentence[-1]
                # image = self.prob_viz(image)
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    gt = Gestes()
