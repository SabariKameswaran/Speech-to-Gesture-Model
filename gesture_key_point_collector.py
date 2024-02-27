import cv2
import numpy as np
import mediapipe as mp
import os
import all_actions as aa

class DataKeyCollector:
    # path of image is path_ and root of image root_
    def __init__(self):
        self.path_of_root = aa.root_of_all
        self.file = 'sorry'
        self.images_path = os.path.join(aa.data_path,self.file)
        self.array_paths = os.path.join(aa.array_path,self.file)
        self.all_image_list = np.array(os.listdir(self.images_path))
        self.path_array_creation()
        self.mp_holistic = mp.solutions.holistic
        self.mp_draw = mp.solutions.drawing_utils
        self.array_extractor()

    def path_array_creation(self):
        if os.path.exists((self.array_paths)):
            print("it is already exist")
            arr_list = os.listdir(self.array_paths)
            if len(arr_list) == 900:
                print("it's over already")
                exit()
        else:
            os.makedirs(self.array_paths)

    def mediapipe_detection(self, image, model):
        # image get BGR format convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = model.process(image)  # process to convert
        image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # reconverting to BGR
        return image, result

    def draw_style_landmarks(self, image, results):
        self.mp_draw.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, self.mp_draw.DrawingSpec(
            color=(105, 237, 17), thickness=1, circle_radius=1), self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1))
        self.mp_draw.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, self.mp_draw.DrawingSpec(color=(
            105, 237, 17), thickness=1, circle_radius=1), self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2))
        self.mp_draw.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.mp_draw.DrawingSpec(color=(
            105, 237, 17), thickness=1, circle_radius=1), self.mp_draw.DrawingSpec(color=(26, 209, 209), thickness=1, circle_radius=2))
        self.mp_draw.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.mp_draw.DrawingSpec(color=(
            105, 237, 17), thickness=1, circle_radius=1), self.mp_draw.DrawingSpec(color=(26, 209, 209), thickness=1, circle_radius=2))

    def array_extractor(self):
        index = 0
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for images in self.all_image_list:
                frame = cv2.imread(os.path.join(self.images_path, images))
                image, results = self.mediapipe_detection(image=frame, model=holistic)
                self.draw_style_landmarks(image=image, results=results)
                key_points = self.extract_key_points(results)
                npy_path = os.path.join(self.array_paths,f"{self.file}_{index}.npy")
                np.save(npy_path,key_points)
                index+=1

    def extract_key_points(self, results):
        pose_array = np.array([[i.x, i.y, i.z, i.visibility] for i in results.pose_landmarks.landmark]).flatten(
        ) if results.pose_landmarks else np.zeros(33*4)
        right_hand_array = np.array([[i.x, i.y, i.z] for i in results.right_hand_landmarks.landmark]).flatten(
        ) if results.right_hand_landmarks else np.zeros(21*3)
        left_hand_array = np.array([[i.x, i.y, i.z] for i in results.left_hand_landmarks.landmark]).flatten(
        ) if results.left_hand_landmarks else np.zeros(21*3)
        face_array = np.array([[i.x, i.y, i.z] for i in results.face_landmarks.landmark]).flatten(
        ) if results.face_landmarks else np.zeros(468*3)
        return np.concatenate([pose_array, face_array, left_hand_array, right_hand_array])


if __name__ == "__main__":
    dkc = DataKeyCollector()
