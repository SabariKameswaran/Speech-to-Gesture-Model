import os
import cv2
import mediapipe as mp
from datetime import datetime
from gesture_key_point_collector import DataKeyCollector as dkc


class DataCollector:
    def __init__(self):
        self.name_of_file = input("enter the label : ")
        self.resolution = '480'
        self.STD_DIMENSIONS = {
            "480p": (640, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4k": (3840, 2160),
        }
        self.video_path_save = r"./Test-Latifa"
        self.image_path_root = ""
        self.image_data_path()
        self.cap = cv2.VideoCapture(0)
        self.mp_holistic = mp.solutions.holistic
        self.mp_draw = mp.solutions.drawing_utils
        self.main()

    def image_data_path(self):
        data_path = self.video_path_save
        if os.path.exists(os.path.join(data_path, "Data", self.name_of_file)):
            print("path exist")
        else:
            os.makedirs(os.path.join(data_path, "Data", self.name_of_file))
        self.image_path_root = os.path.join(
            data_path, "Data", self.name_of_file)

    def get_dims(self):
        width, height = self.STD_DIMENSIONS["480p"]
        if self.resolution in self.STD_DIMENSIONS:
            width, height = self.STD_DIMENSIONS[self.resolution]
        self.change_res(width, height)
        return width, height

    def change_res(self, width, height):
        self.cap.set(3, width)
        self.cap.set(4, height)

    def imageCreation(self):
        recording = False
        time_change = []
        time_count = 0
        frame_count = 0
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Camera has something Issue")
                    break
                if cv2.waitKey(1) == ord("r") or recording == True:
                    time_change.append(int(datetime.now().timestamp()))
                    recording = True
                    if frame_count > 99:
                        cv2.imwrite(os.path.join(self.image_path_root, f"{self.name_of_file}_{frame_count}.jpg"), frame)
                        pa = os.path.join(
                            self.image_path_root, f"{self.name_of_file}_{frame_count-99}.jpg")
                        print(f"Image Saved at {pa}")
                    frame_count += 1
                    time_count = time_change[-1] - time_change[0]
                    image, results = self.mediapipe_detection(
                        image=frame, model=holistic)
                    self.draw_style_landmarks(image=image, results=results)
                    cv2.putText(image, f"{self.name_of_file}: {int(time_count)} frame: {frame_count}",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    frame = image.copy()
                cv2.imshow("Recorder",frame)
                if cv2.waitKey(1) & 0xFF == ord('q') or frame_count == 1000:
                    recording = False
                    break
        self.cap.release()
        cv2.destroyAllWindows()
        del time_change


    def draw_style_landmarks(self, image, results):
        self.mp_draw.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, self.mp_draw.DrawingSpec(
            color=(105, 237, 17), thickness=1, circle_radius=1), self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1))
        self.mp_draw.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, self.mp_draw.DrawingSpec(
            color=(105, 237, 17), thickness=1, circle_radius=1), self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2))
        self.mp_draw.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.mp_draw.DrawingSpec(
            color=(105, 237, 17), thickness=1, circle_radius=1), self.mp_draw.DrawingSpec(color=(26, 209, 209), thickness=1, circle_radius=2))
        self.mp_draw.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, self.mp_draw.DrawingSpec(
            color=(105, 237, 17), thickness=1, circle_radius=1), self.mp_draw.DrawingSpec(color=(26, 209, 209), thickness=1, circle_radius=2))

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, result

    def main(self):
        self.image_data_path()
        self.get_dims()
        self.imageCreation()


if __name__ == "__main__":
    gdc = DataCollector()