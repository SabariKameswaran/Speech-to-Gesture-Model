import os
import numpy as np
import cv2
import speech_recognition as sr
import all_actions as aa
from vosk import Model, KaldiRecognizer
import pyaudio
import time
from better_profanity import profanity
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


class Gesture_predictor:
    def __init__(self):
        self.folder_path = r"D:\WORKSPACE\projects\Test Latifa\Data"
        self.actions = aa.load_all_actions()  # list of actions
        print(self.actions)
        self.image_mapping = {}
        for i in self.actions:
            self.image_mapping[i] = [os.path.join(
                self.folder_path, i, f"{i}_{j}.jpg") for j in range(50)]
        self.model = Model(
            r"D:\WORKSPACE\projects\Test Latifa\vosk-model-small-en-us-0.15")
        self.rec = KaldiRecognizer(self.model, 16000)
        self.mic = pyaudio.PyAudio()
        self.stream = self.mic.open(
            format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
        self.stream.start_stream()
        self.all_keys = []
        self.class_mapping = {
            "404": "404",
            "hello": "hello",
            "listening": "listening",
            "please": "please",
            "sorry": "sorry",
            "thanks": "thanks",
            "thank you": "thanks",
            "thank": "thanks",
            "you are welcome": "you are welcome"
        }
        self.model_path = r'D:\WORKSPACE\projects\Test Latifa\GoogleNews-vectors-negative300.bin'
        self.stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
        self.word2vec_model = KeyedVectors.load_word2vec_format(
            self.model_path, binary=True)


    def get_class_label(self, text):
        max_similarity = 0
        class_label = None
        for synonym, label in self.class_mapping.items():
            similarity = self.calculate_similarity(text, synonym)
            if similarity > max_similarity:
                max_similarity = similarity
                class_label = label
        return class_label

    def calculate_similarity(self, word1, word2):
        if word1 in self.word2vec_model.vocab and word2 in self.word2vec_model.vocab:
            vec1 = self.word2vec_model[word1].reshape(1, -1)
            vec2 = self.word2vec_model[word2].reshape(1, -1)
            return cosine_similarity(vec1, vec2)[0][0]
        else:
            return 0  # Return 0 if any of the words is not in the vocabulary

    def Speech_Capture(self):
        self.stream.start_stream()
        while True:
            text_dict = {}
            text = ""
            data = self.stream.read(4096)
            if self.rec.AcceptWaveform(data=data):
                text_dict = eval(self.rec.Result())
                text = text_dict.get('text', '').strip()
                if text:
                    class_label = self.get_class_label(text)
                    if class_label:
                        self.display_images_for_class(class_label)
            self.stream.read(self.stream.get_read_available(),
                             exception_on_overflow=False)
            time.sleep(0.1)

    def display_images_for_class(self, keys):
        for key in keys:
            # Ensure key is lowercase for case-insensitive matching
            key = key.lower()
            if key not in self.image_mapping:
                print(f"Images for key '{key}' not found.")
                continue
            image_files = self.image_mapping[key]
            for image_file in image_files:
                image_path = os.path.join(self.folder_path, image_file)
                if not os.path.isfile(image_path):
                    image_path = r"D:\WORKSPACE\projects\Test Latifa\Data\404\vadivelu.gif"
                    continue
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error loading image: {image_path}")
                    continue
                cv2.putText(
                    img, f"{key}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Sign_Gesture", img)
                cv2.waitKey(25)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
        self.Speech_Capture()


# Create an instance of Gesture_predictor and run Speech_Capture
gr = Gesture_predictor()
gr.Speech_Capture()
