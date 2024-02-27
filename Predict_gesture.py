import os
import numpy as np
import cv2
import speech_recognition as sr
import all_actions as aa
# from vosk import Model, KaldiRecognizer
import pyaudio
import time
from better_profanity import profanity
# from gensim.models import KeyedVectors


class Gesture_predictor:
    def __init__(self):
        self.folder_path = r"./Data"
        self.actions = aa.load_all_actions()
        print(self.actions)
        self.image_mapping = {}
        for i in self.actions:
            self.image_mapping[i] = [os.path.join(
                self.folder_path, i, f"{i}_{j}.jpg") for j in range(50)]
        # self.model = Model(
        #     r"D:\WORKSPACE\projects\Test-Latifa\vosk-model-small-en-us-0.15")
        # self.rec = KaldiRecognizer(self.model, 16000)
        # self.mic = pyaudio.PyAudio()
        # self.stream = self.mic.open(
        #     format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
        # self.stream.start_stream()
        # self.all_keys = []
        # self.model_path = r"D:\WORKSPACE\projects\Test-Latifa\GoogleNews-vectors-negative300.bin"
        # self.word2vec_model = KeyedVectors.load_word2vec_format(
        #     self.model_path, binary=True)
        self.Speech_Capture()

    def clean_text(self, text):
        cleaned_text = profanity.censor(text)
        return cleaned_text

    def vector_check(self, key):
        word = key[0].split()
        for w in word:
            similar_words = self.word2vec_model.most_similar(w, topn=5)
            for sw in similar_words:
                for k in self.all_keys:
                    if sw == k:
                        return k
                    else:
                        return " "

    def Speech_Capture(self):
        # while True:
        #     text_dict = {}
        #     data = self.stream.read(4096)
        #     if self.rec.AcceptWaveform(data=data):
        #         text_dict = eval(self.rec.Result())
        #         text = text_dict.get('text', '')
        #         if text.strip():
        #             cleaned_text = self.clean_text(text)
        #             keys_input = cleaned_text.strip()
        #             keys = [key.strip() for key in keys_input.split(',')]
        #             self.all_keys.append(keys)
        #             print(keys)
        #     #         self.vector_check(keys)
        #             if "stop recording" in keys:
        #                 print(self.all_keys)
        #                 # exit()
        #                 break
        #             self.display_images_for_keys(keys=keys)
        #         self.stream.read(self.stream.get_read_available(),exception_on_overflow=False)
        #     time.sleep(0.1)
        text = input()

    def display_images_for_keys(self, keys):
        for key in keys:
            key = key.lower()
            if key not in self.image_mapping:
                print(f"Images for key '{key}' not found.")
                continue
            image_files = self.image_mapping[key]
            for image_file in image_files:
                image_path = os.path.join(self.folder_path, image_file)
                if not os.path.isfile(image_path):
                    image_path = r"./Data/404/vadivelu.gif"
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


gr = Gesture_predictor()






















































































































# def Speech_Capture(self):
#      while True:
#         text_dict = {}
#         text = ""
#         data = self.stream.read(4096)
#         if self.rec.AcceptWaveform(data=data):
#             text_dict = eval(self.rec.Result())
#             text = text_dict.get('text', '')
#             if text.strip():
#                 keys_input = text.strip()
#                 keys = [key.strip() for key in keys_input.split(',')]
#                 self.all_keys.append(keys)
#                 print(keys)
#                 if "stop" in keys:
#                     print(self.all_keys)
#                     break
#                 self.display_images_for_keys(keys=keys)
#         self.stream.read(self.stream.get_read_available(), exception_on_overflow=False)
# time.sleep(0.1)  # Adjust the sleep time as needed
# r = sr.Recognizer()
# with sr.Microphone() as source:
#     print("Speak:")
#     audio = r.listen(source)
# try:
#     text = r.recognize_google(audio)
#     print("You said:", text)
# except sr.UnknownValueError:
#     print("Could not understand audio")
#     text = "404"
# except sr.RequestError as e:
#     print("Request error; {0}".format(e))
# keys =[]
# for i in range(2):
#     text_dict= {}
#     text = ""
#     data = self.stream.read(4096)
#     if self.rec.AcceptWaveform(data=data):
#         text_dict = eval(self.rec.Result())
#         text = text_dict['text']
#         if text != "" or text != " ":
#             keys_input = text
#             keys = [key.strip() for key in keys_input.split(',')]
#             self.all_keys.append(keys)
#             print(keys)
#             for i in keys:
#                 if i == "stop recording":
#                     print(self.all_keys)
#                     exit()
# self.display_images_for_keys(keys=keys)
# time.sleep(0.5)
# keys_input =  text
# keys = [key.strip() for key in keys_input.split(',')]
# for i in keys:
#     if i == "stop recording":
#         exit()
# self.display_images_for_keys(keys=keys)
# text_dict = {}
# text = input("say : ")
# data = self.stream.read(4096)
# if self.rec.AcceptWaveform(data=data):
#     text_dict = eval(self.rec.Result())
#     text = text_dict.get('text', '')
#     if text.strip():
#         cleaned_text = self.clean_text(text)
#         keys_input = cleaned_text.strip()
#         keys = [key.strip() for key in keys_input.split(',')]
#         self.all_keys.append(keys)
#         print(keys)
#         self.vector_check(keys)
#         if "stop recording" in keys:
#             print(self.all_keys)
#             break
# self.display_images_for_keys(keys=keys)
# self.stream.read(self.stream.get_read_available(),
#                  exception_on_overflow=False)
# time.sleep(0.1)
