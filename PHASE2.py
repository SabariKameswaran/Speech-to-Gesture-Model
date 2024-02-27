import speech_recognition as sr
import cv2
import mediapipe as mp
import os

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
        return None

def display_images(text_input):
    folder_path = "./Data"
    image_mapping = {
        "hello": os.listdir("./Data/hello"),
        "listening": os.listdir("./Data/listening"),
        "please": os.listdir("./Data/please"),
        "sorry": os.listdir("./Data/sorry"),
        "thanks": os.listdir("./Data/thanks"),
        "welcome": os.listdir("./Data/welcome"),
    }
    if text_input in image_mapping:
        image_files = image_mapping[text_input]
        for i, image_file in enumerate(image_files[:20]):
            image_path = os.path.join(folder_path, text_input, image_file)
            img = cv2.imread(image_path)
            cv2.imshow("Image", img)
            cv2.waitKey(100)  
        cv2.destroyAllWindows()

def perform_gesture(text):
    if text:
        display_images(text)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gesture_detected = False
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture_detected = True
            if gesture_detected:
                print("Gesture Detected!")
            cv2.imshow('MediaPipe Hands', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
while True:
    text = recognize_speech()
    perform_gesture(text)