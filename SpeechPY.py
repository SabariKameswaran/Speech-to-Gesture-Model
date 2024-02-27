import cv2
import os
import speech_recognition as sr


def display_images_for_keys(keys, image_mapping):
    for key in keys:
        # Ensure key is lowercase for case-insensitive matching
        key = key.lower()

        if key not in image_mapping:
            print(f"Images for key '{key}' not found.")
            continue

        image_files = image_mapping[key]

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)

            if not os.path.isfile(image_path):
                print(f"Image file not found: {image_path}")
                continue

            img = cv2.imread(image_path)

            if img is None:
                print(f"Error loading image: {image_path}")
                continue

            cv2.imshow("Sign_Gesture", img)
            cv2.waitKey(20)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    folder_path = "D:\\WORKSPACE\\projects\\Test-Latifa\\Data"

    image_mapping = {
        "hello" : [os.listdir("D:\\WORKSPACE\\projects\\Test-Latifa\\Data\\hello")]
    }
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak:")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("You said:", text)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Request error; {0}".format(e))

    keys_input = input("Enter key names (comma-separated): ")
    keys = [key.strip() for key in keys_input.split(',')]

    display_images_for_keys(keys, image_mapping)
# os.listdir("D:\\WORKSPACE\\projects\\Test-Latifa\\Data\\hello")
# import os 
# print(os.listdir("D:\\WORKSPACE\\projects\\Test-Latifa\\Data\\hello"))
