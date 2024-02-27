import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import all_actions as aa

class ModelTrain:
    def __init__(self):
        self.PATH_ARRAY = aa.array_path
        self.ROOT = aa.root_of_all
        self.PATH_OF_ALL = aa.load_all_actions_array()
        # print(self.PATH_OF_ALL)
        self.main()

    def data_setting_of(self):
        actions = self.PATH_OF_ALL
        no_sequence = 30
        sequence_length = 30
        self.label_map = {action: i for i, action in enumerate(actions)}
        self.sequences, self.labels = [], []
        for action in actions:
            frames_no = 0
            for sequence in range(no_sequence):
                window = []
                for frame_num in range(sequence_length):
                    res = np.load(os.path.join(
                        self.PATH_ARRAY, action, f"{action}_{frames_no}.npy"))
                    print(os.path.join(
                    self.PATH_ARRAY, action, f"{action}_{frames_no}.npy"))
                    window.append(res)
                    frames_no += 1
                self.sequences.append(window)
                self.labels.append(self.label_map[action])
        self.X = np.array(self.sequences)
        self.y = to_categorical(np.array(self.labels)).astype(int)
        np.save(os.path.join(self.ROOT,"meta", "X.npy"), self.X)
        np.save(os.path.join(self.ROOT,"meta", "y.npy"), self.y)
        print("finish")

    def test_and_train(self):
        self.X = np.load(os.path.join(self.ROOT,"meta", "X.npy"))
        self.y = np.load(os.path.join(self.ROOT,"meta", "y.npy"))
        actions = aa.load_all_actions_array()
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.05)
        log_dir = os.path.join(os.getcwd(), 'Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(actions.shape[0], activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.fit(x_train, y_train, epochs=201, callbacks=[tb_callback])
        model.summary()
        res = model.predict(x_test)
        np.save(r"D:\WORKSPACE\projects\Test Latifa\meta\predict_res.npy",res)
        print(actions[np.argmax(res[1])])
        print(actions[np.argmax(y_test[1])])
        model.save(r'D:\WORKSPACE\projects\Test Latifa\meta\action_6_lab.h5')
        model.load_weights(
            r'D:\WORKSPACE\projects\Test Latifa\meta\action_6_lab.h5')

    def main(self):
        # self.data_setting_of()
        self.test_and_train()


if __name__ == "__main__":
    ml = ModelTrain()
