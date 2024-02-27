import numpy as np
import os

Data = "Data"
Array = "Array"
root_of_all = r"D:\WORKSPACE\projects\Test-Latifa"
data_path = os.path.join(root_of_all,Data)
array_path = os.path.join(root_of_all,Array)
get_all_actions = []



def get_all_actions():
    __act = os.listdir(os.path.join(root_of_all, Data))
    all_actions = np.array(__act)
    np.save(r"D:\WORKSPACE\projects\Test-Latifa\meta\actions_all_data.npy", all_actions)
    return all_actions


def load_all_actions():
    get_all_actions()
    return np.load(r"D:\WORKSPACE\projects\Test-Latifa\meta\actions_all_data.npy")


def get_all_actions_array():
    __act = os.listdir(os.path.join(root_of_all, Array))
    all_actions = np.array(__act)
    np.save(r"D:\WORKSPACE\projects\Test-Latifa\meta\actions_all_array.npy", all_actions)
    return all_actions


def load_all_actions_array():
    get_all_actions_array()
    return np.load(r"D:\WORKSPACE\projects\Test-Latifa\meta\actions_all_array.npy")

if __name__ == "__main__":
    load_all_actions()