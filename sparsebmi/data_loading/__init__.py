import os
from sys import platform

from .load_finger import load_neural_finger_dataset
from .load_finger import load_emg_finger_dataset
from .load_nlb import load_nlb
from .load_simulated import load_data_simulated
from .load_mouse_face import load_mouse_face


def get_server_data_path(is_monkey=True):
    """ Returns the path to the server data folder """

    # choose the standard path based on the OS
    if platform == "linux" or platform == "linux2":
        path = '/run/user/1000/gvfs/smb-share:server=cnpl-drmanhattan.engin.umich.edu,share=share/Data'
    elif platform == "darwin":
        path = 'smb://cnpl-drmanhattan.engin.umich.edu/share/Data'
    elif platform == "win32":
        path = 'Z:/Data'

    if is_monkey:
        path = os.path.join(path, 'Monkeys')
    else:
        path = os.path.join(path, 'Humans')

    return path
