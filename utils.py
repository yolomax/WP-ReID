import os
import errno
import pickle
import json
import numpy as np
from pathlib import Path


__all__ = ['check_path', 'DataPacker', 'np_filter']


def np_filter(arr, *arg):
    temp_arr = arr
    for i_axis, axis_i in enumerate(arg):
        map_list = []
        for i_elem, elem_i in enumerate(axis_i):
            temp_elem_arr = temp_arr[temp_arr[:, i_axis] == elem_i]
            map_list.append(temp_elem_arr)
        temp_arr = np.concatenate(map_list, axis=0)
    return temp_arr


def check_path(folder_dir, create=False):
    folder_dir = Path(folder_dir)
    if not folder_dir.exists():
        if create:
            try:
                os.makedirs(folder_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        else:
            raise IOError
    return folder_dir


def file_abs_path(arg):
    return Path(os.path.realpath(arg)).parent


class DataPacker(object):
    @staticmethod
    def dump(info, file_path):
        check_path(Path(file_path).parent, create=True)
        with open(file_path, 'wb') as f:
            pickle.dump(info, f)
        print('Store data ---> ' + str(file_path), flush=True)

    @staticmethod
    def load(file_path):
        check_path(file_path)
        with open(file_path, 'rb') as f:
            info = pickle.load(f)
            print('Load data <--- ' + str(file_path), flush=True)
            return info

    @staticmethod
    def json_dump(info, file_path):
        check_path(Path(file_path).parent, create=True)
        with open(file_path, 'w') as f:
            json.dump(info, f)
        print('Store data ---> ' + str(file_path), flush=True)

    @staticmethod
    def json_load(file_path, acq_print=True):
        check_path(file_path)
        with open(file_path, 'r') as f:
            info = json.load(f)
            if acq_print:
                print('Load data <--- ' + str(file_path), flush=True)
            return info
