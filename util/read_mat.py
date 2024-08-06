import os
import mat73
import h5py
import numpy
from pathlib import Path
import scipy.io as spio


def read_mat(mat_file_path: str, mat_file_key: str = None, with_key: bool = True):
    __check_extension__(mat_file_path)

    try:
        return with_scipy(mat_file_path, mat_file_key, with_key)
    except NotImplementedError:
        return with_mat73(mat_file_path, mat_file_key)


def with_h5py(mat_file_path: str):
    __check_extension__(mat_file_path)
    return h5py.File(mat_file_path, 'r')


def with_mat73(mat_file_path: str, mat_file_key: str = None):
    __check_extension__(mat_file_path)
    file = mat73.loadmat(mat_file_path)
    if mat_file_key is None:
        return file
    else:
        return file[mat_file_key]


def with_scipy(mat_file_path: str, mat_file_key: str = None, with_key: bool = True):
    __check_extension__(mat_file_path)
    if mat_file_key is None:
        mat_file_key = __get_filename__(mat_file_path)
    # file = spio.loadmat(mat_file_path)
    file = loadmat(mat_file_path)
    if with_key:
        return numpy.stack(file[mat_file_key])
    else:
        return file


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def __check_extension__(mat_file_path: str):
    assert mat_file_path.endswith(".mat"), "file should end with '.mat': " + mat_file_path
    assert os.path.isfile(mat_file_path), "file does not exist: " + mat_file_path


def __get_filename__(mat_file_path: str) -> str:
    return Path(mat_file_path).stem
