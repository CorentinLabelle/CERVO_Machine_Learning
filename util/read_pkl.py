import pickle


def read(pkl_file: str):
    return pickle.load(open(pkl_file, 'rb'))
