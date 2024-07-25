import pickle


def save(variable: any, filename: str) -> None:
    pickle.dump(variable, open(filename, 'wb'))
