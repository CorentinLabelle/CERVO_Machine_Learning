import os.path
from typing import List
import pathlib


def create_filename(tags: tuple, delimiter: str, extension: str = None) -> str:
    filtered_tags = []
    for tag in tags:
        if tag is None or tag == '':
            continue
        filtered_tags.append(tag)

    filename = delimiter.join(filtered_tags)

    if extension is not None:
        if not extension.startswith('.'):
            extension = '.' + extension
        filename += extension

    return filename


def join_path(*args) -> str:
    path = ''
    for arg in args:
        if arg is None:
            continue
        path = os.path.join(path, arg)
    return path


def sort_list_in_alpha_num_order(list_to_sort: list) -> list:
    sorted_indexes = alpha_num_sorted_indexes(list_to_sort)
    return reorder_list_with_indexes(list_to_sort, sorted_indexes)


def alpha_num_sorted_indexes(list_to_sort: list) -> list:
    """
        Sort a list in the way that humans expect.
        Source: https://nedbatchelder.com/blog/200712/human_sorting.html
    """
    import re

    def try_int(s):
        """
        Return an int if possible, or `s` unchanged.
        """
        try:
            return int(s)
        except ValueError:
            return s

    def alphanum_key(s):
        """
        Turn a string into a list of string and number chunks.

        alphanum_key("z23a")
        ["z", 23, "a"]

        """
        return [try_int(c) for c in re.split('([0-9]+)', s)]

    sorted_list = list_to_sort.copy()
    sorted_list.sort(key=alphanum_key)
    sorted_indexes = [list_to_sort.index(element) for element in sorted_list]
    return sorted_indexes


def reorder_list_with_indexes(list_to_reorder: list, indexes: list) -> list:
    return [list_to_reorder[i] for i in indexes]


def list_directory_content(directory: str) -> List[str]:
    content = os.listdir(directory)
    sorted_content = sort_list_in_alpha_num_order(content)
    return sorted_content


def directory_contains_subdirectory(directory: str) -> bool:
    """
    Checks if a directory contains at least one subdirectory.
    :param: str
    :return: bool
    """
    filenames = os.listdir(directory)
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        if os.path.isdir(filepath):
            return True
    return False


def find_with_patterns(directory: str, pattern: str) -> List[str]:
    directory = pathlib.Path(directory)
    content = list(directory.glob(pattern))
    content = [str(c) for c in content]
    sorted_content = sort_list_in_alpha_num_order(content)
    return sorted_content

