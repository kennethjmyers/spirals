import skimage
import os
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def mirror_right_over_left(arr: np.ndarray):
    """
    Mirrors the right half over the left half
    came across this by accident while trying to flip horizontal
    https://stackoverflow.com/questions/22847410/swap-two-values-in-a-numpy-array
    :param arr:
    :return:
    """
    arr = arr.copy()  # prevents overwriting the original object passed
    h,w,c = arr.shape
    for i in range(h):
        for j in range(w//2):
            arr[i][j], arr[i][w-j-1] = arr[i][w-j-1], arr[i][j]
    return arr


def mirror_left_over_right(arr: np.ndarray):
    """
    Mirrors the left half over the right half
    :param arr:
    :return:
    """
    arr = arr.copy()  # prevents overwriting the original object passed
    h,w,c = arr.shape
    for i in range(h):
        for j in range(w//2):
            arr[i][w-j-1], arr[i][j]  = arr[i][j], arr[i][w-j-1]
    return arr


def flip_horizontal(arr: np.ndarray):
    """
    Flips the image horizontally
    :param arr:
    :return:
    """
    arr = arr.copy()  # prevents overwriting the original object passed
    h,w,c = arr.shape
    for i in range(h):
        for j in range(w//2):
            arr[i][j], arr[i][w-j-1] = arr[i][w-j-1], arr[i][j].copy()  # copy is key here
    return arr


def average_halves(arr: np.ndarray):
    new_arr = arr.copy()  # prevents overwriting the original object passed
    h, w, c = new_arr.shape
    for i in range(h):
        for j in range(w):
            opp_index = w - j - 1
            value_to_insert = np.mean(np.append(arr[i][j].reshape(1,4),arr[i][opp_index].reshape(1,4), axis=0), axis=0)
            new_arr[i][j] = value_to_insert
    return new_arr


def max_halves(arr: np.ndarray):
    new_arr = arr.copy()  # prevents overwriting the original object passed
    h, w, c = new_arr.shape
    for i in range(h):
        for j in range(w):
            opp_index = w - j - 1
            value_to_insert = np.max(np.append(arr[i][j].reshape(1,4),arr[i][opp_index].reshape(1,4), axis=0), axis=0)
            new_arr[i][j] = value_to_insert
    return new_arr


if __name__=="__main__":
    test_image_prefix = "test001"
    test_image_path = os.path.join(THIS_DIR, f'test_images/{test_image_prefix}.png')
    image_arr = skimage.io.imread(test_image_path)

    for func in [mirror_right_over_left, mirror_left_over_right, flip_horizontal, average_halves, max_halves]:
        new_image_arr = func(image_arr)

        # display image
        title = f"{test_image_prefix}_{func.__name__}"
        fig = make_subplots(rows=1, cols=2)
        fig.update_layout(title=title)
        fig.add_trace(go.Image(z=image_arr), row=1, col=1)
        fig.add_trace(go.Image(z=new_image_arr), row=1, col=2)
        # fig.show()

        # save image under function name
        output_image_path = os.path.join(THIS_DIR, f"output_images/{title}.png")
        fig.write_image(output_image_path)

