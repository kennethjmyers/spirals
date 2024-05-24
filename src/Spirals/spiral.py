import skimage
import os
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Callable

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
            arr[i][w-j-1], arr[i][j] = arr[i][j], arr[i][w-j-1]
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


def stack_pixels(arr1: np.ndarray, arr2: np.ndarray):
    """
    Given two pixels, convert them from two (n,) arrays to (2,n) arrays
    :param arr1:
    :param arr2:
    :return:
    """
    assert arr1.shape == arr2.shape
    n = arr1.shape[0]
    return np.append(arr1, arr2).reshape(2, *arr1.shape)


def func_apply_to_halves(arr: np.ndarray, func: Callable, preprocessing_func=stack_pixels, postprocessing_func= lambda x:x):
    h, w, c = arr.shape

    left_half_idx_end = (w//2)+1
    right_half_idx_start = -left_half_idx_end

    left_half = arr.copy()[:, :left_half_idx_end, :]
    right_half = arr.copy()[:, right_half_idx_start:, :]
    proc_arr = postprocessing_func(func(preprocessing_func(left_half, right_half)))
    new_arr = np.append(proc_arr, proc_arr[:, ::-1, :], axis=1)

    return new_arr


def average_halves(arr: np.ndarray):
    """
    Applies np.mean to pixels on opposite sides of an image
    :param arr:
    :return:
    """
    return func_apply_to_halves(arr, func=lambda x: np.mean(x, axis=0))


def average_halves_glitched(arr: np.ndarray):
    """
    This produces a 'glitched' version of average_halves.
    The operation here replicates what happens if you try to add the pixels together then floor divide by 2.
    The datatype of the arrays are uint8 so the channels are bound by [0,255]. When the arrays are added a `%256` is applied to each channel value resulting in unexpected results.
    :param arr:
    :return:
    """
    return func_apply_to_halves(arr, func=lambda x: (np.sum(x, axis=0)%256)//2)


def sum_halves(arr: np.ndarray):
    """
    This produces a sum of both halves.
    The datatype of the arrays are uint8 so the channels are bound by [0,255]. When the arrays are added a `%256` is applied to each channel value resulting in unexpected results.
    :param arr:
    :return:
    """
    return func_apply_to_halves(arr, func=lambda x: np.sum(x, axis=0))


def min_of_all_channels_halves(arr: np.ndarray):
    """
    Applies np.min to pixels on opposite sides of an image. This min is taken over all values of both pixels and not per-channel.
    :param arr:
    :return:
    """
    return func_apply_to_halves(arr, func=np.min)


def max_of_all_channels_halves(arr: np.ndarray):
    """
    Applies np.max to pixels on opposite sides of an image. This max is taken over all values of both pixels and not per-channel.
    The alpha-channel is ignored since that value would be held constant at 255 producing a white image
    :param arr:
    :return:
    """
    return func_apply_to_halves(arr, func=lambda x: np.max(x, axis=2), preprocessing_func=lambda x,y: np.append(x, y, axis=2), postprocessing_func= lambda x: x.reshape(*x.shape,1))


def min_halves(arr: np.ndarray):
    """
    Applies np.min to pixels on opposite sides of an image. Min value is taken per-channel.
    :param arr:
    :return:
    """
    return func_apply_to_halves(arr, func=lambda x:np.min(x, axis=0))


def max_halves(arr: np.ndarray):
    """
    Applies np.max to pixels on opposite sides of an image. Max value is taken per-channel.
    :param arr:
    :return:
    """
    return func_apply_to_halves(arr, func=lambda x: np.max(x, axis=0))


if __name__=="__main__":
    test_image_prefix = "test001"
    test_image_path = os.path.join(THIS_DIR, f'test_images/{test_image_prefix}.png')
    image_arr = skimage.io.imread(test_image_path)

    funcs = [mirror_right_over_left, mirror_left_over_right, flip_horizontal, average_halves, average_halves_glitched,
     sum_halves, min_of_all_channels_halves, min_halves, max_of_all_channels_halves, max_halves]
    funcs = [max_of_all_channels_halves]

    for func in funcs:
        print(f"applying function {func.__name__}")
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

