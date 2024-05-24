import skimage
import os
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Callable


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def stack_images(arr1: np.ndarray, arr2: np.ndarray, flip_second=True):
    """
    Given two images, convert them from two (r,g,b,c) arrays to (2,r,g,b,c) arrays
    :param arr1:
    :param arr2:
    :param flip_second:
    :return:
    """
    assert arr1.shape == arr2.shape
    n = arr1.shape[0]
    if flip_second:
        arr2 = np.flip(arr2.copy(), axis=1)
    return np.append(arr1, arr2).reshape(2, *arr1.shape)


def ignore_alpha(arr: np.ndarray):
    return arr[:, :, :3]


def func_apply_to_halves(
        arr: np.ndarray,
        preprocessing_func: Callable = stack_images,
        func: Callable = lambda x:x,
        postprocessing_func: Callable = lambda x:x
):
    h, w, c = arr.shape

    left_half_idx_end = (w//2)+1
    right_half_idx_start = -left_half_idx_end

    left_half = arr.copy()[:, :left_half_idx_end, :]
    right_half = arr.copy()[:, right_half_idx_start:, :]

    # apply functions
    preproc_arr = preprocessing_func(left_half, right_half)
    func_applied_arr = func(preproc_arr)
    postproc_arr = postprocessing_func(func_applied_arr)

    # take the half and the flipped half and append together
    new_arr = np.append(postproc_arr, np.flip(postproc_arr[:, :, :], axis=1), axis=1)

    return new_arr


def mirror_right_over_left(arr: np.ndarray):
    """
    Mirrors the right half over the left half
    came across this by accident while trying to flip horizontal
    https://stackoverflow.com/questions/22847410/swap-two-values-in-a-numpy-array
    :param arr:
    :return:
    """
    return func_apply_to_halves(
        arr=arr,
        preprocessing_func=lambda x,y: y,  # essentially given the left half and the right half, keep the right half
        func=lambda y: np.flip(y, axis=1)  # take the right side and flip it over to left, this becomes our new left side
    )
    # this flipped right side will be combined with the reversed version of itself (original right) on the right side


def mirror_left_over_right(arr: np.ndarray):
    """
    Mirrors the left half over the right half
    :param arr:
    :return:
    """
    return func_apply_to_halves(
        arr=arr,
        preprocessing_func=lambda x, y: x  # essentially given the left half and the right half, keep the right half
    )


def flip_horizontal(arr: np.ndarray):
    """
    Flips the image horizontally
    :param arr:
    :return:
    """
    arr = arr.copy()  # prevents overwriting the original object passed
    return np.flip(arr, axis=1)


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
    return func_apply_to_halves(
        arr,
        preprocessing_func=lambda x, y: np.append(ignore_alpha(x), np.flip(ignore_alpha(y), axis=1), axis=2),
        func=lambda x: np.min(x, axis=2),
        postprocessing_func=lambda x: np.repeat(x.reshape(*x.shape, 1), 3, axis=2)
    )


def max_of_all_channels_halves(arr: np.ndarray):
    """
    Applies np.max to pixels on opposite sides of an image. This max is taken over all values of both pixels and not per-channel.
    The alpha-channel is ignored since that value would be held constant at 255 producing a white image
    :param arr:
    :return:
    """
    return func_apply_to_halves(
        arr,
        preprocessing_func=lambda x,y: np.append(ignore_alpha(x), np.flip(ignore_alpha(y), axis=1), axis=2),
        func=lambda x: np.max(x, axis=2),
        postprocessing_func= lambda x: np.repeat(x.reshape(*x.shape,1), 3, axis=2)
    )


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
    # funcs = [min_of_all_channels_halves]

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

