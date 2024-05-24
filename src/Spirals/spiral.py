import skimage
import os
import matplotlib
import numpy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def mirror_right_over_left(arr: numpy.ndarray):
    """
    Mirrors the right half over the left half
    came across this by accident while trying to flip horizontal
    https://stackoverflow.com/questions/22847410/swap-two-values-in-a-numpy-array
    :param arr:
    :return:
    """
    h,w,c = arr.shape
    for i in range(h):
        for j in range(w//2):
            arr[i][j], arr[i][w-j-1] = arr[i][w-j-1], arr[i][j]
    return arr

def mirror_left_over_right(arr: numpy.ndarray):
    """
    Mirrors the left half over the right half
    :param arr:
    :return:
    """
    h,w,c = arr.shape
    for i in range(h):
        for j in range(w//2):
            arr[i][w-j-1], arr[i][j]  = arr[i][j], arr[i][w-j-1]
    return arr


def flip_horizontal(arr: numpy.ndarray):
    """
    Flips the image horizontally
    :param arr:
    :return:
    """
    h,w,c = arr.shape
    for i in range(h):
        for j in range(w//2):
            arr[i][j], arr[i][w-j-1] = arr[i][w-j-1], arr[i][j].copy()  # copy is key here
    return arr

if __name__=="__main__":
    test_image_path = os.path.join(THIS_DIR, 'test_images/test001.png')
    image_arr = skimage.io.imread(test_image_path)

    # flip image horizontal
    new_image_arr = mirror_left_over_right(image_arr)

    skimage.io.imshow(image_arr)
    skimage.io.imshow(new_image_arr)
    matplotlib.pyplot.show()
