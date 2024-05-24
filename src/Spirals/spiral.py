import skimage
import os
import matplotlib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__=="__main__":
    test_image_path = os.path.join(THIS_DIR, 'test_images/test001.png')
    image_arr = skimage.io.imread(test_image_path)
    skimage.io.imshow(image_arr)
    matplotlib.pyplot.show()