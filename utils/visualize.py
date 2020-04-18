import cv2
from test import test_single_picture
from configuration import Config


def visualize_training_results(pictures, model, epoch):
    """
    :param pictures: List of image directories.
    :param model:
    :param epoch:
    :return:
    """
    index = 0
    for picture in pictures:
        index += 1
        result = test_single_picture(picture_dir=picture, model=model)
        cv2.imwrite(filename=Config.training_results_save_dir + "epoch-{}-picture-{}.jpg".format(epoch, index), img=result)

