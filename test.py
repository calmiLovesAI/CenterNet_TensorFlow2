import tensorflow as tf
import cv2
import numpy as np

from configuration import Config
from core.centernet import CenterNet, PostProcessing
from data.dataloader import DataLoader


def idx2class():
    return dict((v, k) for k, v in Config.pascal_voc_classes.items())


def draw_boxes_on_image(image, boxes, scores, classes):
    idx2class_dict = idx2class()
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        class_and_score = "{}: {:.3f}".format(str(idx2class_dict[classes[i]]), scores[i])
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]), color=(250, 206, 135), thickness=2)

        text_size = cv2.getTextSize(text=class_and_score, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        text_width, text_height = text_size[0][0], text_size[0][1]
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 0] + text_width, boxes[i, 1] - text_height), color=(203, 192, 255), thickness=-1)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
    return image


def test_single_picture(picture_dir, model):
    image_array = cv2.imread(picture_dir)
    image = DataLoader.image_preprocess(is_training=False, image_dir=picture_dir)
    image = tf.expand_dims(input=image, axis=0)

    outputs = model(image, training=False)
    post_process = PostProcessing()
    boxes, scores, classes = post_process.testing_procedure(outputs, [image_array.shape[0], image_array.shape[1]])
    image_with_boxes = draw_boxes_on_image(image_array, boxes.astype(np.int), scores, classes)
    return image_with_boxes


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    centernet = CenterNet()
    centernet.load_weights(filepath=Config.save_model_dir + "saved_model")

    image = test_single_picture(picture_dir=Config.test_single_image_dir, model=centernet)

    cv2.namedWindow("detect result", flags=cv2.WINDOW_NORMAL)
    cv2.imshow("detect result", image)
    cv2.waitKey(0)
