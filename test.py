import tensorflow as tf
import cv2
import numpy as np

from configuration import Config
from core.centernet import CenterNet, PostProcessing
from data.dataloader import DataLoader


def idx2class():
    return dict((v, k) for k, v in Config.pascal_voc_classes.items())


def draw_boxes_on_image(image, boxes, scores, classes):
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        class_and_score = str(idx2class()[classes[i]]) + ": " + str(scores[i])
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]), color=(255, 0, 0), thickness=2)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1.5, color=(0, 255, 255), thickness=2)
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
