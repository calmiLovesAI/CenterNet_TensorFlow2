import tensorflow as tf
import numpy as np

from configuration import Config
from utils.gaussian import gaussian_radius, draw_umich_gaussian


class DetectionDataset:
    def __init__(self):
        self.txt_file = Config.txt_file_dir
        self.batch_size = Config.batch_size

    @staticmethod
    def __get_length_of_dataset(dataset):
        length = 0
        for _ in dataset:
            length += 1
        return length

    def generate_datatset(self):
        dataset = tf.data.TextLineDataset(filenames=self.txt_file)
        length_of_dataset = DetectionDataset.__get_length_of_dataset(dataset)
        train_dataset = dataset.batch(batch_size=self.batch_size)
        return train_dataset, length_of_dataset


class DataLoader:

    input_image_height = Config.get_image_size()[0]
    input_image_width = Config.get_image_size()[1]
    input_image_channels = Config.image_channels

    def __init__(self):
        self.max_boxes_per_image = Config.max_boxes_per_image

    def read_batch_data(self, batch_data):
        batch_size = batch_data.shape[0]
        image_file_list = []
        boxes_list = []
        for n in range(batch_size):
            image_file, boxes = self.__get_image_information(single_line=batch_data[n])
            image_file_list.append(image_file)
            boxes_list.append(boxes)
        boxes = np.stack(boxes_list, axis=0)
        image_tensor_list = []
        for image in image_file_list:
            image_tensor = DataLoader.image_preprocess(is_training=True, image_dir=image)
            image_tensor_list.append(image_tensor)
        images = tf.stack(values=image_tensor_list, axis=0)
        return images, boxes

    def __get_image_information(self, single_line):
        """
        :param single_line: tensor
        :return:
        image_file: string, image file dir
        boxes_array: numpy array, shape = (max_boxes_per_image, 5(xmin, ymin, xmax, ymax, class_id))
        """
        line_string = bytes.decode(single_line.numpy(), encoding="utf-8")
        line_list = line_string.strip().split(" ")
        image_file, image_height, image_width = line_list[:3]
        image_height, image_width = int(float(image_height)), int(float(image_width))
        boxes = []
        num_of_boxes = (len(line_list) - 3) / 5
        if int(num_of_boxes) == num_of_boxes:
            num_of_boxes = int(num_of_boxes)
        else:
            raise ValueError("num_of_boxes must be type 'int'.")
        for index in range(num_of_boxes):
            if index < self.max_boxes_per_image:
                xmin = int(float(line_list[3 + index * 5]))
                ymin = int(float(line_list[3 + index * 5 + 1]))
                xmax = int(float(line_list[3 + index * 5 + 2]))
                ymax = int(float(line_list[3 + index * 5 + 3]))
                class_id = int(line_list[3 + index * 5 + 4])
                xmin, ymin, xmax, ymax = DataLoader.box_preprocess(image_height, image_width, xmin, ymin, xmax, ymax)
                boxes.append([xmin, ymin, xmax, ymax, class_id])
        num_padding_boxes = self.max_boxes_per_image - num_of_boxes
        if num_padding_boxes > 0:
            for i in range(num_padding_boxes):
                boxes.append([0, 0, 0, 0, -1])
        boxes_array = np.array(boxes, dtype=np.float32)
        return image_file, boxes_array

    @classmethod
    def box_preprocess(cls, h, w, xmin, ymin, xmax, ymax):
        resize_ratio = [DataLoader.input_image_height / h, DataLoader.input_image_width / w]
        xmin = int(resize_ratio[1] * xmin)
        xmax = int(resize_ratio[1] * xmax)
        ymin = int(resize_ratio[0] * ymin)
        ymax = int(resize_ratio[0] * ymax)
        return xmin, ymin, xmax, ymax

    @classmethod
    def image_preprocess(cls, is_training, image_dir):
        image_raw = tf.io.read_file(filename=image_dir)
        decoded_image = tf.io.decode_image(contents=image_raw, channels=DataLoader.input_image_channels, dtype=tf.dtypes.float32)
        decoded_image = tf.image.resize(images=decoded_image, size=(DataLoader.input_image_height, DataLoader.input_image_width))
        return decoded_image


class GT:
    def __init__(self, batch_labels):
        self.downsampling_ratio = Config.downsampling_ratio
        self.features_shape = np.array(Config.get_image_size(), dtype=np.int32) // self.downsampling_ratio
        self.batch_labels = batch_labels
        self.batch_size = batch_labels.shape[0]

    def get_gt_values(self):
        gt_heatmap = np.zeros(shape=(self.batch_size, self.features_shape[0], self.features_shape[1], Config.num_classes), dtype=np.float32)
        gt_reg = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image, 2), dtype=np.float32)
        gt_wh = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image, 2), dtype=np.float32)
        gt_reg_mask = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image), dtype=np.float32)
        gt_indices = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image), dtype=np.float32)
        for i, label in enumerate(self.batch_labels):
            label = label[label[:, 4] != -1]
            hm, reg, wh, reg_mask, ind = self.__decode_label(label)
            gt_heatmap[i, :, :, :] = hm
            gt_reg[i, :, :] = reg
            gt_wh[i, :, :] = wh
            gt_reg_mask[i, :] = reg_mask
            gt_indices[i, :] = ind
        return gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices

    def __decode_label(self, label):
        hm = np.zeros(shape=(self.features_shape[0], self.features_shape[1], Config.num_classes), dtype=np.float32)
        reg = np.zeros(shape=(Config.max_boxes_per_image, 2), dtype=np.float32)
        wh = np.zeros(shape=(Config.max_boxes_per_image, 2), dtype=np.float32)
        reg_mask = np.zeros(shape=(Config.max_boxes_per_image), dtype=np.float32)
        ind = np.zeros(shape=(Config.max_boxes_per_image), dtype=np.float32)
        for j, item in enumerate(label):
            item[:4] = item[:4] / self.downsampling_ratio
            xmin, ymin, xmax, ymax, class_id = item
            class_id = class_id.astype(np.int32)
            h, w = int(ymax - ymin), int(xmax - xmin)
            radius = gaussian_radius((h, w))
            radius = max(0, int(radius))
            ctr_x, ctr_y = (xmin + xmax) / 2, (ymin + ymax) / 2
            center_point = np.array([ctr_x, ctr_y], dtype=np.float32)
            center_point_int = center_point.astype(np.int32)
            draw_umich_gaussian(hm[:, :, class_id], center_point_int, radius)
            reg[j] = center_point - center_point_int
            wh[j] = 1. * w, 1. * h
            reg_mask[j] = 1
            ind[j] = center_point_int[1] * self.features_shape[1] + center_point_int[0]
        return hm, reg, wh, reg_mask, ind
