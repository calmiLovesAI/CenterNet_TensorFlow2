import xml.etree.ElementTree as ET

from pathlib import Path
from configuration import Config


class VOC:
    def __init__(self):
        self.annotations_dir = Config.pascal_voc_labels
        self.images_dir = Config.pascal_voc_images
        self.label_names = VOC.get_filenames(self.annotations_dir, "*.xml")

    @staticmethod
    def get_filenames(root_dir, pattern):
        p = Path(root_dir)
        filenames = [x for x in p.glob(pattern)]
        return filenames

    def __len__(self):
        return len(self.label_names)

    def __getitem__(self, item):
        label_file = self.label_names[item]
        tree = ET.parse(label_file)
        image_name = tree.find("filename").text
        image_width = float(tree.find("size").find("width").text)
        image_height = float(tree.find("size").find("height").text)
        objects = tree.findall("object")
        class_ids = []
        bboxes = []
        for i, obj in enumerate(objects):
            class_id = Config.pascal_voc_classes[obj.find("name").text]
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            class_ids.append(class_id)
            bboxes.append([xmin, ymin, xmax, ymax])
        sample = {
            "image_file_dir": self.images_dir + "/" + image_name,
            "image_height": image_height,
            "image_width": image_width,
            "class_ids": class_ids,
            "bboxes": bboxes
        }
        return sample







