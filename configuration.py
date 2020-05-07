
class Config:
    epochs = 50
    batch_size = 8
    learning_rate_decay_epochs = 10

    # save model
    save_frequency = 5
    save_model_dir = "saved_model/"
    load_weights_before_training = False
    load_weights_from_epoch = 0

    # test image
    test_single_image_dir = ""
    test_images_during_training = False
    training_results_save_dir = "./test_pictures/"
    test_images_dir_list = ["", ""]

    image_size = {"resnet_18": (384, 384), "resnet_34": (384, 384), "resnet_50": (384, 384),
                  "resnet_101": (384, 384), "resnet_152": (384, 384),
                  "D0": (512, 512), "D1": (640, 640), "D2": (768, 768),
                  "D3": (896, 896), "D4": (1024, 1024), "D5": (1280, 1280),
                  "D6": (1408, 1408), "D7": (1536, 1536)}
    image_channels = 3

    # dataset
    num_classes = 20
    pascal_voc_root = "./data/datasets/VOCdevkit/VOC2012/"
    pascal_voc_images = pascal_voc_root + "JPEGImages"
    pascal_voc_labels = pascal_voc_root + "Annotations"

    pascal_voc_classes = {"person": 0, "bird": 1, "cat": 2, "cow": 3, "dog": 4,
                          "horse": 5, "sheep": 6, "aeroplane": 7, "bicycle": 8,
                          "boat": 9, "bus": 10, "car": 11, "motorbike": 12,
                          "train": 13, "bottle": 14, "chair": 15, "diningtable": 16,
                          "pottedplant": 17, "sofa": 18, "tvmonitor": 19}

    # txt file
    txt_file_dir = "data.txt"

    max_boxes_per_image = 50

    # network architecture
    downsampling_ratio = 8  # efficientdet: 8, others: 4

    # efficientdet
    width_coefficient = {"D0": 1.0, "D1": 1.0, "D2": 1.1, "D3": 1.2, "D4": 1.4, "D5": 1.6, "D6": 1.8, "D7": 1.8}
    depth_coefficient = {"D0": 1.0, "D1": 1.1, "D2": 1.2, "D3": 1.4, "D4": 1.8, "D5": 2.2, "D6": 2.6, "D7": 2.6}
    dropout_rate = {"D0": 0.2, "D1": 0.2, "D2": 0.3, "D3": 0.3, "D4": 0.4, "D5": 0.4, "D6": 0.5, "D7": 0.5}
    # bifpn channels
    w_bifpn = {"D0": 64, "D1": 88, "D2": 112, "D3": 160, "D4": 224, "D5": 288, "D6": 384, "D7": 384}
    # bifpn layers
    d_bifpn = {"D0": 2, "D1": 3, "D2": 4, "D3": 5, "D4": 6, "D5": 7, "D6": 8, "D7": 8}

    heads = {"heatmap": num_classes, "wh": 2, "reg": 2}
    head_conv = {"no_conv_layer": 0, "resnets": 64, "dla": 256, "efficientdet": 64}
    backbone_name = "D0"
    # can be selected from: resnet_18, resnet_34, resnet_50, resnet_101, resnet_152, D0~D7

    # loss
    hm_weight = 1.0
    wh_weight = 0.1
    off_weight = 1.0

    score_threshold = 0.3

    @classmethod
    def get_image_size(cls):
        return cls.image_size[cls.backbone_name]

    @classmethod
    def get_width_coefficient(cls, backbone_name):
        return cls.width_coefficient[backbone_name]

    @classmethod
    def get_depth_coefficient(cls, backbone_name):
        return cls.depth_coefficient[backbone_name]

    @classmethod
    def get_dropout_rate(cls, backbone_name):
        return cls.dropout_rate[backbone_name]

    @classmethod
    def get_w_bifpn(cls, backbone_name):
        return cls.w_bifpn[backbone_name]

    @classmethod
    def get_d_bifpn(cls, backbone_name):
        return cls.d_bifpn[backbone_name]
