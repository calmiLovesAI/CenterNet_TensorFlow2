
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
    test_single_image_dir = "./test_pictures/2007_000032.jpg"
    test_images_during_training = False
    training_results_save_dir = "./test_pictures/"
    test_images_dir_list = ["", ""]

    image_size = (384, 384)
    image_channels = 3

    # dataset
    num_classes = 20
    pascal_voc_root = "./data/datasets/VOCdevkit/VOC2012/"
    pascal_voc_images = pascal_voc_root + "JPEGImages"
    pascal_voc_labels = pascal_voc_root + "Annotations"
    # '0' for background
    pascal_voc_classes = {"person": 1, "bird": 2, "cat": 3, "cow": 4, "dog": 5,
                          "horse": 6, "sheep": 7, "aeroplane": 8, "bicycle": 9,
                          "boat": 10, "bus": 11, "car": 12, "motorbike": 13,
                          "train": 14, "bottle": 15, "chair": 16, "diningtable": 17,
                          "pottedplant": 18, "sofa": 19, "tvmonitor": 20}

    # txt file
    txt_file_dir = "data.txt"

    max_boxes_per_image = 50

    # network architecture
    downsampling_ratio = 4
    heads = {"heatmap": num_classes, "wh": 2, "reg": 2}
    head_conv = {"no_conv_layer": 0, "resnets": 64, "dla": 256}
    backbone_name = "resnet_18"

    # loss
    hm_weight = 1.0
    wh_weight = 0.1
    off_weight = 1.0

    score_threshold = 0.3
