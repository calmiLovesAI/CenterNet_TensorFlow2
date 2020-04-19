# CenterNet_TensorFlow2
A tensorflow2.x implementation of CenterNet.

## Requirements:
+ Python >= 3.6
+ TensorFlow == 2.2.0rc3
+ numpy
+ opencv-python

## Results
The following are the detection results of some pictures in the PASCAL VOC 2012 dataset.
![img_1](https://github.com/calmisential/CenterNet_TensorFlow2/blob/master/assets/1.png)<br>
![img_2](https://github.com/calmisential/CenterNet_TensorFlow2/blob/master/assets/2.png)<br>
![img_3](https://github.com/calmisential/CenterNet_TensorFlow2/blob/master/assets/3.png)

## Usage
### Train on PASCAL VOC 2012
1. Download the [PASCAL VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/).
2. Unzip the file and place it in the 'data/datasets' folder, make sure the directory is like this : 
```
|——data
    |——datasets
        |——VOCdevkit
            |——VOC2012
                |——Annotations
                |——ImageSets
                |——JPEGImages
                |——SegmentationClass
                |——SegmentationObject
```
3. Run **write_to_txt.py** to generate **data.txt**.
4. Run **train.py** to start training, before that, you can change the value of the parameters in **configuration.py**.

### Test on single picture
1. Change the *test_single_image_dir* in **configuration.py**.
2. Run **test.py** to test on single picture.

## Acknowledgments
1. Official PyTorch implementation of CenterNet: https://github.com/xingyizhou/CenterNet
2. A TensorFlow implementation of CenterNet: https://github.com/MioChiu/TF_CenterNet


## References
1. [Objects as Points](https://arxiv.org/abs/1904.07850)