## This project tackles the problem of Object Localization for the Towel Data Set

### Prerequisites
1.  pandas
2.  numpy
3.  pytorch and torchvision
4.  skimage
5.  tqdm
6.  imgaug (https://github.com/aleju/imgaug)

### Here I explain the steps to run the whole experiment.

1.  Images provided to me by Sewt go in `data/test` folder.
2.  We run `python augmentor.py`, this makes augmented images and puts the new images in `data/train` along with the labels in `data/train/BB_labels.txt`. The augmentations are randomized and the scheme resides in `augmentor.py line 20 - line 110`. You can also specify `-n` or `--copies_per_image` as any integer greater than 0. A `-n 10` means one image in `data/test` will be augmented 10 times and hence the size of final dataset will be 10 times the size of `data/test`. Refer to `python augmentor.py --help` for more configurable settings.
3.  Now, we run following.
    *   `python feature_extractor.py -i data/test -b data/test/BB_labels.txt -o data/test/features.pkl`
    *   `python feature_extractor.py -i data/train -b data/train/BB_labels.txt -o data/train/features.pkl`
    This compresses each image into a vector of 1 x 2048, where this vector is a result of the forward pass from a pretrained resnet152. The result is stored in `data/train/features.pkl` and `data/test/features.pkl` respectively.
4.  After extracting features, we use them to train a configurable feed forward network. This network tries to regress the bounding boxes for images. (Note: The bounding boxes at this point are normalized to be between 0 and 1). We train by executing following `python train.py`. The structure of network and other parameters can be confgiured, please refer to `python train.py --help`. The model is then stored in `trained_ffn.pkl` in the root direcotry.
5.  Finally we use the model produced in step[4] and use it to evaluate predictions for images in data/test, we also compare it with the actual bounding boxes and display final MSE. We do so using `python test.py` (Note: Please make sure to always use the same settings to model structure in step [4] and this step to ensure correct loading of the model).

### Remarks

I realize there are better stratigies to solve the problem of image localization such as yolov3, however in this specific solution I intend to demonstrate my understanding of the problem, clean and structured code with extendible and reusable modules and transfer learning. Therefore I've reduced the solution to mere regression task. This whle project follows pythoncodestyle guidelines.

PS. I am writing this read me while suffering from migrane, please mind the mistakes.