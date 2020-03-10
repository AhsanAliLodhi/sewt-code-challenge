import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
import random
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import shutil
import argparse


# A helper function which makes any augmention run only half the times
def sometimes(aug):
    return iaa.Sometimes(0.5, aug)


# A scheme augmentations which are carried out at random
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.SomeOf((0, 2), [
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
        ]),
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            # scale images to 80-120% of their size, individually per axis
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate by -20 to +20 percent (per axis)
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            # use nearest neighbour or bilinear interpolation (fast)
            order=[0, 1],
            # if mode is constant, use a cval between 0 and 255
            cval=(0, 255),
            # use any of scikit-image's warping modes (see 2nd image from the
            # top for examples)
            mode=ia.ALL
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                iaa.OneOf([
                    # blur images with a sigma between 0 and 3.0
                    iaa.GaussianBlur((0, 3.0)),
                    # blur image using local means with kernel sizes between 2
                    # and 7
                    iaa.AverageBlur(k=(2, 7)),
                    # blur image using local medians with kernel sizes between
                    # 2 and 7
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                # sharpen images
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                # emboss images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0,
                                           1.0)),
                ])),
                # add gaussian noise to images
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255),
                                          per_channel=0.5),
                iaa.OneOf([
                    # randomly remove up to 10% of the pixels
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05),
                                      per_channel=0.2),
                ]),
                # invert color channels
                iaa.Invert(0.05, per_channel=True),
                # change brightness of images (by -10 to 10 of original value)
                iaa.Add((-10, 10), per_channel=0.5),
                # change hue and saturation
                iaa.AddToHueAndSaturation((-20, 20)),
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                # improve or worsen the contrast
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                # move pixels locally around (with random strengths)
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5),
                          sigma=0.25)),
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)


# Iterate over images from a certain location
def image_iterator(img_path, bbox_csv_path, extention="jpg"):
    idx = 0
    labels = pd.read_csv(bbox_csv_path)
    for file in os.listdir(img_path):
        if file.endswith(extention):
            image = imageio.imread(os.path.join(img_path, file))
            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=labels.iloc[idx, 1],
                            y1=labels.iloc[idx, 2],
                            x2=labels.iloc[idx, 3],
                            y2=labels.iloc[idx, 4])
            ], shape=image.shape)
            idx += 1
            yield file, image, bbs


# Go through all the images and their bounding boxes, for each image
# run random augmentations copies_per_image times, store the augmented images
# in out_img_path folder and bboxes in out_bbox_path. You can also save some
# samples with bounded box drawn on them to see how they look.
def run(img_path, bbox_path, copies_per_image=100, out_img_path="augmented",
        out_bbox_path="augmented_bbox.csv", img_extention="jpg",
        sample_path=None):
    if os.path.exists(out_img_path):
        shutil.rmtree(out_img_path)
    print(out_img_path)
    os.makedirs(out_img_path)
    if sample_path is not None and os.path.exists(sample_path):
        shutil.rmtree(sample_path)
        os.makedirs(sample_path)
    if os.path.exists(out_bbox_path):
        os.remove(out_bbox_path)
    total_files = len([name for name in os.listdir(img_path) if
                      name.endswith(img_extention)])
    with open(out_bbox_path, 'a') as out_bbox_file:
        out_bbox_file.write(",0,1,2,3\n")
        for filename, image, bbox in tqdm(image_iterator(img_path, bbox_path,
                                          img_extention), total=total_files):
            for idx in range(copies_per_image):
                image_aug, bbs_aug = seq(image=image, bounding_boxes=bbox)
                img = Image.fromarray(image_aug)
                img_name = (filename.split(".")[0] +
                            "_"+str(idx)+"."+filename.split(".")[-1])
                img.save(os.path.join(out_img_path, img_name))
                x1 = bbs_aug.bounding_boxes[0].x1
                x2 = bbs_aug.bounding_boxes[0].x2
                y1 = bbs_aug.bounding_boxes[0].y1
                y2 = bbs_aug.bounding_boxes[0].y2
                csv_line = ",".join([img_name, str(x1), str(y1),
                                    str(x2), str(y2)])+"\n"
                out_bbox_file.write(csv_line)
                if sample_path is not None and random.random() < 0.1:
                    sample = Image.fromarray(bbs_aug.draw_on_image(image_aug))
                    sample.save(os.path.join(sample_path, img_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    Generate augmented images. Go through all the images and their bounding,
    boxes for each image run random augmentations 'copies_per_image' times,
    then store the augmented images in 'out_img_path' folder and bboxes in.
    'out_bbox_path' You can also save some samples with bounded box drawn on
    them to see how they look with 'sample_path'.''')
    parser.add_argument('-i', '--img_path', type=str,
                        help='directory where the images reside.',
                        default="data/test/")
    parser.add_argument('-b', '--bbox_path', type=str,
                        help='path for bbox csv.',
                        default="data/test/BB_labels.txt")
    parser.add_argument('-n', '--copies_per_image', type=int,
                        help='number of times to make copies of an image.',
                        default=30)
    parser.add_argument('-o', '--out_img_path', type=str,
                        help='path to store augmented images.',
                        default="data/train/")
    parser.add_argument('-p', '--out_bbox_path', type=str,
                        help='file to store the bbox csv for augmented images',
                        default="data/train/BB_labels.txt")
    parser.add_argument('-s', '--sample_path', type=str,
                        help='path to store the samples at',
                        default="samples")
    args = parser.parse_args()
    run(args.img_path, args.bbox_path,
        copies_per_image=args.copies_per_image,
        out_img_path=args.out_img_path,
        out_bbox_path=args.out_bbox_path,
        sample_path=args.sample_path)
