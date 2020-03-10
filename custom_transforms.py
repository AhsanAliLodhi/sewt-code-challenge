import torch
import numpy as np


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']
        h, w = image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h))
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        w_r = new_w / w
        h_r = new_h / h
        bbox = [bbox[0] * w_r,
                bbox[1] * h_r,
                bbox[2] * w_r,
                bbox[3] * h_r]
        bbox = [bbox[0] / new_w,
                bbox[1] / new_h,
                bbox[2] / new_w,
                bbox[3] / new_h]
        sample['image'] = img
        sample['bbox'] = bbox
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        bbox = np.array(bbox).astype(float)
        sample['image'] = torch.from_numpy(image).float()
        sample['bbox'] = torch.from_numpy(bbox).float()
        return sample
