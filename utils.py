import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.utils.data as data
import torch


# A wrapper class for a generic datapoint where X is the feature vector
# and y is the target vector
class DataPoints(data.Dataset):
    def __init__(self, X, y=None, extras=None):
        self.data = X
        self.target = y
        self.n_samples = len(X)
        self.extras = extras

    def __len__(self):   # Length of the dataset.
        return self.n_samples

    # Function that returns one point and one label.
    def __getitem__(self, index):
        returns = [torch.Tensor(self.data[index])]
        if self.target is not None:
            returns.append(torch.Tensor(self.target[index]))
        if self.extras is not None:
            returns.append(self.extras[index])
        return tuple(returns)


def show_image(image, bbox):
    """Show image with landmarks"""
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)
    # Create a Rectangle patch
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                             linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()
    plt.pause(0.001)  # pause a bit so that plots are updated
