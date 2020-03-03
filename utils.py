import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_image(image, bbox):
    """Show image with landmarks"""
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(image)
    # Create a Rectangle patch
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()
    plt.pause(0.001)  # pause a bit so that plots are updated