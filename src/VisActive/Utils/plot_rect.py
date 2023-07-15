import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Tested
def visualize_vc_rect(input_x, coordinates, num_vc, image_file_names):
    """
    visualize the vc as a rectangles on the original images
    """

    # Create output directory to store the rectangular images to the disk
    sub_model_directory = f'hvc_model_rectangular_VC'
    if not os.path.exists(sub_model_directory):
        os.makedirs(sub_model_directory)

    batch_size = np.shape(input_x)[0]

    for image_idx in range(batch_size):

        image_file_name = image_file_names[image_idx]

        # Plot every vc for image sample
        for vc_idx in range(num_vc):

            plt.imshow(input_x[image_idx])

            # Get (x, y) coordinates
            x1, y1, x2, y2 = coordinates[image_idx, vc_idx, :]
            gap_x = x2 - x1
            gap_y = y2 - y1

            # Get the current reference
            ax = plt.gca()

            # Create a rectangle patch
            # This function reflect the x and y dim
            rect = patches.Rectangle((y1, x1), gap_y, gap_x, linewidth=4, edgecolor='b', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

            image_name = f'{image_file_name}_vc_{vc_idx}.jpg'
            plt.savefig(sub_model_directory + '/' + image_name)
            plt.clf()
