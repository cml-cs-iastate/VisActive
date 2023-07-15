import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_image(path, size):
    ret = PIL.Image.open(path)
    ret = ret.resize((size, size))
    ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
    ret = ret * 1./255
    return ret


def vc_interpretation(input_x, image_file_name, vc_idx, save_directory, hvc_model, counter):
    """
    """

    # ------------------------------------------------------------------------------------------------------------------
    # Generating VC

    # Run the hvc_model to get the output of the VCL
    vcl_output = hvc_model.predict(input_x)
    coordinates = vcl_output[4]

    # Plot only the most similar vc for image sample
    plt.imshow(input_x[0])

    # Get (x, y) coordinates
    x1, y1, x2, y2 = coordinates[0, vc_idx, :]
    gap_x = x2 - x1
    gap_y = y2 - y1

    # Get the current reference
    ax = plt.gca()

    # Create a rectangle patch
    # This function reflect the x and y dim
    rect = patches.Rectangle((y1, x1), gap_y, gap_x, linewidth=3, edgecolor='b', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    image_name = f'{counter}_{image_file_name}_vc_{vc_idx}.jpg'
    plt.savefig(save_directory + '/' + image_name)
    plt.clf()
