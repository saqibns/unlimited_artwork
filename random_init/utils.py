import numpy as np
from scipy import misc
import os

def generate_nd_inputs(width, height, scaling_factor, shift_factor, radial_distance,
                       random_low, random_high, num_randoms):
    """
    Create a multidimensional input for the art generator using the 2d matrix coordinates
    :param width: Width of the image to be generated
    :param height: Height of the image to be generated
    :param scaling_factor: Factor to scale the coordinates by
    :param shift_factor: Factor to shift the coordinates by
    :param radial_distance: Distance of the coordinate from the origin 
    :param random_low: Lower range of random numbers
    :param random_high: Upper range of random numbers
    :param num_randoms: Number of random dimensions
    :return: 
    """
    dimensions = list()
    xs = np.arange(width)
    ys = np.arange(height)
    pairs = np.transpose(np.array([np.tile(xs, len(ys)), np.repeat(ys, len(xs))]))
    pairs = pairs / scaling_factor
    pairs = pairs - shift_factor
    dimensions.append(pairs)

    if radial_distance:
        rs = np.sqrt(np.square(pairs[:, 0]) + np.square(pairs[:, 1]))
        dimensions.append(rs)

    for i in range(num_randoms):
        zs = np.repeat(np.random.uniform(random_low, random_high), width*height)
        dimensions.append(zs)

    return np.column_stack(dimensions)


def save_image(output_dir, img_name, img_arr, width, height):

    img_arr = img_arr.astype(np.uint8)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    final_img = np.zeros((width, height, 3))
    idx = 0
    for i in range(width):
        for j in range(height):
            img_subarr = img_arr[idx]
            final_img[i, j, 0] = img_subarr[0]
            final_img[i, j, 1] = img_subarr[1]
            final_img[i, j, 2] = img_subarr[2]
            idx += 1
            # print(idx, img_subarr)

    misc.imsave(os.path.join(output_dir, img_name), final_img)
