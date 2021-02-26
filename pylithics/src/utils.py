import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def area_contour(contour):
    """
    Function that calculates the area within cont contour using the open-cv library.

    Parameters
    ----------
    contour: array (array with coordinates defining the contour.)

    Returns
    -------
    A number

    """
    # Expand numpy dimensions and convert it to UMat object
    c = cv2.UMat(np.expand_dims(contour.astype(np.float32), 1))
    area = cv2.contourArea(c)

    return area

def contour_desambiguiation(df_contours):
    """

    Funtion that selects contours by their size and removes duplicates.

    Parameters
    ----------
    df_contours: dataframe
        Dataframe with contour information.

    Returns
    -------

    """

    norm = max(df_contours['area_px'])
    index_to_drop = []

    for i in range(df_contours.shape[0]):
        area = df_contours['area_px'].iloc[i]
        percentage = area / norm * 100

        if percentage < 1:
            index_to_drop.append(i)


    cent_df = df_contours[['area_px', 'centroid']]

    import itertools

    for i, j in itertools.combinations(cent_df.index, 2):

        if ((i in index_to_drop) or (j in index_to_drop)):
            continue

        d_ij_area = np.linalg.norm(cent_df.loc[i]['area_px'] - cent_df.loc[j]['area_px'])
        d_ij_centroid = np.linalg.norm(np.asarray(cent_df.loc[i]['centroid']) - np.asarray(cent_df.loc[j]['centroid']))

        if d_ij_centroid<300:
            if d_ij_area/norm<0.1:
                if cent_df.loc[i]['area_px'] < cent_df.loc[j]['area_px']:
                    index_to_drop.append(i)
                else:
                    index_to_drop.append(j)

    return index_to_drop

def mask_image(image_array, contour, innermask = False):
    """

    Function that masks an image for cont given contour.

    Parameters
    ----------
    image_array
    contour

    Returns
    -------

    """

    import scipy.ndimage as ndimage

    r_mask = np.zeros_like(image_array, dtype='bool')
    r_mask[np.round(contour[:, 1]).astype('int'), np.round(contour[:, 0]).astype('int')] = 1

    r_mask = ndimage.binary_fill_holes(r_mask)

    if innermask:
        new_image = r_mask
    else:
        new_image = np.multiply(r_mask,image_array)

    return new_image


def contour_characterisation(cont, conversion = 96):
    """

    For cont given contour calculate characteristics (area, lenght, etc.)

    Parameters
    ----------
    cont: array
        Array of pairs of pixel coordinates
    conversion: float
        Value to convert pixels to inches

    Returns
    -------
    A dictionary

    """
    cont_info = {}


    # Expand numpy dimensions and convert it to UMat object
    area = area_contour(cont)

    cont_info['lenght'] = len(cont)
    cont_info['area_px'] = area
    cont_info['area_cm'] = round(area / conversion, 1) # to cm based on centimeters: 1cm = 96px/2.54

    return cont_info

def classify_distributions(image_array):
    """
    Given an input image array classify it by their distribution of pixel intensities.
    Returns True is the ditribution is narrow and skewed to values of 1.

    Parameters
    ----------
    image_array: array

    Returns
    -------
    a boolean

    """

    is_narrow = False

    fig, axes = plt.subplots(figsize=(8, 2.5))

    axes.hist(image_array.ravel(), bins=256)
    axes.set_title('Histogram')
    plt.close(fig)

    image_array_nonzero = image_array > 0

    mean = np.mean(image_array[image_array_nonzero])

    std = np.std(image_array[image_array_nonzero])

    if mean>0.9 and std<0.15:
        is_narrow = True

    return is_narrow

def add_highest_level_parent(hierarchies):

    """ For a list of contour hierarchies find the index of the
    highest level parent for each contour.

     Parameters
    ----------
    hierarchies: list
        List of hierarchies

    Returns
    -------
    A list
    """

    parent_index = []

    for index, hierarchy in enumerate(hierarchies, start=0):

        parent = hierarchy[-1]
        if parent == -1:
            parent_index.append(parent)

        else:

            while (parent!=-1):
                index = parent
                parent = hierarchies[index][-1]

            parent_index.append(index)

    return parent_index

def pixulator (image_scale_array, scale_size):

    """
    Converts image/scale dpi and pixel count to cm conversion rate.
    :param image_scale_array:
    :param cm:
    :return: Image dpi conversion to centimeters.
    """


    # dimension information in pixels
    px_width = image_scale_array.shape[0]
    px_height = image_scale_array.shape[1]

    if px_width > px_height:
        orientation = px_width
    else:
        orientation = px_height

    px_conversion = orientation/scale_size
    print(f"1 cm will equate to {px_conversion} pixels.")
    return(px_conversion)










