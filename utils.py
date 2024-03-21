import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def highlight_object(image, binary_mask):
    image[binary_mask == 255] = np.array([255,0,0])
    return image

def dilate_and_erode(image, kernel_size=7, iterations=2):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv.dilate(image, kernel, iterations=iterations)
    eroded = cv.erode(dilated, kernel, iterations=iterations)
    eroded[eroded > 0] = 255
    return eroded

def shift_object(image, mask, current_left_corner, new_left_corner):

    dx = new_left_corner[0] - current_left_corner[0]
    dy = new_left_corner[1] - current_left_corner[1]
    dx = int(dx)
    dy = int(dy)
    shifted_image = np.roll(image, dx, axis=1)
    shifted_image = np.roll(shifted_image, dy, axis=0)

    shifted_mask = np.roll(mask, dx, axis=1)
    shifted_mask = np.roll(shifted_mask, dy, axis=0)

    return shifted_image, shifted_mask

def move_image(img_arr, mask_arr):
    min_row, min_col, max_row, max_col = bbox(mask_arr)
    centreObject = ((min_row + max_row) / 2, (min_col + max_col) / 2)
    centreImage = (img_arr.shape[0] / 2, img_arr.shape[1] / 2)

    rowDiff = centreImage[0] - centreObject[0]
    colDiff = centreImage[1] - centreObject[1]

    new_min_row = min_row + rowDiff
    new_max_row = max_row + rowDiff
    new_min_col = min_col + colDiff
    new_max_col = max_col + colDiff

    newImg = np.ones_like(img_arr) * 255

    newImg[int(new_min_row):int(new_max_row), int(new_min_col):int(new_max_col)] = img_arr[min_row:max_row, min_col:max_col]

    return newImg

def bbox(mask_arr):
    rows = mask_arr.shape[0]
    cols = mask_arr.shape[1]

    min_row = rows
    min_col = cols
    max_row = 0
    max_col = 0

    for i in range(rows):
        for j in range(cols):
            if mask_arr[i, j] > 0:
                min_row = min(min_row, i)
                min_col = min(min_col, j)
                max_row = max(max_row, i)
                max_col = max(max_col, j)

    return min_row, min_col, max_row, max_col

def saveImage(image_array, path, cmap='gray'):
    if cmap is None:
        plt.imsave(path, image_array)
    else:
        plt.imsave(path, image_array, cmap=cmap)

def threshold_mask(mask_arr):
    ret, mask_arr = cv.threshold(mask_arr, 200, 255, cv.THRESH_BINARY)
    if mask_arr.shape[-1] == 3:
        mask_arr = mask_arr[:, :, 0]
    mask_arr = mask_arr.astype(np.uint8)
    return mask_arr

def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]
