import numpy as np
import cv2


# A collection of image processing snippets for ease of use

def translate(image, x, y):
    """
    Shifts an image
    :param image: Image to shift
    :param x: Shift X
    :param y: Shift Y
    :return: Shifted Image
    """
    # Get image dimensions
    (h, w) = image.shape[:2]

    translation = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image.copy(), translation, (w, h))
    return shifted


def rotate(image, angle, center=None, scale=1.0):
    """
    Rotate an image by the given center.
    :param image: Image to rotate
    :param angle: Rotation angle
    :param center: Point to rotate from
    :param scale: Size to resize image
    :return: Rotated image
    """
    # Get image dimensions
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    rotation = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image.copy(), rotation, (w, h))

    return rotated


def resize(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    """
    Resize an image
    :param image: Image to be resized
    :param width: New width
    :param height: New height
    :param interpolation: Algorithm to use
    :return: Resized image
    """
    dim = None
    # Get image dimensions
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image.copy(), dim, interpolation=interpolation)

    return resized


def flip(image, horizontal=False, vertical=False):
    """
    Flip and image
    :param image: Image to be flipped
    :param horizontal: If flip horizontally
    :param vertical: If flip vertically
    :return: Flipped image
    """
    # Code to be used when flipping
    code = None

    if not horizontal and not vertical:
        return image
    elif horizontal and vertical:
        code = -1
    elif horizontal:
        code = 1
    else:
        code = 0

    flipped = cv2.flip(image.copy(), code)

    return flipped


def crop(image, startX, startY, endX, endY):
    """
    Crop an image
    :param image: Image to be cropped
    :param startX: Starting X coord
    :param startY: Starting Y coord
    :param endX: Ending X coord
    :param endY: Ending Y coord
    :return: Cropped image
    """

    cropped = image[startY:endY, startX:endX]

    return cropped


def combine(fore_image, back_image, x_offset, y_offset):
    """
    Places an foreground image inside a background image
    :param fore_image: The image to be placed on top
    :param back_image: The image to be edited
    :param x_offset: x-axis offset for the fore_image
    :param y_offset: y-axis offset for the fore_image
    :return: Combined image
    """
    # Copy image to avoid errors
    background_img = back_image.copy()

    # Grab the region of interest (ROI) in the background
    rows, cols, channels = fore_image.shape
    roi = background_img[y_offset:rows + y_offset, x_offset:cols + x_offset]

    # Mask the foreground image
    img2gray = cv2.cvtColor(fore_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Place fore img silhouette in the ROI
    back_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of fore img
    fore_fg = cv2.bitwise_and(fore_image, fore_image, mask=mask)

    # Place fore img in ROI
    dst = cv2.add(back_bg, fore_fg)

    # Replace ROI in original background image
    background_img[y_offset:rows + y_offset, x_offset:cols + x_offset] = dst

    return background_img


def create_blank(width, height, rgb=(0, 0, 0)):
    # Create blank image
    image = np.zeros((height, width, 3), np.uint8)
    # Convert to BGR
    color = tuple(reversed(rgb))
    # Fill image with color
    image[:] = color

    return image
