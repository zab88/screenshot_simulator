import cv2
import numpy as np
import math


def f1(bg, mouse, location, filename):
    """
    :param bg: Background image (OpenCV image)
    :param mouse: mage of a mouse pointer (OpenCV image)
    :param location: tuple (x,y) in percentages of background image width/height
    :param filename: Filename where to save the result image (string)
    :return: OpenCV image
    """
    rows, cols, channels = mouse.shape
    if channels != 4:
        print('cursor image without transparent layer!')
        return None

    # real offset in pixels
    bg_h, bg_w = bg.shape[:2]
    x, y = int(float(location[0]) * float(bg_w) / 100.), int(float(location[1]) * float(bg_h) / 100.)
    roi = bg[y:y + rows, x:x + cols]

    # Now create a mask of cursor and create its inverse mask also
    if channels == 4:
        mask = np.array(mouse[:, :, 3], np.uint8)
    else:
        mask = np.zeros((rows, cols), np.uint8)

    mouse = cv2.merge([mouse[:, :, 0], mouse[:, :, 1], mouse[:, :, 2]])
    mask_inv = cv2.bitwise_not(mask)

    # if mouse out of bounds, mask should be contracted
    mask = mask[:roi.shape[0], :roi.shape[1]]
    mask_inv = mask_inv[:roi.shape[0], :roi.shape[1]]
    mouse = mouse[:roi.shape[0], :roi.shape[1]]

    # Now black-out the area of cursor in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of cursor from cursor image.
    img2_fg = cv2.bitwise_and(mouse, mouse, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    bg[y:y + rows, x:x + cols] = dst

    # Saving image
    cv2.imwrite(filename, bg)

    return bg


def f2(cv_img, scale_params, transparent_bg=True):
    """
    :param cv_img: opencv image
    :param scale_params: tuple height/width/rotation degree
    :return: image, matrix
    """
    rows, cols, channels = cv_img.shape

    # first - resize width and height
    # if new width or height is None, then no resize
    if (scale_params[0] is None) or (scale_params[1] is None):
        res = cv_img
    else:
        fy = (float(scale_params[0]) / rows)
        fx = (float(scale_params[1]) / cols)
        res = cv2.resize(cv_img, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)  # INTER_AREA is faster then INTER_CUBIC

    # if no rotation
    if not scale_params[2]:
        return res, res[:,:,3] if channels == 4 else np.zeros((rows, cols), np.uint8)

    # to avoid missing corners after rotation - image should be expanded
    rows, cols = res.shape[:2]
    img_ext = res.copy()
    radius = int(math.ceil(math.sqrt(rows * rows + cols * cols) / 2.0))  # transformed image definitely in circle with that radius
    lr_border, tb_border = radius - int(float(cols) / 2.0), radius - int(float(rows) / 2.0)
    # extending with borders
    img_ext = cv2.copyMakeBorder(img_ext, tb_border, tb_border, lr_border, lr_border, cv2.BORDER_CONSTANT, img_ext, (0, 0, 0))
    rows_ext, cols_ext = img_ext.shape[:2]

    # mask creating
    mask_ = np.zeros((rows_ext, cols_ext), np.uint8)
    cv2.rectangle(mask_, (lr_border, tb_border), (cols_ext - lr_border, rows_ext - tb_border), 255, -1)

    # second - apply rotation
    # rotation matrix creation
    M = cv2.getRotationMatrix2D((cols_ext / 2, rows_ext / 2), scale_params[2], 1)

    # affine transform
    dst = cv2.warpAffine(img_ext, M, (cols_ext, rows_ext))  # borderMode=cv2.BORDER_TRANSPARENT, borderValue=0

    # mask can be not rectangle, if input image was with transparency
    if dst.shape[2] == 4:
        mask_ = dst[:,:,3]
    else:
        mask_ = cv2.warpAffine(mask_, M, (cols_ext, rows_ext))

    return dst, mask_
