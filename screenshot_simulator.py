import cv2
import numpy as np

def f1(bg, mouse, location, filename):
    """
    :param bg: Background image (OpenCV image)
    :param mouse: mage of a mouse pointer (OpenCV image)
    :param location: tuple (x,y) in percentages of background image width/height
    :param filename: Filename where to save the result image (string)
    :return:
    """
    rows, cols, channels = mouse.shape
    if channels != 4:
        print('cursor image without transparent layer!')
        return None

    # real offset in pixels
    bg_h, bg_w = bg.shape[:2]
    x, y =  int(float(location[0])*float(bg_w)/100.), int(float(location[1])*float(bg_h)/100.)
    roi = bg[y:y+rows, x:x+cols ]

    # Now create a mask of cursor and create its inverse mask also
    mask = mouse[:,:,3]
    mouse = cv2.merge([mouse[:,:,0], mouse[:,:,1], mouse[:,:,2]])
    mask_inv = cv2.bitwise_not(mask)
    print(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(mouse, mouse, mask = mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    bg[y:y+rows, x:x+cols ] = dst

    # Saving image
    cv2.imwrite(filename, bg)

    return bg

def f2(cv_img, scale_params):
    """
    :param cv_img: opencv image
    :param scale_params: tuple height/width/rotation degree
    :return: image, matrix
    """
    rows, cols = cv_img.shape[:2]

    # first - resize width and height
    fy = (float(scale_params[0]) / rows)
    fx = (float(scale_params[1]) / cols)
    res = cv2.resize(cv_img, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)  # INTER_AREA is faster then INTER_CUBIC

    # second - apply rotation
    # rotation matrix creation
    rows, cols = res.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), scale_params[2], 1)
    # M = np.float32([[1,0,100],[0,1,50]])

    # adding scale params
    dst = cv2.warpAffine(res, M, (cols, rows))
    return dst, M
