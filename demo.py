import cv2
import numpy as np
import screenshot_simulator as ss

# print cv2.__version__ # 2.4.13

# reading image and cursor
img = cv2.imread('51.jpg')
mouse = cv2.imread('cursors/arrow_m.png', cv2.IMREAD_UNCHANGED)

fg_in = cv2.imread('home-256.png', cv2.IMREAD_UNCHANGED)
fg, mask = ss.distort_image(fg_in, (None, 128, 45))
f1_img = ss.paste_image(img, fg, (30, 30), 'out/test_test.jpg')

cv2.imshow('fg_in', fg_in)
cv2.imshow('fg', fg)
cv2.imshow('mask', mask)
cv2.imshow('vis', f1_img)

# f2 test
# res, mask = ss.f2(img, (300, 200, 45))

# cv2.imshow('img', img)
# cv2.imshow('res', res)
# cv2.imshow('mask', mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
