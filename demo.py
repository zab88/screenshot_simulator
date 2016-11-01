import cv2
import numpy as np
import screenshot_simulator as ss

# print cv2.__version__ # 2.4.13

# reading image and cursor
img = cv2.imread('51.jpg')
mouse = cv2.imread('cursors/01.png', cv2.IMREAD_UNCHANGED)

# f1 test
f1_img = ss.f1(img, mouse, (30, 30), 'out/test_f1.jpg')
cv2.imshow('vis', f1_img)

# f2 test
res, Mx = ss.f2(img, (300, 200, 45))
print(Mx)

cv2.imshow('img', img)
cv2.imshow('res', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
