import cv2
import numpy as np
import screenshot_simulator as ss
import time

# test on 100 images
def speed_test(bg, mouse):
    start_time = time.time()
    for x in range(20, 81, 15):
        for y in range(20, 81, 15):
            ss.paste_image(bg, ss.distort_image( mouse, (120, 120, 0)), (x,y), 'out/0_%i_%i.jpg' % (x, y))
            ss.paste_image(bg, ss.distort_image( mouse, (120, 120, 90)), (x,y), 'out/90_%i_%i.jpg' % (x, y))
            ss.paste_image(bg, ss.distort_image( mouse, (120, 120, 180)), (x,y), 'out/180_%i_%i.jpg' % (x, y))
            ss.paste_image(bg, ss.distort_image( mouse, (120, 120, 270)), (x,y), 'out/270_%i_%i.jpg' % (x, y))
    print("--- %s seconds ---" % (time.time() - start_time))

# reading image and cursor
bg = cv2.imread('bg_100x100.jpg')
mouse = cv2.imread('cursors/arrow_m.png', cv2.IMREAD_UNCHANGED)

speed_test(bg, mouse)


fg = ss.distort_image(mouse, (150, 50, 180))
bg = ss.paste_image(bg, fg, (20, 70), 'out/test_test.jpg')

cv2.imshow('fg', fg)
cv2.imshow('bg', bg)

cv2.waitKey(0)
cv2.destroyAllWindows()

