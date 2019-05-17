import cv2

import os

image_path = 'HSV/'

save_path_hsv = 'HSV/'

save_path_ycrcb = 'HSV/'

# *********cv2.read(image)=(B, G, R)**********************
# save_path_bgr = 'HSV/'
# image_path = 'HSV/M1.jpg'
# img = cv2.imread(image_path)
# imgcopy1=img
# imgcopy2=img
# imgcopy3=img
# # imgcopy1[:, :, 1:3] = 0
# # b=imgcopy1
# # imgcopy2[:, :, 0] = 0
# # imgcopy2[:, :, 2] = 0
# # g=imgcopy2
# imgcopy3[:, :, 0:2] = 0
# r=imgcopy3
#
# save_b = save_path_bgr + 'b.jpg'
# save_g = save_path_bgr  + 'g.jpg'
# save_r = save_path_bgr  + 'r.jpg'
#
# # cv2.imwrite(save_b, b)
# # cv2.imwrite(save_g, g)
# cv2.imwrite(save_r, r)
#
# from skimage import io, data, color
# img=io.imread('HSV/M1.jpg')
# # image = image.chelsea()
# image_hsv = color.convert_colorspace(img, 'RGB', 'HSV')
# io.imshow(image_hsv)
# io.show()
# io.imsave('HSV/M1_hsv_ski.jpg',image_hsv)

#
def brg2hsv_ycrcb(image_path, save_path_hsv, save_path_ycrcb):
    filenames = os.listdir(image_path)

    for filename in filenames:
        examname = filename[:-4]

        type = filename.split('.')[-1]

        img = cv2.imread(image_path + '/' + filename)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        save_hsv = save_path_hsv + examname + '_HSV' + '.' + type
        save_gray = save_path_hsv + examname + '_GRAY' + '.' + type
        save_ycrcb = save_path_ycrcb + examname + '_YCrCb' + '.' + type

        cv2.imwrite(save_hsv, img_hsv)
        cv2.imwrite(save_gray, img_gray)
        cv2.imwrite(save_ycrcb, img_ycrcb)


if __name__ == '__main__':
    brg2hsv_ycrcb(image_path, save_path_hsv, save_path_ycrcb)