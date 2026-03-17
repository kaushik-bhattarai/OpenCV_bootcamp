import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt

from downloader import download_and_unzip

URL = r"https://www.dropbox.com/s/0oe92zziik5mwhf/opencv_bootcamp_assets_NB4.zip?dl=1"
folder_path = r"/home/kaushik/OpenCV_bootcamp/utils"
asset_zip_path = os.path.join(folder_path, "opencv_bootcamp_assets_NB4.zip")

#download
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
else:
    print("asset already exists, skipping download...")

#original image
img_bgr = cv2.imread("utils/New_Zealand_Coast.jpg", cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#display
plt.imshow(img_rgb)
plt.show()

#Addition or Brightness
matrix = np.ones(img_rgb.shape, dtype="uint8") * 50

img_rgb_brighter = cv2.add(img_rgb, matrix)
img_rgb_darker = cv2.subtract(img_rgb, matrix)

#show the images
plt.figure(figsize=[18,5])
plt.subplot(131);plt.imshow(img_rgb_darker);plt.title("darker");
plt.subplot(132);plt.imshow(img_rgb);plt.title("original");
plt.subplot(133);plt.imshow(img_rgb_brighter);plt.title("brighter");
plt.show()

#Multiplication or Contrast
matrix1 = np.ones(img_rgb.shape) * 0.8
matrix2 = np.ones(img_rgb.shape) * 1.2


img_rgb_darker = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))
img_rgb_brighter = np.uint8(cv2.multiply(np.float64(img_rgb), matrix2))

plt.figure(figsize=[18,5])
plt.subplot(131);plt.imshow(img_rgb_darker);plt.title("lower contrast");
plt.subplot(132);plt.imshow(img_rgb);plt.title("original");
plt.subplot(133);plt.imshow(img_rgb_brighter);plt.title("higher contrast");
plt.show()

#handling overflow using np.clip()

img_rgb_lower = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))
img_rgb_higher = np.uint8(np.clip(cv2.multiply(np.float64(img_rgb), matrix2), 0, 255))

plt.figure(figsize=[18,5])
plt.subplot(131);plt.imshow(img_rgb_lower);plt.title("lower contrast");
plt.subplot(132);plt.imshow(img_rgb);plt.title("original");
plt.subplot(133);plt.imshow(img_rgb_higher);plt.title("higher contrast");
plt.show()

#Image thresholding 
#retval, dst = cv2.threshold( src, thresh, maxval, type[, dst] )
img_read = cv2.imread("utils/building-windows.jpg", cv2.IMREAD_GRAYSCALE)
retval, img_thresh = cv2.threshold(img_read, 100, 255, cv2.THRESH_BINARY)

plt.figure(figsize=[18, 5])

plt.subplot(121);plt.imshow(img_read, cmap="gray");plt.title("original")
plt.subplot(122);plt.imshow(img_thresh, cmap="gray");plt.title("Threshold")
plt.show()
print(img_thresh.shape)

#application: sheet music reader

sheet_img = cv2.imread("utils/Piano_Sheet_Music.png", cv2.IMREAD_GRAYSCALE)
retval, sheet_img_thresh_gbl_1 = cv2.threshold(sheet_img, 50, 225, cv2.THRESH_BINARY)
retval, sheet_img_thresh_gbl_2 = cv2.threshold(sheet_img, 130, 225, cv2.THRESH_BINARY)
img_thresh_adp = cv2.adaptiveThreshold(sheet_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

plt.figure(figsize=[18,15])
plt.subplot(221);plt.imshow(sheet_img, cmap="gray");plt.title("original")
plt.subplot(222);plt.imshow(sheet_img_thresh_gbl_1, cmap="gray");plt.title("threshold(global 50)")
plt.subplot(223);plt.imshow(sheet_img_thresh_gbl_2, cmap="gray");plt.title("threshold(global 130)")
plt.subplot(224);plt.imshow(img_thresh_adp, cmap="gray");plt.title("threshold(adaptive)")
plt.show()

#bitwise operations

'''
Example API for cv2.bitwise_and(). Others include: cv2.bitwise_or(), cv2.bitwise_xor(), cv2.bitwise_not()

dst = cv2.bitwise_and( src1, src2[, dst[, mask]] )
'''

img_rec = cv2.imread("utils/rectangle.jpg", cv2.IMREAD_GRAYSCALE)
img_cir = cv2.imread("utils/circle.jpg", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=[20, 5])
plt.subplot(121);plt.imshow(img_rec, cmap="gray")
plt.subplot(122);plt.imshow(img_cir, cmap="gray")
plt.show()

#bitwise and operation
img_bit_and = cv2.bitwise_and(img_rec, img_cir, mask=None)
plt.imshow(img_bit_and, cmap="gray")
plt.show()

#bitwise or 
img_bit_or = cv2.bitwise_or(img_rec, img_cir, mask=None)
plt.imshow(img_bit_or, cmap="gray")
plt.show()

img_bit_xor = cv2.bitwise_xor(img_rec, img_cir, mask=None)
plt.imshow(img_bit_xor, cmap="gray")
plt.show()

#logo manipulation

#Read foreground image
img_bgr_foreground = cv2.imread("utils/coca-cola-logo.png")
img_rgb_foreground = cv2.cvtColor(img_bgr_foreground, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb_foreground)
plt.show()

logo_h = img_rgb_foreground.shape[0]
logo_w = img_rgb_foreground.shape[1]

#read background color
img_bgr_background = cv2.imread("utils/checkerboard_color.png")
img_rgb_background = cv2.cvtColor(img_bgr_background, cv2.COLOR_BGR2RGB)

#set desired width and maintain image aspect ratio
aspect_ratio = logo_w / img_rgb_background.shape[1]
dim = (logo_w, int(img_rgb_background.shape[0] * aspect_ratio))

#resize background image to same size as logo image
img_rgb_background = cv2.resize(img_rgb_background, dim, interpolation=cv2.INTER_AREA)

plt.imshow(img_rgb_background)
plt.show()

#create mask for original image
img_gray_foreground = cv2.cvtColor(img_rgb_foreground, cv2.COLOR_RGB2GRAY)


#apply global thresholding to create binary mask
retval, img_mask = cv2.threshold(img_gray_foreground, 127, 255, cv2.THRESH_BINARY)
plt.imshow(img_mask, cmap="gray")
plt.show()

#invert the mask
img_mask_inv = cv2.bitwise_not(img_mask)
plt.imshow(img_mask_inv, cmap="gray")

#apply background on the mask

img_background = cv2.bitwise_and(img_rgb_background, img_rgb_background, mask=img_mask)
plt.imshow(img_background)
plt.show()

#isolate foreground from image
img_foreground = cv2.bitwise_and(img_rgb_foreground, img_rgb_foreground, mask=img_mask_inv)
plt.imshow(img_foreground)
plt.show()

#merge foreground and background
result = cv2.add(img_background, img_foreground)
plt.imshow(result)
plt.show()

cv2.imwrite("utils/logo_final.png", result[:,:,::-1])