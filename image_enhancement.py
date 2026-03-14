import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt

from downloader import download_and_unzip

URL = r"https://www.dropbox.com/s/0oe92zziik5mwhf/opencv_bootcamp_assets_NB4.zip?dl=1"
folder_path = r"/home/kaushik/opencv/utils"
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