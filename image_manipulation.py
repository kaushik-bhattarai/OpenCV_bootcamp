import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve
from downloader import download_and_unzip

URL = r"https://www.dropbox.com/s/rys6f1vprily2bg/opencv_bootcamp_assets_NB2.zip?dl=1"
asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_NB2.zip")


#download
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
else:
    print("asset already exist. skipping download.")


'''
#Read image 
cb_img = cv2.imread("checkerboard_18x18.png", 0)
plt.imshow(cb_img, cmap= "gray")
print(cb_img)
plt.show()
'''

'''
#accessing individual pixels
print(cb_img[0,0])
print(cb_img[0,6])
'''

'''
#modifying image pixels
cb_img_copy = cb_img.copy()
cb_img_copy[2:4,2:4] = 200
plt.imshow(cb_img_copy, cmap="gray")
print(cb_img_copy)
plt.show()
'''


#cropping image 
imz_NZ_bgr = cv2.imread("New_Zealand_Boat.jpg", cv2.IMREAD_COLOR)
imz_NZ_rgb = cv2.cvtColor(imz_NZ_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(imz_NZ_rgb)
plt.show()
cropped_region = imz_NZ_rgb[200:400, 300:600]
plt.imshow(cropped_region)
plt.show()


#resizing an Image
'''
dst = resize(src, dsize[,dst[,fx[,fy[,interpolation]]]])

resized_cropped_region_2x = cv2.resize(cropped_region, None, fx=2, fy=2)
plt.imshow(resized_cropped_region_2x)
plt.show()

#method 2

desired_width = 100
desired_heigt = 200
dim = (desired_width, desired_heigt)

resized_cropped_region = cv2.resize(cropped_region, dsize = dim, interpolation = cv2.INTER_AREA)
plt.imshow(resized_cropped_region)
print(resized_cropped_region)
plt.show()

#resize while maintaining aspect ratio
desired_width = 250
aspect_ratio = desired_width/ cropped_region.shape[1]
desired_heigt= int(cropped_region.shape[0] * aspect_ratio)
dim = (desired_width, desired_heigt)

resized_cropped_region = cv2.resize(cropped_region, dsize = dim, interpolation = cv2.INTER_AREA)
plt.imshow(resized_cropped_region)
plt.show()
'''

#flipping an image

'''
dst = cv2.flip(src, flipcode)
flipcode:flag
0-> flip around x-axis
+ve(eg 1)-> flip around y-axis
-ve(eg -1)-> flip around both axes
'''
imz_NZ_rgb_flipped_horz = cv2.flip(imz_NZ_rgb, 1)
imz_NZ_rgb_flipped_vert = cv2.flip(imz_NZ_rgb, 0)
imz_NZ_rgb_flipped_both = cv2.flip(imz_NZ_rgb, -1)

plt.figure(figsize=(18, 5))
plt.subplot(141);plt.imshow(imz_NZ_rgb_flipped_horz);plt.title("Horizontal flip");
plt.subplot(142);plt.imshow(imz_NZ_rgb_flipped_vert);plt.title("Vertical flip");
plt.subplot(143);plt.imshow(imz_NZ_rgb_flipped_both);plt.title("Both flip");
plt.subplot(144);plt.imshow(imz_NZ_rgb);plt.title("Original");
plt.show()