import os
import cv2
import math
import glob
import numpy as np
import matplotlib.pyplot as plt

from downloader import download_and_unzip

URL = r"https://www.dropbox.com/s/0o5yqql1ynx31bi/opencv_bootcamp_assets_NB9.zip?dl=1"
folder_path = r"/home/kaushik/OpenCV_bootcamp/utils"
asset_zip_path = os.path.join(folder_path, "opencv_bootcamp_assets_NB9.zip")

#download if zip doesnot exist:
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
else:
    print("asset aleady exists, skipping download")
 

imagefiles = glob.glob(os.path.join(folder_path, "boat", "*"))
imagefiles.sort()

images = []

for filename in imagefiles:
    img = cv2.imread(filename)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

num_images = len(images)

# Display Images
plt.figure(figsize=[30, 10])
num_cols = 3
num_rows = math.ceil(num_images / num_cols)
for i in range(0, num_images):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.axis("off")
    plt.imshow(images[i])
plt.show()

# Stitch Images
stitcher = cv2.Stitcher_create()
status, result = stitcher.stitch(images)

if status == 0:
    plt.figure(figsize=[30, 10])
    plt.imshow(result)
plt.show()

