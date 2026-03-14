import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from zipfile import ZipFile
from urllib.request import urlretrieve
from downloader import download_and_unzip

URL = r"https://www.dropbox.com/s/48hboi1m4crv1tl/opencv_bootcamp_assets_NB3.zip?dl=1"
folder_path = r"/home/kaushik/opencv/utils"
asset_zip_path = os.path.join(folder_path, "opencv_bootcamp_assets_NB3.zip")

#download
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
else:
    print("asset aleady exists, skipping download")
    

image = cv2.imread("utils/Apollo_11_Launch.jpg", cv2.IMREAD_COLOR)
color_correct_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(color_correct_image)
plt.show()


#drawing a line

'''
img = cv2.line(img, pt1, pt2, color[,thickness[,lineType[,shift]]])
'''
imageLine = color_correct_image.copy()

cv2.line(imageLine, (200,100), (400,100), (0,0,255), thickness=5, lineType=cv2.LINE_AA)

plt.imshow(imageLine)
plt.show()

#drawing a circle
'''
img = cv2.circle(img, center, radius, color[,thickness[,lineType[,shift]]])
'''
imageCircle = color_correct_image.copy()

cv2.circle(imageCircle, (600,100), 100, (0,0,255), thickness=5, lineType=cv2.LINE_AA)

plt.imshow(imageCircle)
plt.show()

#rectangle
'''
img = cv2.rectangle(img, pt1, pt2, color[,thickness[,lineType[,shift]]])# pt1 and pt2 are two vertices, top left and bottom right
'''

imageRectangle = color_correct_image.copy()

cv2.rectangle(imageRectangle, (500,100), (700,600), (0,0,255), thickness=5, lineType=cv2.LINE_AA)
plt.imshow(imageRectangle)
plt.show()

#adding text
'''
img = cv2.putText(img, text, org, fontFace, fontScale, color[,thickness[,lineType[,bottomLeftOrigin]]])



    img: Image on which the text has to be written.

    text: Text string to be written.

    org: Bottom-left corner of the text string in the image.

    fontFace: Font type

    fontScale: Font scale factor that is multiplied by the font-specific base size.

    color: Font color


'''

imageText = color_correct_image.copy()
text = "Apollo 11 Saturn V Launch, July 16, 1969"

cv2.putText(imageText, text, (200,700),fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 2.3,
            color = (0,255,0), thickness = 2, lineType=cv2.LINE_AA)

plt.imshow(imageText)
plt.show()

