import numpy as np
import cv2
import matplotlib.pyplot as plt

img_path = "/home/kaushik/Downloads/PageRanks-Example.svg.png"
img = cv2.imread(img_path)

#color correction
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#split the image to B,G,R components 
b,g,r = cv2.split(img)
#merge the split images if needed
#merged_img = cv2.merge((b,g,r))

plt.figure(figsize=[20, 5])
plt.subplot(141); plt.imshow(r, cmap='gray'); plt.title("red channel")
plt.subplot(142); plt.imshow(g, cmap='gray'); plt.title("green channel")
plt.subplot(143); plt.imshow(b, cmap='gray'); plt.title("blue channel")
plt.subplot(144); plt.imshow(img, cmap='gray'); plt.title("merged channel") 
plt.show()

hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#split the image to H,S,V components 
h,s,v = cv2.split(hsv_img)

plt.figure(figsize=[20, 5])
plt.subplot(141); plt.imshow(h, cmap='gray'); plt.title("hue channel")
plt.subplot(142); plt.imshow(s, cmap='gray'); plt.title("saturation channel")
plt.subplot(143); plt.imshow(v, cmap='gray'); plt.title("value channel")
plt.subplot(144); plt.imshow(img, cmap='gray'); plt.title("merged channel") 
plt.show()

#modifying individual channel
h_new = h+10
new_img = cv2.merge((h_new,s,v))
img_modified_rgb = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)

plt.figure(figsize=[20, 5])
plt.subplot(141); plt.imshow(h_new, cmap='gray'); plt.title("hue channel")
plt.subplot(142); plt.imshow(s, cmap='gray'); plt.title("saturation channel")
plt.subplot(143); plt.imshow(v, cmap='gray'); plt.title("value channel")
plt.subplot(144); plt.imshow(img_modified_rgb, cmap='gray'); plt.title("merged channel") 
plt.show()


#print the image data,2D numpy array
#print(img)

#img shape and data type 
#print(f"the shape of image is {img.shape}")
#print(f"the data type of image is: {img.dtype}")

#plt.imshow(img)
#plt.show()


#saving img
#cv2.imwrite("/home/kaushik/Downloads/", img_modified_rgb)  