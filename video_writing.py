import os
import cv2
import matplotlib.pyplot as plt

from downloader import download_and_unzip

URL = r"https://www.dropbox.com/s/p8h7ckeo2dn1jtz/opencv_bootcamp_assets_NB6.zip?dl=1"
folder_path = r"/home/kaushik/OpenCV_bootcamp/utils"
asset_zip_path = os.path.join(folder_path, "opencv_bootcamp_assets_NB6.zip")

# Download if assest ZIP does not exists. 
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)   

#read video from source
source = 'utils/race_car.mp4'
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("Error opening video stream or file")

ret, frame = cap.read()
if not ret or frame is None:
    print("Failed to read frame")
else:
    # Convert BGR to RGB for correct colors in matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.show()


#write video

'''
VideoWriter object = cv.VideoWriter(filename, fourcc, fps, frameSize )


fourcc: 4-character code of codec used to compress the frames.
For example, VideoWriter::fourcc('P','I','M','1') is a MPEG-1 codec,
VideoWriter::fourcc('M','J','P','G') is a motion-jpeg codec etc.
List of codes can be obtained at Video Codecs by FOURCC page.
FFMPEG backend with MP4 container natively uses other values as fourcc code: see ObjectType, 
so you may receive a warning message from OpenCV about fourcc code conversion.
'''

# Default resolutions of the frame are obtained.
# Convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

#define codec and create VideoWriter object
out_avi = cv2.VideoWriter("utils/race_car_out.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"),fps,(frame_width, frame_height) )
out_mp4 = cv2.VideoWriter("utils/race_car_out.mp4", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

#read frames and write to file
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        out_avi.write(frame)
        out_mp4.write(frame)
    else:
        break

cap.release()
out_avi.release()
out_mp4.release()

#play video
win_name = "MP4 Playback"
cap = cv2.VideoCapture("utils/race_car_out.mp4")
cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE) 

while cv2.waitKey(10) != 27:  # Exit on Escape
    has_frame, frame = cap.read()
    if not has_frame:
        break

    cv2.imshow(win_name, frame)

cap.release()
cv2.destroyWindow(win_name)


