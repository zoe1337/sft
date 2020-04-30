#!/usr/bin/env python3
import numpy as np
import cv2
import imageio
import os
import argparse
import fcntl
import v4l2

target_imgw, target_imgh = 1280, 720 # what we want from the camera
#target_imgw, target_imgh = 1920, 1080 # what we want from the camera
target_fps = 30
cropw, croph = 420, 480 # this is the size of the RoI (region of interest)
moving_window_size = 10 # frames of position averaging
draw_debug_rectangles = True
sf = 0.25 # scale factor; downsample image to speed up feature detection
loopbackDevName = '/dev/video2'

class Cat:
    def __init__(self, cropw, croph, imgw, imgh):
        global moving_window_size
        self.imgw = imgw
        self.imgh = imgh
        self.cropw = cropw
        self.croph = croph
        self.last_x, self.last_y = imgw/2, imgh/2
        self.centerx, self.centery = self.last_x, self.last_y
        self._moving_window_size = moving_window_size
        self._index = 0
        self._xarray = np.array([self.last_x] * self._moving_window_size)
        self._yarray = np.array([self.last_y] * self._moving_window_size)

    def update(self, x, y, w, h):
        global sf # scale factor
        self.last_x = (x + w/2)/sf
        self.last_y = (y + h/2)/sf

        self._xarray[self._index] = self.last_x
        self._yarray[self._index] = self.last_y
        self._index = self._index + 1
        if self._index >= self._moving_window_size:
            self._index = 0
        
        self.centerx = np.sum(self._xarray) / self._moving_window_size
        self.centery = np.sum(self._yarray) / self._moving_window_size
        pass

    # returns a x,y,w,h crop parameter based on the moving average
    def getCrop(self):
        pad_left = int(np.clip(self.centerx - self.cropw/2, 0, self.imgw - self.cropw))
        pad_top  = int(np.clip(self.centery - self.croph/2, 0, self.imgh - self.croph))
        return (pad_left, pad_top, self.cropw, self.croph)

    # gives you the minimum distance of the cat frame from the param coords x,y
    def getDistance(self, x, y, w, h):
        dist = np.sqrt((self.centerx - (x+w/h)/sf)**2 + (self.centery - (y+h/2)/sf)**2)
        return dist

use_classifier = "haarcascade" # or "lbpcascades"

opencv_base_path = '/usr/share/opencv4/'
if use_classifier == "haarcascade":
    face_cascade = cv2.CascadeClassifier(opencv_base_path + 'haarcascades/haarcascade_frontalface_default.xml')
    face_profile = cv2.CascadeClassifier(opencv_base_path + 'haarcascades/haarcascade_profileface.xml')
    #eye_cascade = cv2.CascadeClassifier(opencv_base_path + 'haarcascades/haarcascade_eye.xml')
else:
    face_cascade = cv2.CascadeClassifier(opencv_base_path + 'lbpcascades/lbpcascade_frontalface_improved.xml')
    face_profile = cv2.CascadeClassifier(opencv_base_path + 'lbpcascades/lbpcascade_profileface.xml')


def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def draw_frame(img, x, y, w, h, color):
    global draw_debug_rectangles
    if draw_debug_rectangles:
        global sf
        origx, origy = int(x / sf), int(y / sf)
        origw, origh = int(w / sf), int(h / sf)
        img = cv2.rectangle(img,(origx,origy),(origx+origw,origy+origh),color,2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Shitty Face Tracker')
    parser.add_argument('--gui', action='store_true', help='Show the video output in a window.')
    parser.add_argument('--debug', action='store_true', help='Draw debugging rectangles on the video.')

    args = parser.parse_args()
    draw_debug_rectangles = args.debug

    cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_V4L2)
    set_res(cap, target_imgw, target_imgh)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    print("Capture format before asking: %s" % cap.get(cv2.CAP_PROP_FORMAT))
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    print("Capture format now: %s" % cap.get(cv2.CAP_PROP_FORMAT))
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'BGR3'))
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YV12'))

    if not os.path.exists(loopbackDevName):
        print ("Error: device {} does not exist".format(loopbackDevName))
        exit(1)
    device = open(loopbackDevName, 'wb')

    ret, frame = cap.read() # TODO ret is false if no image has been returned
    imgh, imgw, channels = frame.shape
    roi = Cat(cropw, croph, imgw, imgh)

    o_height, o_width             = croph, cropw
    loformat                      = v4l2.v4l2_format()
    loformat.type                 = v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT
    loformat.fmt.pix.field        = v4l2.V4L2_FIELD_NONE
    loformat.fmt.pix.pixelformat  = v4l2.V4L2_PIX_FMT_BGR24
    loformat.fmt.pix.width        = o_width
    loformat.fmt.pix.height       = o_height
    loformat.fmt.pix.bytesperline = o_width * channels
    loformat.fmt.pix.sizeimage    = o_width * o_height * channels

    result = fcntl.ioctl(device, v4l2.VIDIOC_S_FMT, loformat)
    if result != 0:
        print("Error setting format! (Error code {})".format(result))
        exit(2)


    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        img = frame
        resized = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_LINEAR)
        grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        faceArray = face_cascade.detectMultiScale(grayImg, 1.3, 5)
        color = (200,30,30)

        if len(faceArray) == 0:
            faceArray = face_profile.detectMultiScale(grayImg, 1.3, 5)
            color = (110,0,110)

        if len(faceArray) > 0:
            if len(faceArray) > 1:
                distances = []
                for (x,y,w,h) in faceArray:
                    distances.append(roi.getDistance(x, y, w, h))
                    draw_frame(img, x, y, w, h, (70,70,70))
                nearestIdx = np.argmin(distances)
            else:
                nearestIdx = 0
            (x,y,w,h) = faceArray[nearestIdx]
            roi.update(x,y,w,h)
            draw_frame(img, x, y, w, h, color)
            
        cx, cy, cw, ch = roi.getCrop()
        # print("face at x: %i, y: %i, w: %i, h: %i" % (x/sf,y/sf,w/sf,h/sf))
        # print("crop at x: %i, y: %i, w: %i, h: %i" % (cx,cy,cw,ch))
        roi_color = img[cy:cy+ch, cx:cx+cw]

        if args.gui:
            cv2.imshow('cat', roi_color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        device.write(roi_color.ravel())

    cap.release()
    cv2.destroyAllWindows()
