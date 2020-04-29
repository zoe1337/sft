#!/usr/bin/env python3
import numpy as np
import cv2
import imageio
import os

control_file = "/tmp/lm-ctrl"
anim_file = "lm-anim.gif"
mask_file = "lm-mask.png"

target_imgw, target_imgh = 1280, 720 # what we want from the camera
#target_imgw, target_imgh = 1920, 1080 # what we want from the camera

target_fps = 30
cropw, croph = 420, 480 # this is the size of the RoI (region of interest)
moving_window_size = 10 # frames of position averaging
draw_debug_rectangles = True
sf = 0.5 # scale factor; downsample image to speed up feature detection
overlay_magic_sf = 1.5 # magic scale factor, determined empirically
use_classifier = "haarcascade" # or "lbpcascades"
overlay_enabled = True # can be changed through control_file
rectangle_proximity_factor = 0.5
rectangle_dedup_factor = 0.2
historic_colors = []
historic_colors.append((0,20,0))
historic_colors.append((0,70,0))
historic_colors.append((0,130,0))
historic_colors.append((30,200,30))
historic_colors.append((0,0,20))
historic_colors.append((0,0,70))
historic_colors.append((0,0,130))
historic_colors.append((30,30,200))

# class Cat:
#     def __init__(self, cropw, croph, imgw, imgh):
#         global moving_window_size
#         self.imgw = imgw
#         self.imgh = imgh
#         self.cropw = cropw
#         self.croph = croph
#         self.last_x, self.last_y = imgw/2, imgh/2
#         self.centerx, self.centery = self.last_x, self.last_y
#         self._moving_window_size = moving_window_size
#         self._index = 0
#         self._xarray = np.array([self.last_x] * self._moving_window_size)
#         self._yarray = np.array([self.last_y] * self._moving_window_size)

#     def update(self, x, y, w, h):
#         global sf # scale factor
#         self.last_x = (x + w/2)/sf
#         self.last_y = (y + h/2)/sf

#         self._xarray[self._index] = self.last_x
#         self._yarray[self._index] = self.last_y
#         self._index = self._index + 1
#         if self._index >= self._moving_window_size:
#             self._index = 0
        
#         self.centerx = np.sum(self._xarray) / self._moving_window_size
#         self.centery = np.sum(self._yarray) / self._moving_window_size
#         pass

#     # returns a x,y,w,h crop parameter based on the moving average
#     def getCrop(self):
#         pad_left = int(np.clip(self.centerx - self.cropw/2, 0, self.imgw - self.cropw))
#         pad_top  = int(np.clip(self.centery - self.croph/2, 0, self.imgh - self.croph))
#         return (pad_left, pad_top, self.cropw, self.croph)

#     # gives you the minimum distance of the cat frame from the param rectangle x,y,w,h
#     def getDistance(self, x, y, w, h):
#         dist = np.sqrt((self.centerx - (x+w/h)/sf)**2 + (self.centery - (y+h/2)/sf)**2)
#         return dist

class FaceTracker:
    def __init__(self):
        self.data = []
        pass

    def invalidate(self):
        for face in self.data:
            face.invalidate()

    def update(self, detectedFacesArray):
        if len(detectedFacesArray) == 0:
            return
        global sf
        global rectangle_proximity_factor

        rects = set()
        updatedFaces = 0
        for coords in detectedFacesArray:
            rectangle = Rectangle(coords)
            rectangle.multiply(1/sf)
            distances = []
            for face in self.data:
                d = rectangle.getDistance(face.getWeightedAverage())
                distances.append(d)
            if len(distances) > 0:
                closestFaceIdx = np.argmin(distances)
                closestFace = self.data[closestFaceIdx]
                closestFacePos, success = closestFace.getLastKnownPosition()
                drect = distances[closestFaceIdx]
                dmax = rectangle_proximity_factor * closestFacePos.getDiagonal()
                # print("drect: {}, drect_alt: {}, dmax: {}".format(drect, drect_alt, dmax))
                if success and drect <= dmax:
                    self.data[closestFaceIdx].update(rectangle)
                    updatedFaces = updatedFaces + 1
                else:
                    print("Closest rectangle is too far ({}) to be added. (max: {})".format(drect, dmax))
                    rects.add(rectangle)
            else:
                    rects.add(rectangle)
        
        if updatedFaces > 0:
            print("Updated {} already tracked faces of ({} total).".format(updatedFaces, len(self.data)))

        # remove duplicate entries in the detected array
        before_dedup =len(rects)
        for face in self.data:
            knownFacePos, _ = face.getLastKnownPosition()
            diagonal = knownFacePos.getDiagonal()
            rectangle_list = list(rects)
            for rectangle in rectangle_list:
                distance = knownFacePos.getDistance(rectangle)
                if distance < diagonal * rectangle_dedup_factor:
                    rects.discard(rectangle)
        totalDeduplicated = before_dedup - len(rects)
        if totalDeduplicated > 0:
            print("Deduplicated {} rectangles".format(totalDeduplicated))

        beforeAdding = len(self.data)
        # add new rectangles
        for newRect in rects:
            f = Face()
            f.create(newRect)
            self.data.append(f)
        addedFaces = len(self.data) - beforeAdding
        if addedFaces > 0:
            print("Added {} newly recognized face(s), collection now contains {} face(s).".format(addedFaces, len(self.data)))
        
    def purge(self):
        beforePurge = len(self.data)
        toBeDeleted = []
        for i in range(len(self.data)):
            face = self.data[i]
            if face.isRemoveable():
                toBeDeleted.append(face)
        self.data = list(set(self.data) - set(toBeDeleted))
        removed = beforePurge - len(self.data)
        if removed > 0:
            print("Purged {} faces from history; collection now has {} faces.".format(removed, len(self.data)))
        
    def composite(self, canvas, overlay, alpha):
        global draw_debug_rectangles
        for face in self.data:
            if draw_debug_rectangles:
                face.drawDebugFrames(canvas)
            face.drawOverlay(canvas, overlay, alpha)
        pass

class Face:
    def __init__(self):
        self._history_length = 4
        self.positions = [Rectangle()] * self._history_length
        self.valid = [False] * self._history_length
        self.age = 0
        self.lastIdx = self._history_length -1
        pass

    def create(self, rectangle):
        self.age = 1
        self.positions[self._history_length-1] = rectangle
        self.valid[self._history_length-1] = True

    def invalidate(self):
        self.age = self.age + 1
        for i in range(self._history_length - 1):
            self.valid[i] = self.valid[i+1]
            self.positions[i] = self.positions[i+1]
        self.valid[self._history_length-1] = False

    def isValid(self):
        NumberOfValid = 0
        for i in range(1, self._history_length):
            NumberOfValid = NumberOfValid + self.valid[i]
        return (NumberOfValid > ((self._history_length-1)/2))

    def update(self, rect):
        self.positions[self._history_length-1] = rect
        self.valid[self._history_length-1] = True

    def isRemoveable(self):
        if self.age >= self._history_length:
            return not self.isValid()
        return False

    def getLastKnownPosition(self):
        s = min(self.age, self._history_length)
        for i in range(s-1, 0, -1):
            if self.valid[i]:
                return self.positions[i], True
            # print("i={} is invalid...".format(i))
        return Rectangle(), False

    def getAveragedSize(self):
        size = 0
        count = 0
        for i in range(len(self.positions)):
            if self.valid[i]:
                count = count+1
                size = size + self.positions[i].getDiagonal()
        if count > 0:
            size = int(size / (count * np.sqrt(2)))
        return size, size

    def getWeightedAverage(self):
        average = Rectangle()
        count = 0
        # add the present rectangle one more time, to give it more weight
        for i in range(1, self._history_length):
            if self.valid[i]:
                count = count + 1
                average.add(self.positions[i])
        i = self._history_length-2
        if self.valid[i]:
            count = count + 1
            average.add(self.positions[i])

        if count > 0:
            average.multiply(1/count)
        else:
            print("can't return average, giving last known position")
            average, success = self.getLastKnownPosition()
            if not success:
                print("failed to get last known position! wtf?")
        return average
    
    def getPresentPosition(self):
        if self.valid[self.lastIdx-1]:
            return self.positions[self.lastIdx-1]
        else:
            return self.getWeightedAverage()

    def drawOverlay(self, img, animFrame, alpha):
        if self.isValid():
            avg = self.getWeightedAverage()
            x,y = avg.getCenterInt()
            w,h = self.getAveragedSize()
            coords = x,y,w,h
            apply_overlay(img, animFrame, alpha, coords)

    def drawDebugFrames(self, img):
        # color_purple = (100, 0, 100) # BGR
        # color_red = (0, 0, 200)
        # color_gray = (70, 70, 70)
        # if self.valid[self._history_length-2]:
        #     color = color_purple
        # else:
        #     color = color_red
        # if not self.isValid():
        #     color = color_gray
        # rect = self.getWeigthtedAverage()
        # rect.drawFrame(img, color)
        global historic_colors
        for i in range(self._history_length):
            c = historic_colors[i]
            if not self.isValid():
                c = historic_colors[i+4]
            r = self.positions[i]
            r.drawFrame(img, c, 1)

class Rectangle:
    def __init__(self, coords=(0,0,0,0)):
        x,y,w,h = coords
        self.x, self.y, self.w, self.h = x,y,w,h
    
    def getCenterInt(self):
        center = self.getCenter()
        x,y = center
        return (int(x), int(y))
    
    def getCenter(self):
        center = (self.x+self.w/2, self.y+self.h/2)
        return center
    
    def getDiagonal(self):
        return np.sqrt(self.w**2 + self.h**2)

    def getDistance(self, other_rectangle):
        ox, oy = other_rectangle.getCenter()
        sx, sy = self.getCenter()
        sd = self.getDiagonal()
        od = other_rectangle.getDiagonal()
        return np.sqrt((ox-sx)**2 + (oy-sy)**2) - sd/2 - od/2

    def add(self, other_rectangle):
        self.x, self.y = self.x + other_rectangle.x, self.y + other_rectangle.y
        self.w, self.h = self.w + other_rectangle.w, self.h + other_rectangle.h
    
    def multiply(self, mf):
        self.x, self.y = self.x * mf, self.y * mf
        self.w, self.h = self.w * mf, self.h * mf

    def getCorners(self):
        x1, x2 = self.x, self.x+self.w
        y1, y2 = self.y, self.y+self.h
        topleft = (int(x1), int(y1))
        bottomright = (int(x2), int(y2))
        return (topleft, bottomright)

    def drawFrame(self, canvas, color, thickness=2):            
        topleft, bottomright = self.getCorners()
        cv2.rectangle(canvas, topleft, bottomright, color, thickness)

def set_res(cap, x,y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def apply_overlay(bg, fg, alpha, coords):
    global overlay_magic_sf
    global overlay_enabled
    if not overlay_enabled:
        return
    x, y, w, h = coords
    fgh, fgw = fg.shape[:2]
    fg_across = np.sqrt(fgw**2 + fgh**2)
    bg_feature_across = np.sqrt(w**2 + h**2)
    fg_scale_factor = overlay_magic_sf * (bg_feature_across / fg_across)

    print("fg_scale_factor: {}".format(fg_scale_factor))
    scaled_fg = cv2.resize(fg, None, fx=fg_scale_factor, fy=fg_scale_factor, interpolation=cv2.INTER_CUBIC)
    scaled_alpha = cv2.resize(alpha, None, fx=fg_scale_factor, fy=fg_scale_factor, interpolation=cv2.INTER_CUBIC)

    sfgh, sfgw = scaled_fg.shape[:2]
    bgh, bgw = bg.shape[:2]

    # calculate virtual corners on the bg coordinate system:
    x1v, y1v = int(x-sfgw/2), int(y-sfgh/2)
    x2v, y2v = x1v+sfgw, y1v+sfgh
    #now x1v, y1v is the 0,0 of the fg (overlay)

    # clip virtual coordinates to the canvas
    x1, x2 = max(0, x1v), min(bgw, x2v)
    y1, y2 = max(0, y1v), min(bgh, y2v)

    # clip overlay coordinates
    x1o = max(0, -x1v)
    x2o = min(sfgw, bgw-x1v)
    y1o = max(0, -y1v)
    y2o = min(sfgh, bgh-y1v)

    #print(" --- ")
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        print("fg completely outside of image!")
        return
    # print("x: %i, y: %i, w: %i, h: %i" % (x,y,w,h))
    # print("x1v: %i, x2v: %i, y1v: %i, y2v: %i" % (x1v,x2v,y1v,y2v))
    # # print("fgw: %i, fgh: %i" % (fgh,fgw))
    # print("scale factor: %f" % fg_scale_factor)
    # print("sfgh: %i, sfgw: %i" % (sfgh,sfgw))
    # print("[y1:y2, x1:x2] = [%i:%i, %i:%i]" % (y1,y2,x1,x2))
    # print("[y1o:y2o, x1o:x2o] = [%i:%i, %i:%i]" % (y1o,y2o,x1o,x2o))

    alpha_crop = scaled_alpha[y1o:y2o, x1o:x2o]
    bg_crop = bg[y1:y2, x1:x2] * (1.0 - alpha_crop)
    fg_crop = scaled_fg[y1o:y2o, x1o:x2o] * alpha_crop
    bg[y1:y2, x1:x2] = bg_crop + fg_crop

if __name__ == '__main__':
    print("PID: %i" % os.getpid())

    lmAnim = imageio.mimread(anim_file)
    lmAlpha = cv2.imread(mask_file)
    alpha = lmAlpha.astype(float) / 255.0
    totalGifFrames = len(lmAnim)
    print("Found %i frames." % totalGifFrames)
    print("Control File: {}\nWrite \"1\" into the file in order to activate.".format(control_file))
    gifFrameIdx = 0
    open(control_file, "wt").write(str(int(overlay_enabled)))

    opencv_base_path = '/usr/share/opencv4/'
    if use_classifier == "haarcascade":
        face_cascade = cv2.CascadeClassifier(opencv_base_path + 'haarcascades/haarcascade_frontalface_default.xml')
        face_profile = cv2.CascadeClassifier(opencv_base_path + 'haarcascades/haarcascade_profileface.xml')
        #eye_cascade = cv2.CascadeClassifier(opencv_base_path + 'haarcascades/haarcascade_eye.xml')
    else:
        face_cascade = cv2.CascadeClassifier(opencv_base_path + 'lbpcascades/lbpcascade_frontalface_improved.xml')
        face_profile = cv2.CascadeClassifier(opencv_base_path + 'lbpcascades/lbpcascade_profileface.xml')

    cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_V4L2)
    #cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_FFMPEG)
    #cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_GSTREAMER)
    set_res(cap, target_imgw, target_imgh) # TODO add fallback options
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    # print("Capture format before asking: %s" % cap.get(cv2.CAP_PROP_FORMAT))
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    print("Capture format now: %i" % cap.get(cv2.CAP_PROP_FORMAT))
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'BGR3'))
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YV12'))

    ret, future = cap.read() # TODO ret is false if no image has been returned
    animFrame = cv2.cvtColor(lmAnim[0], cv2.COLOR_RGB2BGR)
    ft = FaceTracker()

    while(True):
        # Capture frame-by-frame

        # get rid of aged data
        ft.purge()

        ret, capFrame = cap.read()
        animFrame = cv2.cvtColor(lmAnim[gifFrameIdx], cv2.COLOR_RGB2BGR)
        if type(capFrame) == "NoneType":
            print("camera frame missing!")
            continue
        img = future.copy()
        future = capFrame.copy()
        resized = cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_LINEAR)
        grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        frontFaceArray = face_cascade.detectMultiScale(grayImg, 1.3, 6)
        profileFaceArray = face_profile.detectMultiScale(grayImg, 1.3, 6)

        # print("profileFaceArray: %s, frontFaceArray: %s" % (type(profileFaceArray), type(frontFaceArray)))
        if len(profileFaceArray) > 0:
            if len(frontFaceArray) > 0:
                faceArray = np.concatenate([frontFaceArray, profileFaceArray])
            else:
                faceArray = profileFaceArray
        else:
            faceArray = frontFaceArray

        ft.invalidate()
        ft.update(faceArray)
        ft.composite(img, animFrame, alpha)
            
        cv2.imshow('cat', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        gifFrameIdx = gifFrameIdx + 1
        if gifFrameIdx >= totalGifFrames:
            gifFrameIdx = 0

        overlay_enabled = (open(control_file, "r").read(1) == "1")

    cap.release()
    cv2.destroyAllWindows()
