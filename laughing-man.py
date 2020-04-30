#!/usr/bin/env python3

'''
I just hope this will deter all corporate fucks from using my crap.

Shitty Face Tracker Copyright 12020 Zoé Bőle / Fully Automated (“Licensor”)

Hippocratic License Version Number: 2.1.

Purpose. The purpose of this License is for the Licensor named above to permit 
the Licensee (as defined below) broad permission, if consistent with Human 
Rights Laws and Human Rights Principles (as each is defined below), to use and 
work with the Software (as defined below) within the full scope of Licensor’s 
copyright and patent rights, if any, in the Software, while ensuring attribution 
and protecting the Licensor from liability.

Permission and Conditions. The Licensor grants permission by this license 
(“License”), free of charge, to the extent of Licensor’s rights under applicable 
copyright and patent law, to any person or entity (the “Licensee”) obtaining a 
copy of this software and associated documentation files (the “Software”), to do 
everything with the Software that would otherwise infringe (i) the Licensor’s 
copyright in the Software or (ii) any patent claims to the Software that the 
Licensor can license or becomes able to license, subject to all of the following 
terms and conditions:

    Acceptance. This License is automatically offered to every person and entity 
subject to its terms and conditions. Licensee accepts this License and agrees to 
its terms and conditions by taking any action with the Software that, absent 
this License, would infringe any intellectual property right held by Licensor.

    Notice. Licensee must ensure that everyone who gets a copy of any part of 
this Software from Licensee, with or without changes, also receives the License 
and the above copyright notice (and if included by the Licensor, patent, 
trademark and attribution notice). Licensee must cause any modified versions of 
the Software to carry prominent notices stating that Licensee changed the 
Software. For clarity, although Licensee is free to create modifications of the 
Software and distribute only the modified portion created by Licensee with 
additional or different terms, the portion of the Software not modified must be 
distributed pursuant to this License. If anyone notifies Licensee in writing 
that Licensee has not complied with this Notice section, Licensee can keep this 
License by taking all practical steps to comply within 30 days after the notice. 
If Licensee does not do so, Licensee’s License (and all rights licensed 
hereunder) shall end immediately.

    Compliance with Human Rights Principles and Human Rights Laws.

        Human Rights Principles.

        (a) Licensee is advised to consult the articles of the United Nations 
Universal Declaration of Human Rights and the United Nations Global Compact that 
define recognized principles of international human rights (the “Human Rights 
Principles”). Licensee shall use the Software in a manner consistent with Human 
Rights Principles.

        (b) Unless the Licensor and Licensee agree otherwise, any dispute, 
controversy, or claim arising out of or relating to (i) Section 1(a) regarding 
Human Rights Principles, including the breach of Section 1(a), termination of 
this License for breach of the Human Rights Principles, or invalidity of Section 
1(a) or (ii) a determination of whether any Law is consistent or in conflict 
with Human Rights Principles pursuant to Section 2, below, shall be settled by 
arbitration in accordance with the Hague Rules on Business and Human Rights 
Arbitration (the “Rules”); provided, however, that Licensee may elect not to 
participate in such arbitration, in which event this License (and all rights 
licensed hereunder) shall end immediately. The number of arbitrators shall be 
one unless the Rules require otherwise.

        Unless both the Licensor and Licensee agree to the contrary: (1) All 
documents and information concerning the arbitration shall be public and may be 
disclosed by any party; (2) The repository referred to under Article 43 of the 
Rules shall make available to the public in a timely manner all documents 
concerning the arbitration which are communicated to it, including all 
submissions of the parties, all evidence admitted into the record of the 
proceedings, all transcripts or other recordings of hearings and all orders, 
decisions and awards of the arbitral tribunal, subject only to the arbitral 
tribunal’s powers to take such measures as may be necessary to safeguard the 
integrity of the arbitral process pursuant to Articles 18, 33, 41 and 42 of the 
Rules; and (3) Article 26(6) of the Rules shall not apply.

        Human Rights Laws. The Software shall not be used by any person or 
entity for any systems, activities, or other uses that violate any Human Rights 
Laws. “Human Rights Laws” means any applicable laws, regulations, or rules 
(collectively, “Laws”) that protect human, civil, labor, privacy, political, 
environmental, security, economic, due process, or similar rights; provided, 
however, that such Laws are consistent and not in conflict with Human Rights 
Principles (a dispute over the consistency or a conflict between Laws and Human 
Rights Principles shall be determined by arbitration as stated above). Where the 
Human Rights Laws of more than one jurisdiction are applicable or in conflict 
with respect to the use of the Software, the Human Rights Laws that are most 
protective of the individuals or groups harmed shall apply.

        Indemnity. Licensee shall hold harmless and indemnify Licensor (and any 
other contributor) against all losses, damages, liabilities, deficiencies, 
claims, actions, judgments, settlements, interest, awards, penalties, fines, 
costs, or expenses of whatever kind, including Licensor’s reasonable attorneys’ 
fees, arising out of or relating to Licensee’s use of the Software in violation 
of Human Rights Laws or Human Rights Principles.

    Failure to Comply. Any failure of Licensee to act according to the terms and 
conditions of this License is both a breach of the License and an infringement 
of the intellectual property rights of the Licensor (subject to exceptions under 
Laws, e.g., fair use). In the event of a breach or infringement, the terms and 
conditions of this License may be enforced by Licensor under the Laws of any 
jurisdiction to which Licensee is subject. Licensee also agrees that the 
Licensor may enforce the terms and conditions of this License against Licensee 
through specific performance (or similar remedy under Laws) to the extent 
permitted by Laws. For clarity, except in the event of a breach of this License, 
infringement, or as otherwise stated in this License, Licensor may not terminate 
this License with Licensee.

    Enforceability and Interpretation. If any term or provision of this License 
is determined to be invalid, illegal, or unenforceable by a court of competent 
jurisdiction, then such invalidity, illegality, or unenforceability shall not 
affect any other term or provision of this License or invalidate or render 
unenforceable such term or provision in any other jurisdiction; provided, 
however, subject to a court modification pursuant to the immediately following 
sentence, if any term or provision of this License pertaining to Human Rights 
Laws or Human Rights Principles is deemed invalid, illegal, or unenforceable 
against Licensee by a court of competent jurisdiction, all rights in the 
Software granted to Licensee shall be deemed null and void as between Licensor 
and Licensee. Upon a determination that any term or provision is invalid, 
illegal, or unenforceable, to the extent permitted by Laws, the court may modify 
this License to affect the original purpose that the Software be used in 
compliance with Human Rights Principles and Human Rights Laws as closely as 
possible. The language in this License shall be interpreted as to its fair 
meaning and not strictly for or against any party.

    Disclaimer. TO THE FULL EXTENT ALLOWED BY LAW, THIS SOFTWARE COMES “AS IS,” 
WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED, AND LICENSOR AND ANY OTHER CONTRIBUTOR 
SHALL NOT BE LIABLE TO ANYONE FOR ANY DAMAGES OR OTHER LIABILITY ARISING FROM, 
OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THIS LICENSE, UNDER ANY KIND OF 
LEGAL CLAIM.

This Hippocratic License is an Ethical Source license 
(https://ethicalsource.dev) and is offered for use by licensors and licensees at 
their own risk, on an “AS IS” basis, and with no warranties express or implied, 
to the maximum extent permitted by Laws.

'''

import numpy as np
import cv2
import imageio
import os
import argparse
# fcntl and v4l2 are imported later after arguments have been parsed

control_file = "/tmp/lm-ctrl"
anim_file = "lm-anim.gif"
mask_file = "lm-mask.png"

'''
You might need to tweak these below.
'''

loopbackDevName = '/dev/video2'

# try to get this resolution from the camera
target_imgw, target_imgh = 1280, 720 
#target_imgw, target_imgh = 1920, 1080

# try to get this fps from the camera
target_fps = 30

# the size of the face is averaged over a longer time period than its position
# here you can set the size of the moving window
rectangle_size_averaging_window = 7 # frames of size averaging

# scale factor is used to downsample camera image for face detection
# this only affects detection accuracy and speed, won't infuence the output image
sf = 0.5

# this scale factor determines the ratio between the overlay animation and the
# detected face's diagonal. determined empirically.
overlay_magic_sf = 1.5

# this sets the default behavior for the overlay
overlay_enabled = False # can be changed through control_file

# this controls how far maximum will a rectangle still considered to belong to a known face
# expressed as a ratio to the detected rectangle's diagonal
rectangle_proximity_factor = 0.3 

# this controls how far a new, unassigned rectangle must be from a known face, otherwise it
# is rejected. a ratio to the face's diagonal
rectangle_dedup_factor = 0.1

# which opencv face detector to use
use_classifier = "haarcascade" # or "lbpcascades"

# show rectangles around the tracked faces
draw_debug_rectangles = False
# debugging rectangle colors
historic_colors = []
historic_colors.append((0,20,0))
historic_colors.append((0,70,0))
historic_colors.append((0,130,0))
historic_colors.append((30,200,30))
historic_colors.append((0,0,20))
historic_colors.append((0,0,70))
historic_colors.append((0,0,130))
historic_colors.append((30,30,200))

class FaceTracker:
    def __init__(self):
        self.data = []
        pass

    def invalidate(self):
        for face in self.data:
            face.invalidate()

    def update(self, detectedFacesArray, canvas):
        if len(detectedFacesArray) == 0:
            return
        global sf
        global rectangle_proximity_factor

        yellow = (0, 230, 220) # BGR, RGB(220,230,0)
        rects = set()
        updatedFaces = 0
        for coords in detectedFacesArray:
            rectangle = Rectangle(coords)
            rectangle.multiply(1/sf)
            rectangle.drawFrame(canvas, yellow, 2)
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
        global rectangle_size_averaging_window
        self._history_length = 4
        self.positions = [Rectangle()] * self._history_length
        self.valid = [False] * self._history_length
        self.age = 0
        self.lastIdx = self._history_length -1
        self.diagonals = [0] * rectangle_size_averaging_window
        self._diagonal_avg_idx = 0
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
        permissive = False
        if self.age > self._history_length:
            permissive = True
        return (permissive and NumberOfValid > 1) or (NumberOfValid > ((self._history_length-1)/2))

    def update(self, rect):
        self.positions[self._history_length-1] = rect
        self.valid[self._history_length-1] = True
        self.diagonals[self._diagonal_avg_idx] = rect.getDiagonal()
        self._diagonal_avg_idx = self._diagonal_avg_idx + 1
        if self._diagonal_avg_idx >= len(self.diagonals):
            self._diagonal_avg_idx = 0

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
        maxidx = max(min(len(self.diagonals), self.age) - 2, 0)
        size = np.sum(self.diagonals[0:maxidx]) / maxidx
        size = int(size / 1.4142)
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
            avg = self.getPresentPosition()
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
        self.diagonal = 0
    
    def getCenterInt(self):
        center = self.getCenter()
        x,y = center
        return (int(x), int(y))
    
    def getCenter(self):
        center = (self.x+self.w/2, self.y+self.h/2)
        return center
    
    def getDiagonal(self):
        self.diagonal = np.sqrt(self.w**2 + self.h**2)
        return self.diagonal

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

    #print("fg_scale_factor: {}".format(fg_scale_factor))
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

    parser = argparse.ArgumentParser(description='Shitty Face Tracker')
    parser.add_argument('--gui', action='store_true', help='Show the video output in a window.')
    parser.add_argument('--noloopback', action='store_true', help='Don\'t create a loopback device.')
    parser.add_argument('--debug', action='store_true', help='Draw debugging rectangles on the video.')
    parser.add_argument('--overlay_on', action='store_true', help='Show the overlay animation by default.')
    parser.add_argument('--output_bgr', action='store_true', help='Output BGR24 pixel format instead of RGB24 on the loopback. OBS wants this!')

    args = parser.parse_args()
    if not args.noloopback:
        import fcntl
        import v4l2

        if not os.path.exists(loopbackDevName):
            print ("Error: device {} does not exist".format(loopbackDevName))
            exit(1)
        device = open(loopbackDevName, 'wb')

    draw_debug_rectangles = args.debug
    overlay_enabled = args.overlay_on

    print("PID: %i" % os.getpid())

    lmAnim = imageio.mimread(anim_file)
    lmAlpha = cv2.imread(mask_file)
    alpha = lmAlpha.astype(float) / 255.0
    totalGifFrames = len(lmAnim)
    print("Found {} frames in the animation.".format(totalGifFrames))
    print("Control File: {}\nWrite \"1\" into the file in order to activate.\n".format(control_file))
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
    
    if not args.noloopback:
        o_height, o_width, channels = future.shape
        loformat                      = v4l2.v4l2_format()
        loformat.type                 = v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT
        loformat.fmt.pix.field        = v4l2.V4L2_FIELD_NONE
        if args.output_bgr:
            loformat.fmt.pix.pixelformat  = v4l2.V4L2_PIX_FMT_BGR24
        else:
            loformat.fmt.pix.pixelformat  = v4l2.V4L2_PIX_FMT_RGB24
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
        ft.update(faceArray, img)
        ft.composite(img, animFrame, alpha)
        
        if not args.noloopback:
            if args.output_bgr:
                device.write(img)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                device.write(rgb)
        
        if args.gui:
            cv2.imshow('cat', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        gifFrameIdx = gifFrameIdx + 1
        if gifFrameIdx >= totalGifFrames:
            gifFrameIdx = 0

        overlay_enabled = (open(control_file, "r").read(1) == "1")

    cap.release()
    cv2.destroyAllWindows()
