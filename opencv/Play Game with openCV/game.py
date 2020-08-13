## import opencv

import cv2
import numpy as np
from start import PressKey ,ReleaseKey 
import imutils

# https://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
# PressKey and ReleaseKey code was based on this stackoverflow post

# https://wiki.nexusmods.com/index.php/DirectX_Scancodes_And_How_To_Use_Them
# above link will give you the code of other keys, such that you can design it according to your application
left = 205
right = 199

# strat capturing video via webcam

cap = cv2.VideoCapture(0)

# hsv range of your color to be detected
Lower = np.array([50, 70, 0])
Upper = np.array([180,255,253])

current_key_pressed = set()

while True:
    
    keyPressed = False
    keyPressed_lr = False
    
    _, frame = cap.read()
    frame = cv2.flip(frame,1)  # change according to your device, mine flips the frame automatically
    
    # resize frame
    h, w = frame.shape[:2]
    ar = w/h
    frame = cv2.resize(frame, (int(ar*400),400))
    img = frame.copy()   
    
    #storing height and width in varibles 
    height = frame.shape[0]
    width = frame.shape[1]
    
    # vlur and convert to hsv
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # creating mask and reducing noise
    mask = cv2.inRange(hsv, Lower,Upper)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    #up_mask = mask[0:height//2,0:width,]
    #down_mask = mask[height//2:height,width//4:3*width//4,]
    
    cnts_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts_ = imutils.grab_contours(cnts_)
    
    if len(cnts_) > 0:
        c = max(cnts_, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center_ = (int(M["m10"] / (M["m00"]+0.000001)), int(M["m01"] / (M["m00"]+0.000001)))
        if radius >30:
            # draw the circle and centroid on the frame,
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, center_, 5, (0, 0, 255), -1)
             
                #the window size is kept 160 pixels in the center of the frame(80 pixels above the center and 80 below)
            if center_[0] < (width//2 - 10):
                cv2.putText(frame,'LEFT',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),3)
                PressKey(left)
                current_key_pressed.add(left)
                keyPressed = True
                keyPressed_lr = True
            elif center_[0] > (width//2 + 10):
                cv2.putText(frame,'RIGHT',(400,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),3)
                PressKey(right)
                current_key_pressed.add(right)
                keyPressed = True
                keyPressed_lr = True
    
     # show the frame to our screen
    frame_copy = frame.copy()
    
    #draw box for left 
    frame_copy = cv2.rectangle(frame_copy,(0,2),(width//2- 10,height-2 ),(255,255,255),1)
    
    #draw box for right
    frame_copy = cv2.rectangle(frame_copy,(width//2 +10,0),(width-2,height-2 ),(255,255,255),1)

    #display final frame    
    cv2.imshow("Frame", frame_copy)
    
    #We need to release the pressed key if none of the key is pressed else the program will keep on sending
    # the presskey command 
    if not keyPressed and len(current_key_pressed) != 0:
        for key in current_key_pressed:
            ReleaseKey(key)
        current_key_pressed = set()
    
    #cv2.imshow('Frame', frame)
    k = cv2.waitKey(1) & 0xFF
    if k== ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()