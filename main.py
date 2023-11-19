import cv2
import numpy as np
import view

FILE = "18-nov/18nov-libre plastique-3g0-2aimants.avi"

cap = cv2.VideoCapture(FILE)

# 1.AVI [50, 50, 240] +/- 60
#  254, 164, 146
TARGET = np.array([146, 164, 254])
THRESHOLD = 30
lower_red = TARGET - THRESHOLD
upper_red = TARGET + THRESHOLD

# lower_red = np.array([130, 115, 200])
# upper_red = np.array([180, 200, 255])

file = open(FILE+".csv", "w")
file.write("t,x,y\n")

fps = cap.get(cv2.CAP_PROP_FPS)
duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

print("FPS: " + str(fps))

previewShown = True
frameIndex = 0
while True:
    ret, frame = cap.read()
    if not ret: break

    # crop frame
    # frame = frame[int(frame.shape[1] * 0.05):int(frame.shape[1] * 0.5), int(frame.shape[1] * 0.3):int(frame.shape[1] * 0.75)]
    # frame = cv2.resize(frame, (int(frame.shape[1] * 0.75), frame.shape[0]))

    if(frameIndex==0):
        cv2.imwrite("out.png", frame)
    t = frameIndex/fps

    mask = cv2.inRange(frame, lower_red, upper_red)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    result = frame & mask_rgb

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(sorted_contours) > 0:
        contour = sorted_contours[0]

        area = cv2.contourArea(contour)
        cv2.drawContours(frame, [contour], 0, (0, 0, 0), 1)

        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            # Draw a circle at the center
            # cv2.drawMarker(frame, (cx, cy), 5, (255, 0, 0), -1)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 0, 0), 1)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 0, 0), 1)
            
            # print(f"{t},{cx},{cy}")
            file.write(f"{t},{cx},{cy}\n")


    # Display the resulting frame

    if(previewShown):
        # cv2.imshow('Red Object Tracking', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Image', frame)

        key = cv2.waitKey(100)

        if key & 0xFF == ord('q'):
            exit(0)
        elif key & 0xFF == ord('h'):
            previewShown = False
    
    if(frameIndex%50==0):
        print(f"{round(t)}s / {round(duration)}s")

    frameIndex+=1    

cap.release()
cv2.destroyAllWindows()
file.close()

view.viewData(FILE)