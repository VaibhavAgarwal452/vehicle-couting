import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('video.mp4')
vehicle = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
object_detector = cv2.createBackgroundSubtractorKNN()
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    height, width, _ = frame.shape
    if ret == True:
        out.write(frame)

        mask = object_detector.apply(frame)
        cv2.line(frame, (200, 450), (550, 450), (0, 0, 255), 2)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:

            area = cv2.contourArea(cnt)
            if area > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                # cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                xMid = int((x + (x+w))/2)
                yMid = int((y + (y+h))/2)
                cv2.circle(frame, (xMid, yMid), 5, (0, 0, 255), 2)
                if(446 < yMid < 454):
                    print(xMid, yMid)
                    vehicle += 1
        # Display the resulting frame
        cv2.putText(frame, 'Total Vehicles : {}'.format(vehicle),
                    (450, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        cv2.imshow('Frame', frame)

        # cv2.imshow('Mask', mask)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()
