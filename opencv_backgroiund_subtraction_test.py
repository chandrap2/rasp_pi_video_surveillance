from picamera.array import PiRGBArray
from picamera import PiCamera
from datetime import datetime
from twilio.rest import Client
import cv2, time, numpy

"""
client = Client("secretcode1", "secretcode2")
client.messages.create(to = "admin's phone number", from_ = "twilio service number", body = "python program started")
notification_start_time = datetime.now()
notification_period = True
"""

cam = PiCamera()
cam.framerate = 30
cam.resolution = (480, 480)

#waiting for auto gain control to settle
time.sleep(1)

cam.exposure_mode = 'off'
time.sleep(3)
cam.contrast = 0

raw_capture = PiRGBArray(cam, size = (480, 480))

bck_subtractor = cv2.BackgroundSubtractorMOG2(history = 25, varThreshold = 10)
erode_kernel = numpy.ones((5, 5), numpy.uint8)
dilate_kernel = numpy.ones((7, 7), numpy.uint8)

print "setup complete"

for frame in cam.capture_continuous(raw_capture, format = "bgr", use_video_port = True):
    """
    img -> foreground mask -> binarize -> noise erosion -> dilation to combining close blobs ->
    draw bounding rectangles around large blobs
    """ 
    img = frame.array
    #print "original: " + str(img)
    print type(img)
    fgmask = bck_subtractor.apply(img, learningRate = 0.00001)
    thresh, fgmask_binarized = cv2.threshold(fgmask, 240, 255, cv2.THRESH_BINARY)
    erosion = cv2.erode(fgmask_binarized, erode_kernel, iterations = 1)
    dilation = cv2.dilate(erosion, dilate_kernel, iterations = 1)
    #print "dilation: " + str(dilation)
    print type(dilation)
    cv2.imshow("binarized erosion -> dilation", dilation)
    
    contours, heirarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)          

    # mark countours greater than minimum area onto original image
    valid_contour_count = 0
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])

        if (w * h > 300):
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            valid_contour_count += 1

    """
    #text notification logic
    if (valid_contour_count > 0):
        if notification_period:
            client.messages.create(to = "admin's phone number", from_ = "+18312631390", body = (str(valid_contour_count) + " objects detected, no notifications for 10 secs"))
            notification_start_time = datetime.now()
            notification_period = False
        else:
            notification_period = (datetime.now() - notification_start_time).seconds >= 10
    """
    
    cv2.imshow("original", img)
    #cv2.imshow("foreground mask", fgmask)
    cv2.imshow("foreground mask binarized", fgmask_binarized)
    #cv2.imshow("binarized erosion", erosion)

    key = cv2.waitKey(1) & 0xff

    raw_capture.truncate(0)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
