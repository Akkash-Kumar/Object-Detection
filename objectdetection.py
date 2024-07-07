import cv2   #opencv library

import imutils   #imutils library

camera = cv2.VideoCapture(0)   #initialize primary camera

firstFrame = None    #initialize first frame as none

area = 500    #initialize area to detect ( above area 500 will be detected)


while True:             #infinite loop to run camera continuously

    _ , img = camera.read()    #read frame from live camera

    text = "No new object detected"      #initilize text

    resizedImg = imutils.resize(img,width = 1000)    #resize image

    grayImg = cv2.cvtColor(resizedImg , cv2.COLOR_BGR2GRAY)    #convert to grayscale image

    blurImg = cv2.GaussianBlur(grayImg , (21,21) , 0)    # to smooth image

    if firstFrame is None:

        firstFrame = blurImg    #capturing first frame
        continue

    imgDiff = cv2.absdiff(firstFrame , blurImg)   #Find absolute difference between first frame and next coming frames

    blackwhiteImg = cv2.threshold(imgDiff, 25,255,cv2.THRESH_BINARY)[1]  #convert to black&white pic

    blackwhiteImg = cv2.dilate(blackwhiteImg , None,iterations =2)   #remove left overs(get clear bw pic)

    conts = cv2.findContours(blackwhiteImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   #find contours(borders)

    conts = imutils.grab_contours(conts)  #grab all contours found

    for c in conts:

        if cv2.contourArea(c) < area:    #check contour area with area we initialised
            continue                      #skip that contour

        (x,y,w,h) = cv2.boundingRect(c)   #get all coordinates to draw rectangle from contour

        cv2.rectangle(resizedImg , (x,y),(x+w,y+h),(0,255,0),2)   #draw rectangle

        text = "New object detected"

    print(text)

    cv2.putText(resizedImg , text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2)  #put text in image

    cv2.imshow("ObjectDetection",resizedImg)   #display image

    key = cv2.waitKey(10)   #wait for 10 frames

    if key == ord("a"):   #if a-key is pressed, camera will close
        break

camera.release()   #release camera
cv2.destroyAllWindows()  #close window

        

    
