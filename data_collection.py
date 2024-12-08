import cv2
import HandTrackModule as htm
import numpy as np
import time

cap = cv2.VideoCapture("http://192.168.1.8:8080/video")
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 1024, 768)
detector = htm.handDetector(maxHands=1)

img_size = 300

counterMovingCursor = 0
counterMouseClick = 0
counterDragMouse = 0
counterControlVolume = 0
counterRightClick = 0

folderDataMovingCursor = "datasets/data_training/moving_cursor"
folderDataMouseClick = "datasets/data_training/mouse_click"
folderDataDragMouse = "datasets/data_training/drag_mouse"
folderDataControlVolume = "datasets/data_training/control_volume"
folderDataRightClick = "datasets/data_training/right_click"

while True:

    success, img = cap.read()
    _, bbox, _ = detector.findHands(img, padding=80, draw=False)

    if bbox != []:
        x, y, w, h = bbox[0] #extract bounding box info

        imgWhite = np.ones((img_size, img_size, 3), np.uint8)*255
        imgCrop = img[y:y+h, x:x+w]

        aspectRatio = h / w

        if aspectRatio > 1:
            # Height is greater than width
            k = img_size / h
            wCal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, img_size))
            wGap = (img_size - wCal) // 2
            imgWhite[:, wGap:wGap+wCal] = imgResize
        else:
            # Width is greater than height
            k = img_size / w
            hCal = int(k * h)
            imgResize = cv2.resize(imgCrop, (img_size, hCal))
            hGap = (img_size - hCal) // 2
            imgWhite[hGap:hGap+hCal, :] = imgResize

        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("m"):
        counterMovingCursor += 1
        cv2.imwrite(f"{folderDataMovingCursor}/Image_{time.time()}.jpg", imgWhite)
        print("Data snapshot moving cursor: ", counterMovingCursor)
    elif key == ord("c"):
        counterMouseClick += 1
        cv2.imwrite(f"{folderDataMouseClick}/Image_{time.time()}.jpg", imgWhite)
        print("Data snapshot mouse click: ", counterMouseClick)
    elif key == ord("d"):
        counterDragMouse += 1
        cv2.imwrite(f"{folderDataDragMouse}/Image_{time.time()}.jpg", imgWhite)
        print("Data snapshot drag mouse: ", counterDragMouse)
    elif key == ord("v"):
        counterControlVolume += 1
        cv2.imwrite(f"{folderDataControlVolume}/Image_{time.time()}.jpg", imgWhite)
        print("Data snapshot control volume: ", counterControlVolume)
    elif key == ord("r"):
        counterRightClick += 1
        cv2.imwrite(f"{folderDataRightClick}/Image_{time.time()}.jpg", imgWhite)
        print("Data snapshot control volume: ", counterRightClick)