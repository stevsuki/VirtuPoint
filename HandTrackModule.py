import cv2
import mediapipe as mp
import time

class handDetector() :
    def __init__(self, mode=False,
               maxHands=2,
               modelComplexity=1,
               detectionCon=0.5,
               trackingCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True, padding=0, flipHand=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        bboxList = []  # List to store bounding box coordinates
        typeHand = ""

        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                # Initialize coordinates
                xList = []
                yList = []
                
                # Iterate over all landmarks
                for lm in handLms.landmark:
                    h, w, c = img.shape  # Get image dimensions
                    cx, cy = int(lm.x * w), int(lm.y * h)  # Convert landmark coordinates to pixel values
                    xList.append(cx)
                    yList.append(cy)

                # Find bounding box coordinates
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)

                xmin = max(0, xmin - padding)  # Ensure the value doesn't go below 0
                ymin = max(0, ymin - padding)  # Ensure the value doesn't go below 0
                xmax = min(w, xmax + padding)  # Ensure the value doesn't exceed image width
                ymax = min(h, ymax + padding)  # Ensure the value doesn't exceed image height
                boxW = xmax - xmin
                boxH = ymax - ymin
                
                bbox = xmin, ymin, boxW, boxH  # Bounding box tuple (xmin, ymin, xmax, ymax)
                bboxList.append(bbox)  # Add bounding box to list

                if flipHand:
                    if handType.classification[0].label == "Left":
                        typeHand = "Right"
                    else:
                        typeHand = "Left"
                else:
                    typeHand = handType.classification[0].label
                if draw:
                    # Draw landmarks
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    
                    # Draw bounding box
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        return img, bboxList, typeHand  # Return image and bounding box list and type hand
        
    def fingersUp(self, lmlist, typeHand):
        fingers = []
        tipId = [8,12,16,20]
        if len(lmlist) != 0:

                if typeHand == "Right":
                    #right hand thumb id closed
                    if lmlist[4][1] > lmlist[4-1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    #left hand thumb id closed
                    if lmlist[4][1] < lmlist[4-1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                # 4 another closed id fingers
                for id in tipId:
                    if lmlist[id][2] < lmlist[id-2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

        return fingers
    def findPositions(self, img, handNo=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, landmark in enumerate(myHand.landmark):
                img_height, img_width, _ = img.shape
                x = int(landmark.x*img_width)
                y = int(landmark.y*img_height)
                lmList.append([id, x, y])
                if draw:
                    cv2.circle(img, center=(x,y), radius=10, color=(0, 255,255))
            
        return lmList

def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector(maxHands=1)

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPositions(img)

        if len(lmlist) != 0:
            print(lmlist[4])
       
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('Virtual Mouse', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
    
    