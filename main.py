import cvzone
from cvzone.HandTrackingModule import HandDetector
import cv2

# Initialize the webcam to capture video
# The '0' indicates the built-in camera; change if needed
cap = cv2.VideoCapture(0)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(maxHands=1, detectionCon=0.5)

# Initialize the SerialObject for communication

# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    # Find hands in the current frame
    hands, img = detector.findHands(img, draw=True)

    # Check if any hands are detected
    if hands:
        # Get the first hand detected
        hand = hands[0]
        lmList = hand["lmList"]  # List of 21 landmarks
        bbox = hand["bbox"]  # Bounding box

        # Count the number of fingers up
        fingers = detector.fingersUp(hand)
        print(fingers)

        # Send data through the serial port

    # Display the image in a window
    cv2.imshow("Image", img)
    cv2.waitKey(1)
