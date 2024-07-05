import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
import numpy as np

def main():
    # Initialize the HandDetector class with the given parameters
    detector = HandDetector(maxHands=1, detectionCon=0.5)

    st.title("Hand Gesture Control")

    # Initialize the webcam to capture video
    cap = cv2.VideoCapture(0)

    # Create a placeholder for the video
    video_placeholder = st.empty()

    # Continuously get frames from the webcam
    while True:
        # Capture each frame from the webcam
        success, frame = cap.read()
        if not success:
            st.error("Failed to capture image")
            break

        # Find hands in the current frame
        hands, _ = detector.findHands(frame, draw=False)

        # Check if any hands are detected
        if hands:
            # Get the first hand detected
            hand = hands[0]

            # Count the number of fingers up
            fingers = detector.fingersUp(hand)

        # Display the video feed in a Streamlit window
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        video_placeholder.image(frame, channels="RGB")

if __name__ == "__main__":
    main()