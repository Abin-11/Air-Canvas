# Air Canvas (Computer Visualization Project)

This project involves using computer vision and hand tracking to create an 
interactive "Air Canvas" that allows users to draw in the air with their finger. The 
process starts with capturing video input from the user's webcam, followed by 
detecting and tracking hand landmarks using MediaPipe. The detected hand 
positions are then used to draw on a virtual canvas in real-time, with options to 
change colors and clear the canvas.
The steps include initializing the webcam, processing each video frame to detect 
hand landmarks, determining the position of the forefinger and thumb, and using 
these positions to draw on the canvas. The drawing actions are controlled by 
recognizing specific gestures, such as touching the thumb and forefinger together 
to start a new line or moving the forefinger over designated areas to select colors 
or clear the canvas. This method effectively combines real-time video processing, 
hand gesture recognition, and interactive drawing to create an engaging user 
experience.
## Tools and technologies: OpenCV, MediaPipe, NumPy, Webcam
## Programming language: Python
