import cv2

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

# Set the window size
window_size = (200, 200)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # If no faces are detected, display a 200x200 pixel window of the original camera input
    if len(faces) == 0:
        resized_frame = cv2.resize(frame, window_size)
        cv2.imshow('Zoomed Face', resized_frame)
    else:
        # Zoom into the face area and display only that frame
        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI)
            roi = frame[y:y+h, x:x+w]

            # Resize the ROI to the fixed window size (200x200)
            resized_roi = cv2.resize(roi, window_size)

            # Display the zoomed-in face
            cv2.imshow('Zoomed Face', resized_roi)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
