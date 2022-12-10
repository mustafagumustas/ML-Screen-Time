import pickle
import numpy as np
import face_recognition
import cv2

# Load the SVM model from the pickle file
with open("face_recognation_svm.pk", "rb") as f:
    model = pickle.load(f)

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Loop over frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = webcam.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Use the face_recognition library to detect and encode the faces in the frame
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    # Convert the face encodings from 1D arrays to 2D arrays
    face_encodings = np.expand_dims(face_encodings, axis=1)

    # Loop over the detected faces
    for face_location, face_encoding in zip(face_locations, face_encodings):
        # Use the SVM model to label the face
        label = str(model.predict(face_encoding)[0])

        # Draw a rectangle around the face
        y2, x2, y1, x1 = face_location
        y2 *= 4
        x2 *= 4
        y1 *= 4
        x1 *= 4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Draw the label on the frame
        cv2.rectangle(frame, (x1, y1 - 35), (x2, y1), (0, 0, 255), cv2.FILLED)
        cv2.putText(
            frame,
            label,
            (x1 + 6, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    # Show the frame with labels
    cv2.imshow("Frame", frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) == ord("q"):
        break

# Release the webcam
webcam.release()
