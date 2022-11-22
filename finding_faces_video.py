from vidgear.gears import CamGear
import cv2
import pickle
import os
import numpy as np

directory = os.path.dirname(__file__)

# face_cascade = cv2.CascadeClassifier(
#     "SeriesFaces/cascades/data/haarcascade_frontalface_alt2.xml"
# )
# profil_cascade = cv2.CascadeClassifier(
#     "SeriesFaces/cascades/data/haarcascade_profileface.xml"
# )
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

weights = os.path.join("face_detection_yunet_2022mar.onnx")
face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))

labels = {"person_name": 1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}


# cap = cv2.VideoCapture("/Users/mustafagumustas/TFOD/videoplayback480.mp4")
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video stream or file")

unknown_image_count = 0

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        height, width, _ = frame.shape
        face_detector.setInputSize((width, height))

        _, faces = face_detector.detect(frame)
        faces = faces if faces is not None else []

        for face in faces:
            box = list(map(int, face[:4]))
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(frame, box, color, thickness, cv2.LINE_AA)

            confidence = face[-1]
            confidence = "{:.2f}".format(confidence)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 2
            # cv2.putText(
            #     frame, confidence, position, font, scale, color, thickness, cv2.LINE_AA
            # )
            # landmarks = list(map(int, face[4 : len(face) - 1]))
            # landmarks = np.array_split(landmarks, len(landmarks) / 2)
            # for landmark in landmarks:
            #     radius = 5
            #     thickness = -1
            #     cv2.circle(frame, landmark, radius, color, thickness, cv2.LINE_AA)

            # recognition
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
            id_, conf = recognizer.predict(roi_gray)
            print(labels[id_], conf)
            if conf >= 60:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(
                    frame, name, (box[0], box[1]), font, 1, color, stroke, cv2.LINE_AA
                )
            # elif conf < 50:
            #     img_item = f"image_{unknown_image_count}.png"
            #     unknown_image_count += 1
            #     cv2.imwrite(img_item, roi_gray)

        # turing frame into gray
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        # # location of recognised faces are x,y,w,h
        # for x, y, w, h in faces:
        #     roi_gray = gray[y : y + h, x : x + w]
        #     id_, conf = recognizer.predict(roi_gray)
        #     if conf >= 45 and conf <= 85:
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         name = labels[id_]
        #         color = (255, 255, 255)
        #         stroke = 2
        #         cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
        #     # img_item = f"image_{unknown_image_count}.png"
        #     # unknown_image_count += 1
        #     # cv2.imwrite(img_item, roi_gray)

        #     # rectangle around faces
        #     color = (255, 0, 0)
        #     stroke = 2
        #     end_cord_x = x + w
        #     end_cord_y = y + h
        #     cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
        cv2.imshow("Frame", frame)

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
