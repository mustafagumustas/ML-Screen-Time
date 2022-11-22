from vidgear.gears import CamGear
import cv2
import pickle
import os

# face_cascade = cv2.CascadeClassifier(
#     "SeriesFaces/cascades/data/haarcascade_frontalface_alt2.xml"
# )
# profil_cascade = cv2.CascadeClassifier(
#     "SeriesFaces/cascades/data/haarcascade_profileface.xml"
# )
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read("trainner.yml")

weights = os.path.join("face_detection_yunet_2022mar.onnx")
face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))

labels = {"person_name": 1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

stream = CamGear(
    source="https://www.youtube.com/watch?v=BDb__vr8eao", stream_mode=True, logging=True
).start()  # YouTube Video URL as input

unknown_image_count = 0


# infinite loop
while True:
    frame = stream.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width, _ = frame.shape
    face_detector.setInputSize((width, height))

    _, faces = face_detector.detect(frame)
    faces = faces if faces is not None else []
    # location of recognised faces are x,y,w,h
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
        cv2.putText(
            frame, confidence, position, font, scale, color, thickness, cv2.LINE_AA
        )

        # img_item = f"image_{unknown_image_count}.png"
        # unknown_image_count += 1
        # cv2.imwrite(img_item, roi_gray)

    # profil = profil_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # for fx, fy, fw, fh in profil:
    #     roi_gray = gray[fy : fy + fh, fx : fx + fw]
    #     id_, conf = recognizer.predict(roi_gray)
    #     if conf >= 45 and conf <= 85:
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         name = labels[id_]
    #         color = (255, 255, 255)
    #         stroke = 2
    #         cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
    #     cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
    #     img_item = f"image_{unknown_image_count}.png"
    #     unknown_image_count += 1
    #     # cv2.imwrite(img_item, roi_gray)
    # print(unknown_image_count)

    if frame is None:
        break

    cv2.imshow("Output Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    # check for 'q' key-press
    if key == ord("q"):
        # if 'q' key-pressed break out
        break

cv2.destroyAllWindows()
# close output window

# safely close video stream.
stream.stop()
