from face_normalisation import *
from train_model import *
import cv2
cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('P:\\GIT_FILE\\Face_Recognition\\OpenCVDemo\\haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('P:\\GIT_FILE\\Face_Recognition\\OpenCVDemo\\haarcascade_eye.xml')


def get_face():
    print("Hit 'Q' to take photo")
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coord = faceCascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in face_coord:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

            eyes_coord = eyeCascade.detectMultiScale(gray)
            for (ex, ey, ew, eh) in eyes_coord:
                if ex > x and ey > y and ex + ew < x + w and ey + eh < y + h:
                    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q') and len(face_coord) > 0:
            if len(face_coord) > 1:
                print("Multiple faces detected!")
            else:
                cap.release()
                cv2.destroyAllWindows()
                return get_normalised_faces(gray, face_coord)[0]


image, labels, labels_dict = collect_dataset()
face = get_face()

e_pred, e_conf = rec_eigen.predict(face)
f_pred, f_conf = rec_fisher.predict(face)
l_pred, l_conf = rec_lbph.predict(face)

print("Eigen ", labels_dict[e_pred], e_conf)
print("Fisher ", labels_dict[f_pred], f_conf)
print("LBPH ", labels_dict[l_pred], l_conf)
