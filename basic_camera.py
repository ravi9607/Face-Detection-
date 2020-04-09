import cv2

cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1, 5)

    count = 0
    for (x, y, w, h) in faces:
        roi_gray = gray[y: y+h, x: x+w]
        roi_color = frame[y: y+h, x: x+w]

        color = (255, 0, 0)
        stroke = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)
        img = 'img_'+str(count)+".png"
        count += 1
        cv2.imwrite(img, roi_color)
        print(x, y, w, h)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
