import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('images.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
count = 0

for (x, y, w, h) in faces:
    roi_gray = gray_img[y: y+h, x: x+w]

    color = (255, 0, 0)
    stroke = 2
    cv2.rectangle(gray_img, (x, y), (x+w, y+h), color, stroke)
    img = 'output_folder/img_'+str(count)+".png"
    count += 1
    cv2.imwrite(img, roi_gray)
    print(x, y, w, h)

cv2.imshow('gray_img', gray_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

