import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# loading models
age_model = cv2.dnn.readNet("C:\\Users\\raghu\\Desktop\\Internship\\project\\Age prediction\\age_net.caffemodel", "C:\\Users\\raghu\\Desktop\\Internship\\project\\Age prediction\\age_deploy.prototxt")
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

gender_model = load_model("C:\\Users\\raghu\\Desktop\\Internship\\project\\Gender-Detection\\gender_detection.model")
classes = ['man', 'woman']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

faceCascade = cv2.CascadeClassifier("C:\\Users\\raghu\\Desktop\\Internship\\project\\Haarcascade\\haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)
padding = 20

while True:
    ret, frame = video_capture.read()

    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # black and white image
    faces = faceCascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) # detect faces

    for (x, y, w, h) in faces:
        face = frame[max(0, y - padding):min(y + h + padding, frame.shape[0] - 1),
                     max(0, x - padding):min(x + w + padding, frame.shape[1] - 1)]
        
        # detect age
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        age_model.setInput(blob)
        agePred = age_model.forward()
        age = ageList[agePred[0].argmax()]

        # Get Face
        face_crop = cv2.resize(face, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # detect gender 
        conf = gender_model.predict(face_crop)[0]
        idx = np.argmax(conf)
        gender = classes[idx]

        label = "Gender: {}, Age: {}".format(gender, age)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Age-Gender", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()