import cv2 as cv
import face_recognition
import numpy as np

vid = cv.VideoCapture(0)

miftahul_image = face_recognition.load_image_file("miftahul.jpg")
gerald_image = face_recognition.load_image_file("gerald.jpg")
miftahul_encoding = face_recognition.face_encodings(miftahul_image)[0]
gerald_encoding = face_recognition.face_encodings(gerald_image)[0]

known_face_encodings = [
    miftahul_encoding, gerald_encoding
]

known_face_names = [
    "miftahul", "gerald"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

"""while(True):
    ret, frame = vid.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
    process_this_frame = not process_this_frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()"""

video_capture = cv.VideoCapture(0)
while(True):
    #Capture frame-by-frame
    ret, frame = video_capture.read()
    image_gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    detections = face_detector.detectMultiScale(image_gray, minSize=(100,100))
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(
        frame, face_locations)
    name = "Unknown"
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    #draw rectangles
    for(x, y, w, h) in detections:
        print(w, h, face_locations)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.rectangle(frame, (x, y - 35),
                      (x+w, y), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (x + 6, y - 6),
                    font, 1.0, (255, 255, 255), 1)
    #Display result on video
    cv.imshow('Video', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
#when done
video_capture.release()
cv.destroyAllWindows()
