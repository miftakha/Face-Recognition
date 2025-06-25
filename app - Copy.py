import cv2 as cv
import face_recognition
import numpy as np

vid = cv.VideoCapture(0)

miftahul_image = face_recognition.load_image_file("miftahul.jpg")
gerald_image = face_recognition.load_image_file("gerald.jpg")
miftahul_encoding = face_recognition.face_encodings(miftahul_image)[0]
gerald_encoding = face_recognition.face_encodings(gerald_image)[0]

print('hasil encoding miftahul.jpg')
print(miftahul_encoding)
print('\n')
print('hasil encoding gerald.jpg')
print(gerald_encoding)
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

video_capture = cv.VideoCapture(0)
while(True):
    #Capture frame-by-frame
    ret, frame = video_capture.read()
    image_gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(
        frame, face_locations)
    name = "Unknown"
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding)
        print('matches')
        print(matches)
        name = "Unknown"
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        print("best_match_index")
        print(best_match_index)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    #draw rectangles
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)
    #Display result on video
    cv.imshow('Video', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
#when done
video_capture.release()
cv.destroyAllWindows()