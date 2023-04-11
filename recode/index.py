#importation des modules nécessaire
import cv2
import face_recognition
import numpy as np

#utilisation du matériel webcam en étant un détecteur

url="http://ip:port/video" #si webcam mobile à serveur
video=cv2.VideoCapture(0) #0 pour le webcam par défaut 

# chargement du première image à reconnaitre
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# chargement du deuxieme image à reconnaitre
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# chargement de votre image à reconnaitre
#tester_image = face_recognition.load_image_file("image.jpg")
#tester_face_encoding = face_recognition.face_encodings(tester_image)[0]

known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    #tester_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    #"Vous"
]

while True:
    # Grab a single frame of video
    ret, frame = video.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

    print(name)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video.release()
cv2.destroyAllWindows()
