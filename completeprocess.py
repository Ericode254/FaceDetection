import face_recognition
import mysql.connector
import cv2
import numpy as np

# Connect to the database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="@JilloErick254",
    database="verify"
)

# Create a cursor object
cursor = db.cursor()

# Query the database and get the face data
sql = "SELECT name, person_image FROM people"
cursor.execute(sql)
results = cursor.fetchall()

# Create a list of names, images, and encodings
names = []
images = []

for result in results:
    names.append(result[0])
    images.append(result[1])

encoding = face_recognition.face_encodings(images[0])[0]
# Create a video capture object
video_capture = cv2.VideoCapture("Vid/stallone.webm")

# Loop over the frames
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    # resize for better detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the frame to RGB color
    rgb_frame = small_frame[:, :, ::-1]

    # Get the face locations and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop over the faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face with the database
        matches = face_recognition.compare_faces([encoding], face_encoding)
        name = "Unknown"

        # Find the best match
        if True in matches:
            match_index = matches.index(True)
            name = names[match_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Put the name on the frame
        cv2.putText(frame, name[0], (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the frame in a window
    cv2.imshow("Video", frame)

    # Check for keyboard input
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the window
video_capture.release()
cv2.destroyAllWindows()
