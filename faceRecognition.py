import face_recognition
import cv2

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
print("Ready to capture")
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
print(video_capture)
print("video capturing")


# Load a sample picture and learn how to recognize it.
person1_image = face_recognition.load_image_file("person1_image.jpg")
person2_image = face_recognition.load_image_file("person2_image.jpg")
person3_image = face_recognition.load_image_file("person3_image.jpg")

person1_face_encoding = face_recognition.face_encodings(person1_image)[0]
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]
person3_face_encoding = face_recognition.face_encodings(person3_image)[0]


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
names_list =[]
i=0

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        i=i+1
        print(i)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match1 = face_recognition.compare_faces([person1_face_encoding], face_encoding)
            match2 = face_recognition.compare_faces([person2_face_encoding], face_encoding)
            match3 = face_recognition.compare_faces([person3_face_encoding], face_encoding)

            name = "Unknown"

            if match1[0]:
                name = "P1"
                names_list.append(name)
            if match2[0]:
                name = "P2"
                names_list.append(name)
            if match3[0]:
                name = "P3"
                names_list.append(name)


            face_names.append(name)

    process_this_frame = not process_this_frame

    if len(names_list)==50:
        print("list > 50")
        person1 = list(filter(lambda x: x == "P1", names_list))
        person2 = list(filter(lambda x: x == "P2", names_list))
        person3 = list(filter(lambda x: x == "P3", names_list))
        max_value=max(len(person1), len(person2), len(person3))
        print("**************************************************")
        print(max_value)
        if len(person1)==max_value:
            print("Person 1")
        if len(person2)==max_value:
            print("Person 2")
        if len(person3)==max_value:
            print("Person 3")


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
