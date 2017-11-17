import face_recognition
import cv2
import time
import os
from itertools import *

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

def flatten_array(array_of_arrays):
    return [element for array_in_a in array_of_arrays for element in array_in_a]

# Load a sample picture and learn how to recognize it.
person1_image1 = face_recognition.load_image_file("person1_image_copy.jpg")
person1_image2 = face_recognition.load_image_file("person1_image.jpg")
person1_image3 = face_recognition.load_image_file("person1_image3.jpg")
person1_image4 = face_recognition.load_image_file("person1_image4.jpg")
person1_image5 = face_recognition.load_image_file("person1_image5.jpg")

karthiga_image = face_recognition.load_image_file("khaarthiga.jpg")
sathish_image = face_recognition.load_image_file("sathish.jpg")
ashok_image = face_recognition.load_image_file("ashok_image.jpg")

person2_image1 = face_recognition.load_image_file("person2_image.jpg")
person2_image2 = face_recognition.load_image_file("person2_image_copy.jpg")
person2_image3 = face_recognition.load_image_file("person2_image3.jpg")
person2_image4 = face_recognition.load_image_file("person2_image4.jpg")
person2_image5 = face_recognition.load_image_file("person2_image5.jpg")

person3_image = face_recognition.load_image_file("person3_image.jpg")

person1_face_encoding1 = face_recognition.face_encodings(person1_image1)[0]
person1_face_encoding2 = face_recognition.face_encodings(person1_image2)[0]
person1_face_encoding3 = face_recognition.face_encodings(person1_image3)[0]
person1_face_encoding4 = face_recognition.face_encodings(person1_image4)[0]
person1_face_encoding5 = face_recognition.face_encodings(person1_image5)[0]

karthiga_face_encoding = face_recognition.face_encodings(karthiga_image)[0]
sathish_face_encoding = face_recognition.face_encodings(sathish_image)[0]
ashok_face_encoding = face_recognition.face_encodings(ashok_image)[0]

person2_face_encoding1 = face_recognition.face_encodings(person2_image1)[0]
person2_face_encoding2 = face_recognition.face_encodings(person2_image2)[0]
person2_face_encoding3 = face_recognition.face_encodings(person2_image3)[0]
person2_face_encoding4 = face_recognition.face_encodings(person2_image4)[0]
person2_face_encoding5 = face_recognition.face_encodings(person2_image5)[0]

person3_face_encoding = face_recognition.face_encodings(person3_image)[0]


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
names_in_last_20_frame = []
process_this_frame = True
moving_window_size=20
i=0

output_strings=[]

while True:
    start_time = time.time()
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if process_this_frame:

        i=i+1
        output_strings.append('CurrentFrame: %d' % i)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        names_in_current_frame=[]
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces([
                person1_face_encoding1,
                # person1_face_encoding2,
                # person1_face_encoding3,
                # person1_face_encoding4,
                # person1_face_encoding5,
                person2_face_encoding1,
                person3_face_encoding,
                karthiga_face_encoding,
                sathish_face_encoding,
                ashok_face_encoding
            ], face_encoding)
            name = "Unknown"
            # if match[:5]:
            #     name = "Srividhya"
            #     names_in_current_frame.append(name)
            # if match[5:10]:
            #     name = "Shruti"
            #     names_in_current_frame.append(name)
            # if match[11]:
            #     name = "P3"
            #     names_in_current_frame.append(name)
            # if match[12]:
            #     name = "Sathish"
            #     names_in_current_frame.append(name)
            # if match[13]:
            #     name = "Ashok"
            #     names_in_current_frame.append(name)
            if match[0]:
                name = "Srividhya"
                names_in_current_frame.append(name)
            if match[1]:
                name = "Shruti"
                names_in_current_frame.append(name)
            if match[2]:
                name = "P3"
                names_in_current_frame.append(name)
            if match[3]:
                name = "Sathish"
                names_in_current_frame.append(name)
            if match[4]:
                name = "Ashok"
                names_in_current_frame.append(name)


            face_names.append(name)
        names_in_last_20_frame.append(names_in_current_frame)
        names_in_last_20_frame=names_in_last_20_frame[-moving_window_size:]
        
        flattened_array = flatten_array(names_in_last_20_frame)
        # output_strings.append('names in last 20: %s' % flattened_array)
        grouped_data = [[x[0], len(list(x[1]))] for x in groupby(sorted(flattened_array))]
        # output_strings.append('grouped_data: %s' % grouped_data)
        hits_counts=sorted(grouped_data, key=lambda x: x[1], reverse=True)
        # hits_counts = [['Nobody found',0]] if len(hits_counts) == 0 or len(hits_counts[0]) == 0 else hits_counts
        output_strings.append('hits_counts: %s' % hits_counts)

        top_hits = [hits for hits in hits_counts if hits[1] >= hits_counts[0][1]*0.6 and hits[1] >= moving_window_size*0.25]
        output_strings.append('Top Hit: %s' % top_hits)
    # process_this_frame = not process_this_frame

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

    #print("FPS: ", 1.0 / (time.time() - start_time))
    if process_this_frame:
        os.system('clear')
        print('\n'.join(output_strings))
        output_strings=[]

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
