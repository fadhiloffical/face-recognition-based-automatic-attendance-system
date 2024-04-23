import cv2
import os
from flask import Flask, request, render_template, redirect, url_for, flash, session
from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import glob
import face_recognition
import pickle
import sqlite3
import sys


# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def add_attendance(person):
    name = person.split('_')[0]
    roll_no = person.split('_')[1]
    user = sys.argv[1]
    period = sys.argv[2]
    current_time = datetime.now().strftime("%I:%M:%S %p")  #12H format
    conn = sqlite3.connect('users.sqlite')
    c = conn.cursor()
    c.execute("SELECT * FROM attendance WHERE Name=? AND roll_no=? AND Date=? AND Period=?", (name, roll_no, datetoday2, period))
    result = c.fetchone()
    if result is None:
        c.execute("INSERT INTO attendance VALUES (?,?,?,?,?,?)", (name, roll_no, current_time, datetoday2, user, period))
        conn.commit()
        print("Attendance added for", name)
    conn.close()

    # Load the face encodings dictionary from the file
with open("static/face_encodings.pkl", "rb") as f:
    face_encodings_dict = pickle.load(f)

ret = True
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture()
cap.open("http://192.168.1.3:8001")
while ret:
    ret, frame = cap.read()
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        for person_name, person_face_encodings in face_encodings_dict.items():
            # Calculate the face distance
            face_distances = face_recognition.face_distance(person_face_encodings, face_encoding)
            # Find the best match
            best_match_index = np.argmin(face_distances)
            # If the best match is below the threshold, display the detected person
            if face_distances[best_match_index] < 0.4:  # Adjust this value based on your requirement
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, person_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                # Add attendance
                try:
                    add_attendance(person_name)
                except Exception as e:
                    print("Error adding attendance",e)
                    pass
                
    # Display the resulting image
    cv2.imshow('Attendance', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()


