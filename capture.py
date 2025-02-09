import cv2
import numpy as np
import face_recognition
import pickle
import sqlite3
import sys
from datetime import date, datetime
import pytz

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

    # Set the UTC timezone
    utc_tz = pytz.timezone('UTC')
    # Get the current UTC time
    utc_time = datetime.now(utc_tz)

    # Convert the UTC time to your local timezone (e.g., 'Asia/Kolkata')
    local_tz = pytz.timezone('Asia/Kolkata')
    local_time = utc_time.astimezone(local_tz)

    current_time = local_time.strftime("%I:%M:%S %p")  # 12H format
    
    conn = sqlite3.connect('users.sqlite')
    c = conn.cursor()
    c.execute("SELECT * FROM attendance WHERE Name=? AND roll_no=? AND Date=? AND Period=?", 
              (name, roll_no, datetoday2, period))
    result = c.fetchone()
    if result is None:
        c.execute("INSERT INTO attendance VALUES (?,?,?,?,?,?)", 
                  (name, roll_no, current_time, datetoday2, user, period))
        conn.commit()
        print("Attendance added for", name)
    conn.close()

# Load the face encodings dictionary from the file
try:
    with open("static/face_recognition.pkl", "rb") as f:
        model_data = pickle.load(f)
        knn = model_data["model"]
        label_encoder = model_data["label_encoder"]
        face_encodings = model_data["face_encodings"]
        labels = model_data["labels"]
    print("Pickle file loaded successfully.")
except (pickle.UnpicklingError, IOError, EOFError) as e:
    print(f"Error loading the pickle file: {e}")
    exit()

ret = True
cap = cv2.VideoCapture("http://172.24.192.1:8081/video.mjpg")  # Use the correct IP camera URL
# cap = cv2.VideoCapture(0)  # Use the system camera

# Attempt to open the webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully.")

# Create a resizable window
cv2.namedWindow('Attendance', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Attendance', 800, 600)  # Set initial window size

# Get screen dimensions
screen_width = 1920  # Replace with your screen width
screen_height = 1080  # Replace with your screen height

# Get window dimensions
window_width, window_height = cv2.getWindowImageRect('Attendance')[2:4]

# Calculate position to center the window
x_pos = (screen_width - window_width) // 2
y_pos = (screen_height - window_height) // 2

# Move the window to the center of the screen
cv2.moveWindow('Attendance', x_pos, y_pos)

while ret:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings_current_frame = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_current_frame):
        # Predict the person in the current frame
        closest_distances = knn.kneighbors([face_encoding], n_neighbors=1)
        if closest_distances[0][0][0] < 0.4:  # Adjust this threshold as needed
            match_index = knn.predict([face_encoding])[0]
            person_name = label_encoder.inverse_transform([match_index])[0]
            
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
                print("Error adding attendance", e)
                pass
                
    # Display the resulting image
    cv2.imshow('Attendance', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
