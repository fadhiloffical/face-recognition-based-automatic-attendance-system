import cv2
import os
from flask import Flask, request, render_template, redirect, url_for, flash, session,jsonify
from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import glob
import face_recognition
import os
import pickle
import sqlite3
#ADDITIONAL IMPORTS
import subprocess



# Defining Flask App
app = Flask(__name__)
app.secret_key = '12345678'


nimgs = 20

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-Period1-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-Period1-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

def train_model():
    pass



# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    # Find the latest attendance file
    latest_attendance_file = max(glob.glob('Attendance/Attendance-*.csv'), key=os.path.getctime)

    df = pd.read_csv(latest_attendance_file)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%I:%M:%S %p")  #12H format

    # Find the latest attendance file
    latest_attendance_file = max(glob.glob('Attendance/Attendance-*.csv'), key=os.path.getctime)

    df = pd.read_csv(latest_attendance_file)
    if int(userid) not in list(df['Roll']):
        with open(latest_attendance_file, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


## A function to delete a user folder 
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)




@app.route('/logout')
def logout():
    session['logged_in'] = False
    session['user_type'] = None
    session['username'] = None
    return redirect(url_for('login'))


################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():

    # Find the latest attendance file
    latest_attendance_file = max(glob.glob('Attendance/Attendance-*.csv'), key=os.path.getctime)

    # Extract attendance information from the latest file
    df = pd.read_csv(latest_attendance_file)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)

    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.sqlite')
        c = conn.cursor()

        # Query the database
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()

        if user is not None:
            session['logged_in'] = True
            session['user_type'] = user[2]
            session['username'] = user[0]
            return redirect(url_for('home'))
        else:
            flash('Invalid Credentials. Please try again.')
            return redirect(url_for('login'))

    return render_template('login.html')
## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    ## if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/')==[]:
        os.remove('static/face_recognition_model.pkl')
    
    try:
        pass
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Our main Face Recognition functionality. 
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_encodings.pkl' not in os.listdir('static/'):
        return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    # Load the face encodings dictionary from the file
    with open("static/face_encodings.pkl", "rb") as f:
        face_encodings_dict = pickle.load(f)

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
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
                    add_attendance(person_name)

        # Display the resulting image
        cv2.imshow('Attendance', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Route to create a new period (new attendance file) and clear the attendance list
@app.route('/newperiod', methods=['GET'])
def newperiod():
    # Increment the period number
    period_number = len([file for file in os.listdir('Attendance') if file.startswith('Attendance-')])

    # Create a new attendance file for the new period
    new_attendance_file = f'Attendance/Attendance-Period{period_number + 1}-{datetoday}.csv'
    with open(new_attendance_file, 'w') as f:
        f.write('Name,Roll,Time')

    # Redirect back to the home page
    return redirect(url_for('home'))



# Define the subprocess variable
face_recognition_process = None

# Function to start face recognition subprocess
def start_face_recognition(username,period):
    # print("Starting face recognition",session['username'])
    global face_recognition_process
    if face_recognition_process is None or face_recognition_process.poll() is not None:
        face_recognition_process = subprocess.Popen(["python3", "capture.py",username,str(period)])

# Function to stop face recognition subprocess
def stop_face_recognition():
    global face_recognition_process
    if face_recognition_process is not None:
        face_recognition_process.terminate()
        face_recognition_process = None

# Function to check if face recognition subprocess is running
def is_face_recognition_running():
    global face_recognition_process
    return face_recognition_process is not None and face_recognition_process.poll() is None

@app.route('/start-cam', methods=['POST'])
def start_recognition():
    data = request.get_json()
    period = data['period']
    start_face_recognition(session['username'],period)
    return jsonify({"message": "Face recognition started %s" % period})

@app.route('/stop-cam', methods=['POST'])
def stop_recognition():
    stop_face_recognition()
    return jsonify({"message": "Face recognition stopped"})

@app.route('/status-cam')
def status():
    recognition_status = "Running" if is_face_recognition_running() else "Stopped"
    return jsonify({"message": recognition_status})

@app.route('/get-attendance', methods=['GET'])
def getAttendance():
    user = session['username']
    period_number = request.args.get('period')
    conn = sqlite3.connect('users.sqlite')
    c = conn.cursor()
    c.execute("SELECT * FROM attendance WHERE date=? AND period=? AND user=?", (datetoday2, period_number, user))
    result = c.fetchall()
    conn.close()
    return jsonify(result)
@app.route('/get-report-user', methods=['POST'])
def getReport():
    user = session['username']
    data = request.get_json()
    period = data['period']
    date = datetime.strptime(data['date'], "%Y-%m-%d").strftime("%d-%B-%Y")
    conn = sqlite3.connect('users.sqlite')
    c = conn.cursor()
    c.execute("SELECT * FROM attendance WHERE date=? AND period=? AND user=?", (date, period, user))
    result = c.fetchall()
    conn.close()
    return jsonify(result)
@app.route('/get-report-all', methods=['POST'])
def getReportAll():
    user = session['username']
    data = request.get_json()
    period = data['period']
    date = datetime.strptime(data['date'], "%Y-%m-%d").strftime("%d-%B-%Y")
    conn = sqlite3.connect('users.sqlite')
    c = conn.cursor()
    c.execute("SELECT * FROM attendance WHERE date=? AND period=?", (date, period))
    result = c.fetchall()
    conn.close()
    return jsonify(result)

# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)