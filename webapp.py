from flask import Flask, Response, render_template, request, jsonify
import cv2
import datetime
import time
import numpy as np
import os
import json
import numpy as np 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier 
import streamlit



   
app = Flask(__name__)
f_name = "data.npy"
user_file = "users.json"
users = {}
recording = False
predicted_name = ""
camera = cv2.VideoCapture(0)
attendence_sheet = []
attendence_file = 'attendence.json'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
frames = []
outputs = []
total_sheet = {}
image_names = ""

if os.path.exists('attendence.json'):
    with open('attendence.json','r') as file:
        try:
            total_sheet = json.load(file)
            print(total_sheet)
        except json.decoder.JSONDecodeError :
            total_sheet = {}

if os.path.exists('users.json') and  os.path.getsize(user_file) > 0:
    with open(user_file,'r') as file:
        try:
            users = json.load(file)
            attendence_sheet = [{'name': user, 'rollno': users[user]['rollno']} for user in users]
        except json.decoder.JSONDecodeError:
            users = {}
def generate_frames():
    global recording, frames, outputs, image_name
    start_time = time.time()
    while recording:
        ret, frame = camera.read()
        if not ret or frame is None:
            continue

        faces = face_cascade.detectMultiScale(frame, 1.1, 4)

        for (x, y, w, h) in faces:
            cut_face = frame[y:y+h,x:x+w]
            cut_face = cv2.resize(cut_face,(100,100))
            gray_face = cv2.cvtColor(cut_face, cv2.COLOR_BGR2GRAY)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

            face_array = np.array(gray_face).flatten()
            if (len(frames) <= 250):
                 frames.append(face_array)
                 outputs.append([image_name])

        current_time = time.time() - start_time
        current_time = str(datetime.timedelta(seconds=int(current_time)))
        cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
        except Exception as e:
            continue
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        if not recording:
            break

    # Save data
    x = np.array(frames)
    y = np.array(outputs)
    if x.size == 0 or y.size == 0:
        print("No data to save.")


    y = y.reshape(-1, 1)  # Ensure y is a column vector
    data = np.hstack([y, x])

    if os.path.exists(f_name):
        data = np.load('data.npy',allow_pickle=True)
        if old.shape[1] == data.shape[1]:  # Check if dimensions match
            data = np.vstack([old, data])
        else:
            print("Mismatch in data dimensions. Not saving the new data.")
            return

    np.save(f_name, data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('https://facial-attendence-system-70845hviq-shys-projects-25fb9959.vercel.app/adduser')
def adduser():
    return render_template('adduser.html')

@app.route('https://facial-attendence-system-70845hviq-shys-projects-25fb9959.vercel.app/display')
def display():
    return render_template('display.html')
@app.route('https://facial-attendence-system-70845hviq-shys-projects-25fb9959.vercel.app/users')
def get_users():
    print(users)
    if not os.path.exists(f_name):
        return render_template("usersdisplay.html",users={})
    else:
        data = np.load("data.npy")
        y = data[:,0]
        users_list = [{'name': user, 'age': users[user]['age'],'rollno': users[user]['rollno']} for user in users]
    
        return render_template("usersdisplay.html",users=users_list)

@app.route('https://facial-attendence-system-70845hviq-shys-projects-25fb9959.vercel.app/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_name', methods=['POST'])
def set_name():
    global image_name, recording
    try:
        data = request.get_json()
        image_name = data.get('name')
        recording = True
        users[image_name] = {"age" :data.get('age'),'rollno':data.get("rollno")}
        with open(user_file, 'w') as file:
            json.dump(users, file)
        return jsonify({"message": "Image name received"}), 200
    except Exception as e:
        print(f"Error in /set_name: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/stop_recording')
def stop_recording():
    global recording
    recording = False
    camera.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Recording stopped"}), 200

@app.route('https://facial-attendence-system-70845hviq-shys-projects-25fb9959.vercel.app/predict')
def predict():
    return render_template("resultpage.html")
@app.route('/take_attendence')
def take_attendence():
    data = np.load('data.npy',allow_pickle=True)
    X = data[:,1:].astype(int)
    y = data[:,0]
    
    model  = KNeighborsClassifier()
    model.fit(X,y)
    return Response(face_detect(model), mimetype='multipart/x-mixed-replace; boundary=frame')

def face_detect(model):
    global predicted_name

    while True:
        ret ,frame = camera.read()

        if ret:
            faces = face_cascade.detectMultiScale(frame, 1.1, 4)
            for face in faces:
                x ,y ,w ,h = face

                cut = frame[y: y +h ,x: x +w]
                fix = cv2.resize(cut ,(100 ,100))
                gray = cv2.cvtColor(fix ,cv2.COLOR_BGR2GRAY)
        
                out = model.predict([gray.flatten()])
                out_array = np.array(out)
                predicted_name = out_array
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,str(out[0]),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)
        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
        except Exception as e:
            continue
        
        
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
@app.route('https://facial-attendence-system-70845hviq-shys-projects-25fb9959.vercel.app/finish') 
def finish():
    global predicted_name, total_sheet
    camera.release()
    cv2.destroyAllWindows()
    
    today_str = datetime.date.today().strftime("%Y-%m-%d")  # Convert date to string
    
    if today_str not in total_sheet:
        total_sheet[today_str] = []

    for student in attendence_sheet:
        if student['name'] ==  predicted_name:
            if ({'name': student['name'], 'rollno': student['rollno']} not in total_sheet[today_str]):
                total_sheet[today_str].append({'name': student['name'], 'rollno': student['rollno']})
    
    
    
    # Function to handle np.ndarray objects for JSON serialization
    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')
    
    # Dump the dictionary into JSON with a custom encoder to handle ndarrays
    with open('attendence.json', 'w') as file:
        json.dump(total_sheet, file, default=convert_ndarray)
                
    return jsonify({"message": "Attendance has been taken"})

# Example of how to release the camera (assuming camera is a cv2.VideoCapture object)
# camera = cv2.VideoCapture(0)  # This should be defined somewhere in your code


def load_attendence():
    data = {}
    with open('attendence.json', 'r') as file:
        data = json.load(file)
    print(data)
    return data
@app.route('/attendence_display')
def attendence_dislplay():
    return render_template('displayattendence.html',attendence=load_attendence())



app.run(debug=True)


