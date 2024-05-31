# app = Flask(__name__)
# socketio = SocketIO(app)

# Global variables to hold the captured frame and camera status
# current_frame = None
# camera_running = False
# cap = None
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Firebase Admin initialization
# cred = credentials.Certificate("C:\\Users\\mihae\\Downloads\\licenta-ead67-firebase-adminsdk-5u7en-d52175bb0f.json")  # Update with your actual file path
# firebase_admin.initialize_app(cred)

# Dictionary to keep track of active meetings
# active_meetings = {}

# def open_browser(recording_url):
#     webbrowser.open(recording_url)

# @app.route('/start-meeting', methods=['POST'])
# def start_meeting():
#     user_uuid = request.json['uuid']
#     if user_uuid:
#         recording_uuid = str(uuid.uuid4())
#         recording_url = f"{MAIN_URL}/recording/{recording_uuid}"
        
#         # Start the browser in a separate thread
#         threading.Thread(target=open_browser, args=(recording_url,)).start()
        
#         active_meetings[user_uuid] = recording_uuid
#         return jsonify({"message": "Meeting started", "recording_uuid": recording_uuid}), 200
#     else:
#         return jsonify({"message": "Missing UUID"}), 400

# @app.route('/stop-meeting', methods=['POST'])
# def stop_meeting():
#     user_uuid = request.json['uuid']
#     if user_uuid in active_meetings:
#         recording_uuid = active_meetings.pop(user_uuid)
#         return jsonify({"message": "Meeting stopped", "recording_uuid": recording_uuid}), 200
#     else:
#         return jsonify({"message": "No active meeting found for this user"}), 400

# @app.route('/recording/<recording_uuid>')
# def recording(recording_uuid):
#     return f"This is the recording page for {recording_uuid}"


# def detect_face(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

# def capture_frames():
#     global current_frame, cap
#     camera_index = 0  # Use 0 for the default camera, or the appropriate index for your camera
#     cap = cv2.VideoCapture(camera_index)
    
#     if not cap.isOpened():
#         raise Exception("Could not open video device")

#     while camera_running:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.flip(frame, 90)
#         detect_face(frame)
#         current_frame = frame

#     cap.release()

# @app.route('/start-camera', methods=['GET'])
# def start_camera():
#     global camera_running
#     try:
#         # Start the thread to capture frames
#         camera_running = True
#         thread = threading.Thread(target=capture_frames)
#         thread.daemon = True
#         thread.start()
#         return jsonify({"message": "Camera feed started"}), 200
#     except Exception as e:
#         return jsonify({"message": "Failed", "error": str(e)}), 500

# @app.route('/stop-camera', methods=['GET'])
# def stop_camera():
#     global camera_running, cap
#     try:
#         # Stop capturing frames and release camera
#         camera_running = False
#         if cap is not None:
#             cap.release()
#         return jsonify({"status": "Camera feed stopped"}), 200
#     except Exception as e:
#         return jsonify({"status": "Failed", "error": str(e)}), 500

# @socketio.on('request_frame')
# def handle_request_frame():
#     global current_frame
#     if current_frame is not None:
#         _, buffer = cv2.imencode('.jpg', current_frame)
#         frame_bytes = buffer.tobytes()
#         emit('new_frame', frame_bytes)

# @app.route('/main')
# def main():
#     return render_template('index.html')

# @app.route('/', methods=['GET'])
# def index():
#     return jsonify({"message": "Server Is Running"}), 200

# @app.route('/login', methods=['POST'])
# def login():
#     id_token = request.json['idToken']
#     # print(id_token)
#     # return jsonify({"message": "Login successful", "uid": 0}), 200
#     try:
#         decoded_token = auth.verify_id_token(id_token)
#         uuid = decoded_token['user_id']
#         return jsonify({"message": "Login successful", "uuid": uuid}), 200
#     except Exception as e:
#         return jsonify({"message": "Login failed", "error": str(e)}), 401        

# @app.route('/login-page', methods=['GET'])
# def login_page():
#     return render_template('login.html')

# if __name__ == '__main__':
#     socketio.run(app, host=IP, port=PORT, debug=True)
