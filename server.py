from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from services.camera_service import CameraService
from services.auth_service import AuthService
from services.repositories import *

class FlaskServer:
    def __init__(self, ip: str, port: int, main_url: str, camera_service: CameraService, auth_service: AuthService):
        self.ip = ip
        self.port = port
        self.main_url = main_url
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)

        self.camera_service = camera_service        
        self.auth_service = auth_service        
        
        self._setup_routes()
        self._setup_socketio()

    def _setup_routes(self):
        @self.app.route('/', methods=['GET'])
        def index():
            return jsonify({"message": "Server Is Running"}), 200

        @self.app.route('/main')
        def main():
            return render_template('index.html')

        @self.app.route('/login-page', methods=['GET'])
        def login_page():
            return render_template('login.html')

        @self.app.route('/login', methods=['POST'])
        def login():
            jwt_token = request.json.get('idToken')
            user_id = self.auth_service.login(jwt_token)
            if user_id:
                return jsonify({"message":"Logged in successfully", "user_id": user_id}), 200
            else:
                return jsonify({"error": "Invalid token"}), 401

        @self.app.route('/logout', methods=['POST'])
        def logout():
            jwt_token = request.json.get('idToken')
            if self.auth_service.logout(jwt_token):
                return jsonify({"message": "Logged out successfully"}), 200
            else:
                return jsonify({"error": "Invalid token or logout failed"}), 401

        @self.app.route('/start-camera', methods=['GET'])
        def start_camera():
            try:
                self.camera_service.start_camera()
                return jsonify({"message": "Camera feed started"}), 200
            except Exception as e:
                return jsonify({"message": "Failed", "error": str(e)}), 500

        @self.app.route('/stop-camera', methods=['GET'])
        def stop_camera():
            try:
                self.camera_service.stop_camera()
                return jsonify({"message": "Camera feed stopped"}), 200
            except Exception as e:
                return jsonify({"message": "Failed", "error": str(e)}), 500
       
        @self.app.route('/users', methods=['POST'])
        def create_user():
            data = request.json
            UserRepository.create(data)
            return jsonify({"msg": "User created successfully"}), 201

        @self.app.route('/users/<user_id>', methods=['GET'])
        def read_user(user_id):
            user = UserRepository.read(user_id)
            if user:
                return jsonify(user)
            else:
                return jsonify({"error": "User not found"}), 404

        @self.app.route('/users/<user_id>', methods=['PUT'])
        def update_user(user_id):
            data = request.json
            UserRepository.update(user_id, data)
            return jsonify({"msg": "User updated successfully"})

        @self.app.route('/users/<user_id>', methods=['DELETE'])
        def delete_user(user_id):
            UserRepository.delete(user_id)
            return jsonify({"msg": "User deleted successfully"})

        # Course Routes
        
        @self.app.route('/add_course_form')
        def add_course_form():
             return render_template('add_course_form.html')

        @self.app.route('/courses', methods=['GET'])
        def read_courses():
            courses = CourseRepository.read_all()
            return jsonify(courses), 200
        
        @self.app.route('/courses', methods=['POST'])
        def create_course():
            data = request.json
            CourseRepository.create(data)
            return jsonify({"msg": "Course created successfully"}), 201

        @self.app.route('/courses/<course_id>', methods=['GET'])
        def read_course(course_id):
            course = CourseRepository.read(course_id)
            if course:
                return jsonify(course)
            else:
                return jsonify({"error": "Course not found"}), 404

        @self.app.route('/courses/<course_id>', methods=['PUT'])
        def update_course(course_id):
            data = request.json
            CourseRepository.update(course_id, data)
            return jsonify({"msg": "Course updated successfully"})

        @self.app.route('/courses/<course_id>', methods=['DELETE'])
        def delete_course(course_id):
            CourseRepository.delete(course_id)
            return jsonify({"msg": "Course deleted successfully"})

        # Courses Planification Routes
        @self.app.route('/courses_planification', methods=['POST'])
        def create_courses_planification():
            data = request.json
            CoursesPlanificationRepository.create(data)
            return jsonify({"msg": "Courses Planification created successfully"}), 201

        @self.app.route('/courses_planification/<plan_id>', methods=['GET'])
        def read_courses_planification(plan_id):
            plan = CoursesPlanificationRepository.read(plan_id)
            if plan:
                return jsonify(plan)
            else:
                return jsonify({"error": "Courses Planification not found"}), 404

        @self.app.route('/courses_planification/<plan_id>', methods=['PUT'])
        def update_courses_planification(plan_id):
            data = request.json
            CoursesPlanificationRepository.update(plan_id, data)
            return jsonify({"msg": "Courses Planification updated successfully"})

        @self.app.route('/courses_planification/<plan_id>', methods=['DELETE'])
        def delete_courses_planification(plan_id):
            CoursesPlanificationRepository.delete(plan_id)
            return jsonify({"msg": "Courses Planification deleted successfully"})

        # Attendance Routes
        @self.app.route('/attendance', methods=['POST'])
        def create_attendance():
            data = request.json
            AttendanceRepository.create(data)
            return jsonify({"msg": "Attendance created successfully"}), 201

        @self.app.route('/attendance/<attendance_id>', methods=['GET'])
        def read_attendance(attendance_id):
            attendance = AttendanceRepository.read(attendance_id)
            if attendance:
                return jsonify(attendance)
            else:
                return jsonify({"error": "Attendance not found"}), 404

        @self.app.route('/attendance/<attendance_id>', methods=['PUT'])
        def update_attendance(attendance_id):
            data = request.json
            AttendanceRepository.update(attendance_id, data)
            return jsonify({"msg": "Attendance updated successfully"})

        @self.app.route('/attendance/<attendance_id>', methods=['DELETE'])
        def delete_attendance(attendance_id):
            AttendanceRepository.delete(attendance_id)
            return jsonify({"msg": "Attendance deleted successfully"})
        
        @self.app.route('/view-courses', methods=['GET'])
        def view_courses():
            return render_template('courses.html')

    def _setup_socketio(self):
        @self.socketio.on('request_frame')
        def handle_request_frame():
            frame_bytes = self.camera_service.get_current_frame()
            if frame_bytes:
                emit('new_frame', frame_bytes)

    def run(self):
        self.socketio.run(self.app, host=self.ip, port=self.port, debug=True)
        # self.app.run(host=self.ip, port=self.port, debug=True)