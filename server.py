from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from services.camera_service import CameraService
from services.auth_service import AuthService

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

    def _setup_socketio(self):
        @self.socketio.on('request_frame')
        def handle_request_frame():
            frame_bytes = self.camera_service.get_current_frame()
            if frame_bytes:
                emit('new_frame', frame_bytes)

    def run(self):
        self.socketio.run(self.app, host=self.ip, port=self.port, debug=True)
        # self.app.run(host=self.ip, port=self.port, debug=True)