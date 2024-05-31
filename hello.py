from server import FlaskServer
from services.auth_service import AuthService
from services.camera_service import CameraService
from services.face_detection_service import FaceDetectionService
from env import *

FACE_DETECTION_SERVICE = FaceDetectionService(config_required=True, 
                                              confidence_level=0.7, 
                                              model_file_path=FACE_DETECTION_MODEL_FILE, 
                                              config_file_path=FACE_DETECTION_CONFIG_FILE, 
                                              face_cascade_path=FACE_DETECTION_CASCADE_NAME)

CAMERA_SERVICE = CameraService(face_detection_service= FACE_DETECTION_SERVICE)

AUTH_SERVICE = AuthService(credentials_path=AUTH_CREDENTIALS_PATH)

if __name__ == '__main__':
    server = FlaskServer(IP,PORT,MAIN_URL,CAMERA_SERVICE, AUTH_SERVICE)
    server.run()