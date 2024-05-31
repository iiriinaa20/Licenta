import threading
import cv2

# import dlib
# detector = dlib.get_frontal_face_detector()

# # Example usage
# def detect_face_dlib(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray, 1)
#     for face in faces:
#         x, y, w, h = (face.left(), face.top(), face.width(), face.height())
#         cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)



# cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

# # Example usage
# def detect_face_dlib_cnn(image):
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     faces = cnn_face_detector(rgb_image, 1)
#     for face in faces:
#         x, y, w, h = (face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height())
#         cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

class CameraService:
    def __init__(self, face_detection_service):
        self.current_frame = None
        self.cap = None
        self.camera_running = False
        self.face_detection_service = face_detection_service

    def detect_face(self, frame):
        faces = self.face_detection_service.detect_and_return_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    def capture_frames(self):
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.detect_face(frame)
            frame = cv2.flip(frame, 90)
            self.current_frame = frame

        self.cap.release()

    def start_camera(self):
        self.camera_running = True
        thread = threading.Thread(target=self.capture_frames)
        thread.daemon = True
        thread.start()

    def stop_camera(self):
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()

    def get_current_frame(self):
        if self.current_frame is not None:
            _, buffer = cv2.imencode('.jpg', self.current_frame)
            return buffer.tobytes()
        return None
