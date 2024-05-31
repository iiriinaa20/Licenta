import cv2
import os

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'lbpcascade_frontalface.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Function to detect faces and save cropped images
def detect_and_save_faces(input_folder, output_folder,isSave = True):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over files in the input folder
    for i,filename in enumerate(os.listdir(input_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"): # You can add more extensions if needed
            image_path = os.path.join(input_folder, filename)
            face_crop_resized = detect_and_return_face(image_path)
            output_path = os.path.join(output_folder, f"{filename}")
            if(face_crop_resized is not None):
                cv2.imwrite(output_path, face_crop_resized)
                print(f"Face from {filename} saved as {output_path}")
    
    print("All faces saved successfully.")


def detect_and_return_face(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for i, (x, y, w, h) in enumerate(faces):
        face_crop = image[y:y+h, x:x+w]
        face_crop_resized = cv2.resize(face_crop, (128, 128))
        return face_crop_resized


input_folder = r"C:\Users\mihae\Downloads\a\train\n000002"
output_folder = r"C:\Users\mihae\Desktop\ooo\n000002"
detect_and_save_faces(input_folder, output_folder)