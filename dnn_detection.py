import cv2
import os
import numpy as np

# Load the pre-trained DNN model for face detection
modelFile = "./dnn/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./dnn/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Function to detect faces and save cropped images
def detect_and_save_faces(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # You can add more extensions if needed
            image_path = os.path.join(input_folder, filename)
            face_crop_resized = detect_and_return_face(image_path)
            if face_crop_resized is not None:
                output_path = os.path.join(output_folder, f"{filename}")
                cv2.imwrite(output_path, face_crop_resized)
                print(f"Face from {filename} saved as {output_path}")
    
    print(f"All faces from {input_folder} saved successfully.")

def detect_and_return_face(image_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust confidence threshold as needed
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face_crop = image[startY:endY, startX:endX]
            face_crop_resized = cv2.resize(face_crop, (128, 128))
            return face_crop_resized

# Replace these paths with your input and output folder paths
input_base_folder = r"C:\Users\mihae\Downloads\a\val"
output_base_folder = r"C:\Users\mihae\Desktop\ooo\a"

# Iterate over all subfolders in the input base folder
for subfolder in os.listdir(input_base_folder):
    input_folder = os.path.join(input_base_folder, subfolder)
    output_folder = os.path.join(output_base_folder, subfolder)
    
    if os.path.isdir(input_folder):
        detect_and_save_faces(input_folder, output_folder)
