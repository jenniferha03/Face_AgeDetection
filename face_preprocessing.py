# Import required modules
import cv2
import numpy as np
import base64
from deepface import DeepFace
import dlib
import os

# Load dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load Age Detection Model
age_weights = "age_net.caffemodel"
age_config = "age_deploy.prototxt"
age_net = cv2.dnn.readNet(age_weights, age_config)

# Define age categories
age_list = ['(0-3)', '(4-7)', '(8-13)', '(14-24)', '(25-37)', '(38-47)', '(48-59)', '(60-100)']
model_mean = (78.4263377603, 87.7689143744, 114.895847746)

# Function to get eye centers
def get_eye_centers(landmarks):
    left_eye_pts = landmarks[36:42]  # Left eye landmarks (indexes 36-41)
    right_eye_pts = landmarks[42:48]  # Right eye landmarks (indexes 42-47)

    # Compute the mean of the landmark points to get the center
    left_eye_center = np.mean(left_eye_pts, axis=0).astype("int")
    right_eye_center = np.mean(right_eye_pts, axis=0).astype("int")

    return left_eye_center, right_eye_center

# Align face function
def align_face(image, face):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = landmark_predictor(gray, face)
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

    left_eye_center, right_eye_center = get_eye_centers(landmarks)

    # Compute angle between the eyes
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Compute center between eyes
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                   (left_eye_center[1] + right_eye_center[1]) // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    # Apply affine transformation to align the face
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    return aligned_image

# Function for preprocessing (resizing, normalization, and noise reduction)
def preprocess_image(face_img):
    try:
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(face_img, (5, 5), 0)
        # Resize image to the required input size for the age detection model
        resized = cv2.resize(blurred, (227, 227))
        return resized
    
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None

# Detects faces using dlib
def detect_faces(image_path):
    img = cv2.imread(image_path)

    if image_path is None:
        print(f"Error: Could not load image {image_path}")
        return [], image_path
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    face_images = []
    face_boxes = []
    
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Ensure valid bounding box
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            continue

        face_img = img[y:y+h, x:x+w]
        face_images.append(face_img)
        face_boxes.append((x, y, x+w, y+h))

        # Draw rectangle around detected face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return face_images, face_boxes, img

# Perform Age Prediction using OpenCV DNN
def predict_age(face_img):
    try:
        # Preprocessing image for better model input
        face_img_resized = cv2.resize(face_img, (227, 227))
        blob = cv2.dnn.blobFromImage(face_img_resized, 1.0, (227, 227), model_mean, swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        
        # Get predicted age from the list
        predicted_age = age_list[age_preds[0].argmax()]
        age_confidence = age_preds[0][age_preds[0].argmax()] * 100
        
        return predicted_age, age_confidence
    except Exception as e:
        print(f"Error predicting age: {e}")
        return "Unknown"

# Main function to execute the process
def main():
    output_dir = "extracted_faces"
    os.makedirs(output_dir, exist_ok=True)

    img1_path = "/Users/jenniferha/Library/CloudStorage/OneDrive-NorthCentralCollege/SP 25/CSCE 494 Capstone/DeepFace/dataset/kid2.jpeg"
    
    # Read the image (original)
    img = cv2.imread(img1_path)

    if img is None:
        print("❌ Error: Could not load the image.")
        return
    
    faces, face_boxes, processed_img = detect_faces(img1_path) 

    print(f"✅ Face detection complete. {len(faces)} faces detected.")

    if not faces:
        print("❌ No faces detected. Exiting.")
        return

    # Copy of the original image for applying blur
    img_with_blur = img.copy()

    # Process each detected face
    for i, (face, box) in enumerate(zip(faces, face_boxes)):
        x, y, x2, y2 = box

        # Predict Age with confidence
        age, confidence = predict_age(face)

        # Annotate image with detected age and confidence
        text = f"{age} {confidence:.2f}%"
        cv2.putText(processed_img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Preprocess the face (Apply Gaussian blur)
        preprocessed_face = preprocess_image(face)

        if preprocessed_face is None:
            continue

        # Resize preprocessed face to match the original size of the detected face
        preprocessed_face_resized = cv2.resize(preprocessed_face, (x2 - x, y2 - y))

        # Place the resized preprocessed face (blurred) into the original image region
        img_with_blur[y:y2, x:x2] = preprocessed_face_resized

        age1, confidence1 = predict_age(preprocessed_face_resized)

        text1 = f"{age1} {confidence1:.2f}%"
        cv2.putText(img_with_blur, text1, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow("Processed Image", processed_img)
    cv2.waitKey(1)

    for i, (face, box) in enumerate(zip(faces, face_boxes)):
        filename = os.path.join(output_dir, f"image_with_age_{i+1}.jpg")
        cv2.imwrite(filename, processed_img)  
        print(f"✅ Saved processed image with age annotations: {filename}")

        # Save preprocessed face
        filename_preprocessed = os.path.join(output_dir, f"preprocessed_image_{i+1}.jpg")
        cv2.imwrite(filename_preprocessed, img_with_blur)
        print(f"✅ Saved image with Gaussian blur and age annotations: {filename_preprocessed}")


    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
