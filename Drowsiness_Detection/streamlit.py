import subprocess

# Upgrade pip
subprocess.run(["python", "-m", "pip", "install", "--upgrade", "pip"])

# Install dlib
subprocess.run(["python", "-m", "pip", "install", "dlib"])

import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoTransformerBase
from PIL import Image, ImageEnhance ,ImageFilter
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame

class VideoTransformer(VideoTransformerBase):

    def transform(self, frame):
      
        return frame

# def get_image_download_link(image_encoded, filename):
#     href = f'<a href="data:image/png;base64,{image_encoded}" download="{filename}.png">Download {filename}</a>'
#     return href

pygame.mixer.init()
audio_file = "12.mp3"  # Replace with the path to your audio file
pygame.mixer.music.load(audio_file)

face_cascade = cv2.CascadeClassifier("/models/haarcascade_frontalface_default.xml")


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
 
def single_face(img):
    face_cascade = cv2.CascadeClassifier("/models/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("/models/haarcascade_eye.xml")
    
    # Convert the PIL image to a NumPy array
    img_array = np.array(img)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    # Detect all faces in the grayscale image.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle over the face, and detect eyes in faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img_array[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    

    # Convert the NumPy array back to a PIL image
    result_image = Image.fromarray(img_array)
    return result_image



def apply_brightness(image, brightness_factor):
    enhancer = ImageEnhance.Brightness(image)
    modified_image = enhancer.enhance(brightness_factor)
    return modified_image

def change_image_color(image, color_mode):
    if color_mode == "BGR":
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return gray
    elif color_mode == "Gray":
        # Convert the image to grayscale using OpenCV
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # Convert grayscale back to RGB for display
        gray_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        print('Gray')
        return gray_image
    return image

def apply_filter(image, filter_type):
    if filter_type == "Blur Filters":
        return image.filter(ImageFilter.BLUR)
    elif filter_type == "Sharpening Filters":
        return image.filter(ImageFilter.SHARPEN)
    elif filter_type == "Gradient Filters":
        return image  # You can implement gradient filters as needed
    else:
        return image
    
def apply_morphological(image, morphological_operation):
    image_np = np.array(image)  # Convert PIL image to a numpy array
    kernel = np.ones((5, 5), np.uint8)

    if morphological_operation == "Erosion":
        eroded_image = cv2.erode(image_np, kernel, iterations=1)
        return Image.fromarray(eroded_image)  # Convert back to PIL image
    elif morphological_operation == "Dilation":
        dilated_image = cv2.dilate(image_np, kernel, iterations=1)
        return Image.fromarray(dilated_image)  # Convert back to PIL image
    elif morphological_operation == "Opening":
        opening_image = cv2.morphologyEx(image_np, cv2.MORPH_OPEN, kernel)
        return Image.fromarray(opening_image)  # Convert back to PIL image
    elif morphological_operation == "Closing":
        closing_image = cv2.morphologyEx(image_np, cv2.MORPH_CLOSE, kernel)
        return Image.fromarray(closing_image)  # Convert back to PIL image
    elif morphological_operation == "Gradient":
        gradient_image = cv2.morphologyEx(image_np, cv2.MORPH_GRADIENT, kernel)
        return Image.fromarray(gradient_image)  # Convert back to PIL image
    elif morphological_operation == "Black Hat":
        black_hat_image = cv2.morphologyEx(image_np, cv2.MORPH_BLACKHAT, kernel)
        return Image.fromarray(black_hat_image)  # Convert back to PIL image
    elif morphological_operation == "Top Hat":
        top_hat_image = cv2.morphologyEx(image_np, cv2.MORPH_TOPHAT, kernel)
        return Image.fromarray(top_hat_image)  # Convert back to PIL image
    else:
        return image

def apply_thershold(image, threshold_type):
    image_np = np.array(image)  # Convert PIL image to a numpy array
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    if threshold_type == "Binary Thersholding":
        _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        return Image.fromarray(thresholded_image)  # Convert back to PIL image

    elif threshold_type == "Inverse Binary Thersholding":
        _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
        return Image.fromarray(thresholded_image)  # Convert back to PIL image

    elif threshold_type == "Turncate Thersholding":
        _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_TRUNC)
        return Image.fromarray(thresholded_image)  # Convert back to PIL image

    elif threshold_type == "Zero Thersholding":
        _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_TOZERO)
        return Image.fromarray(thresholded_image)  # Convert back to PIL image

    elif threshold_type == "Inverted Zero Thersholding":
        _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_TOZERO_INV)
        return Image.fromarray(thresholded_image)  # Convert back to PIL image

    else:
        return image
    
def apply_canny_edge_detection(image):
    image_np = np.array(image)  # Convert PIL image to a numpy array
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 100, 200)  # You can adjust the threshold values as needed
    
    # Create a three-channel (BGR) image from the single-channel (grayscale) edge image
    edge_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return Image.fromarray(edge_image)

def apply_face_detection(image):
    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier('/models/haarcascade_frontalface_default.xml')

    # Convert the PIL image to a NumPy array
    image_np = np.array(image)

    # Convert the image to grayscale (required for face detection)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert the NumPy array back to a PIL image
    result_image = Image.fromarray(image_np)

    return result_image

def apply_eye_and_mouth_detection(image):
    # Load the Haar Cascade classifiers for eye and mouth detection
    eye_cascade = cv2.CascadeClassifier('/models/haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('/models/haarcascade_mcs_mouth.xml')

    # Convert the PIL image to a NumPy array
    image_np = np.array(image)

    # Convert the image to grayscale (required for detection)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Perform eye detection
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Perform mouth detection
    mouths = mouth_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Draw rectangles around detected mouths
    for (x, y, w, h) in mouths:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert the NumPy array back to a PIL image
    result_image = Image.fromarray(image_np)

    return result_image

def apply_contour_detection(image):
    # Convert the PIL image to a NumPy array
    image_np = np.array(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 30, 70)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set the threshold value for detecting closed eyes (adjust as needed)
    threshold_value = 1000

    # Iterate through the contours and detect closed eyes
    for contour in contours:
        if cv2.contourArea(contour) < threshold_value:
            # Consider this contour as a closed eye
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convert the NumPy array back to a PIL image
    result_image = Image.fromarray(image_np)

    return result_image

def main():
    st.title("Computer Vision Project")
    activities = ["Drowsiness Detection System"]
    summary_choice = st.selectbox("Select Activity", ["Computer Vision Operation On Image", "Drowsiness Detection Using Webcam"])
    if summary_choice == "Computer Vision Operation On Image":
        st.header("Computer Vision Operation On Image")
        st.subheader("Upload an Image")
        # Upload an image file
        image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if image is not None:
            pil_image = Image.open(image)
            st.subheader("Original Image :")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        st.sidebar.header("Select Operations : ")
        brightness = st.sidebar.slider("Brightness",0.0, 2.0, 1.0)
        selected_color = st.sidebar.radio("Select Color : ", ["None","BGR", "Gray"])
        selected_filter = st.sidebar.radio("Select Filter Type : ", ["None","Blur Filters", "Sharpening Filters","Gradient Filters"])
        selected_morphological = st.sidebar.radio("Select Morphological Operation : ", ["None","Erosion", "Dilation", "Opening", "Closing", "Gradient" , "Black Hat" , "Top Hat"])
        selected_thershold = st.sidebar.radio("Select Thershold Type : ", ["None","Binary Thersholding", "Inverse Binary Thersholding", "Turncate Thersholding" , "Zero Thersholding" , "Inverted Zero Thersholding"])
        edge_detection = st.sidebar.checkbox("Canny Edge Detection")
        drowy = st.sidebar.checkbox("Find Drowsy or Not")
        face_detection = st.sidebar.checkbox("Face Detection")
        eye_detection = st.sidebar.checkbox("Eye and Mouth Detection")
        find_contour = st.sidebar.checkbox("Find Contour")

        if brightness != 1.0:
            pil_image = apply_brightness(pil_image, brightness)
        if selected_color != "None":
            pil_image = change_image_color(pil_image,selected_color)
        if selected_filter != "None":
            pil_image = apply_filter(pil_image,selected_filter)
        if selected_morphological != "None":
            pil_image = apply_morphological(pil_image,selected_morphological)
        if selected_thershold != "None":
            pil_image = apply_thershold(pil_image,selected_thershold)
        if edge_detection:
            pil_image = apply_canny_edge_detection(pil_image)
        if face_detection:
            pil_image = apply_face_detection(pil_image)
        if eye_detection:
            pil_image = apply_eye_and_mouth_detection(pil_image)
        if find_contour:
            pil_image = apply_contour_detection(pil_image)
        if drowy:
            pil_image = single_face(pil_image)


        
            
        if image is not None:
            st.subheader("Updated Image :")
            st.image(pil_image, caption="Updated Image", use_column_width=True)
               

            # Perform image processing on the image (you can add your code here)
    if summary_choice == "Drowsiness Detection Using Webcam":
        st.header("Drowsiness Detection Using Webcam")
        start_button = st.button("Start Camera")
        stop_button = st.button("Stop Camera")

        cap = None  # Initialize cap as None

        if start_button:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Webcam not found. Make sure it is connected and accessible.")
            else:
                st.success("Webcam found. Camera is started.")

        if stop_button:
            if cap is not None:
                cap.release()
                st.warning("Camera stopped.")
            cap = None  # Set cap to None to indicate that the camera is not open

        if cap is not None:
            video_element = st.empty()  # Create a placeholder for the video element

            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
            cap = cv2.VideoCapture(0)
            flag = 0
            
            
            while True:
                ret, frame = cap.read()
                frame = imutils.resize(frame, width=450)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                subjects = detect(gray, 0)
                #Detect facial points through detector function

                #Detect faces through haarcascade_frontalface_default.xml
                face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

                #Draw rectangle around each face detected
                for (x,y,w,h) in face_rectangle:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                for subject in subjects:
                    shape = predict(gray, subject)
                    shape = face_utils.shape_to_np(shape)
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    if ear < thresh:
                        flag += 1
                        print(flag)
                        if flag >= frame_check+50:
                            cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame, "****************ALERT!****************", (10, 325),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            pygame.mixer.music.play()
                    else:
                        flag = 0
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                video_element.image(frame, channels="BGR", use_column_width=True)

if __name__ == '__main__':
    main()
