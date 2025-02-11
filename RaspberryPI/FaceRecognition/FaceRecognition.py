import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Function to draw rectangles around detections
def draw_rectangles(frame, detections, color, height = None):
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if height is not None:
            center_x = x + w // 2
            center_y = y + h // 2
            position_text = f"X: {center_x}, Y: {height-center_y}"
            cv2.putText(frame, position_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Function to predict emotions
def predict_emotion(face_region, model, emotions):
    resized = cv2.resize(face_region, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))  # reshape for model input
    predictions = model.predict(reshaped)
    emotion_label = np.argmax(predictions)
    return emotions[emotion_label]

def main():
    # Load cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    # Load emotion recognition model and define emotions
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    model = load_model('emotion_recognition_model.h5')

    # Start webcam
    stream = cv2.VideoCapture(0)
    if not stream.isOpened():
        print("No stream :(")
        exit()

    height = None

    while True:
        ret, frame = stream.read()
        if not ret:
            print("No more stream :(")
            break
        if height == None:
            height = frame.shape[0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=15, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_region = gray[y:y + h, x:x + w]
            emotion = predict_emotion(face_region, model, emotions)
            draw_rectangles(frame, [(x, y, w, h)], (0, 255, 0),height)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Detection with Emotion Recognition', frame)

        if cv2.waitKey(1) != -1:
            break

    stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
