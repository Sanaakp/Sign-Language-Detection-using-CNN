import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("asl_model.h5")
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
               'M', 'N', 'O']  # Update if custom
# Custom label mapping
class_map = {
    'A': 'aboard',
    'B': 'afternoon',
    'C': 'again',
    'D': 'anger',
    'E': 'archery',
    'F': 'arrest',
    'G': 'ascend',
    'H': 'assertive',
    'I': 'attedance',
    'J': 'auditorium',
    'K': 'awake',
    'L': 'enjoy',
    'M': 'drink',
    'N': 'fast',
    'O': 'help'
}

img_height, img_width = 64, 64

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw region of interest
    x1, y1 = 80, 80
    x2, y2 = 480, 480  
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    img = cv2.resize(roi, (img_width, img_height))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    # Threshold for action detection
    CONFIDENCE_THRESHOLD = 0.60

    if confidence < CONFIDENCE_THRESHOLD:
       label = "None"
    else:
       class_letter = class_names[class_index]
       label = f"{class_map[class_letter]} ({confidence*100:.2f}%)"

    # Draw results
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
