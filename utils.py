import cv2
import numpy as np

IMG_SIZE = 224

def preprocess_image(img):
    if img is None or img.size == 0:
        return None

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert BGR → RGB (IMPORTANT for consistency with training)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img