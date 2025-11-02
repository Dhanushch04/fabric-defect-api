import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.responses import JSONResponse
import numpy as np
import tensorflow as tf
import cv2
import joblib
import io

# --- Configuration Constants (Must match your training) ---
IMG_SIZE = (128, 128)
NUM_CHANNELS = 1
MODEL_PATH = 'fabric_defect_model.h5' # Update if needed
ENCODER_PATH = 'label_encoder.pkl'   # Update if needed

# --- Initialization ---
app = FastAPI(title="Fabric Defect Detection API")

# Global variables for the model and encoder
global model, label_encoder, defect_indices

# Load the model and encoder when the application starts
@app.on_event("startup")
def load_resources():
    """Load the Keras model and the LabelEncoder at startup."""
    global model, label_encoder, defect_indices
    try:
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH)

        # Load the label encoder
        label_encoder = joblib.load(ENCODER_PATH)

        # Determine the indices that correspond to defect classes
        # Assuming 'good' is the only non-defect class
        all_classes = list(label_encoder.classes_)
        defect_classes = [c for c in all_classes if c.lower() != 'good']
        defect_indices = [label_encoder.transform([c])[0] for c in defect_classes]
        
        print("Model and encoder loaded successfully.")
        print(f"Defect classes for binary output: {defect_classes}")

    except Exception as e:
        print(f"Error loading resources: {e}")
        # In a production environment, you might raise an error here
        
# --- Prediction Function ---
def preprocess_image(image_bytes):
    """Reads, resizes, and normalizes the image for the model."""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Read image in grayscale
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Could not decode image.")

    # Resize to the model's input size
    img = cv2.resize(img, IMG_SIZE)
    
    # Normalize and reshape
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1) # Add channel dimension (128, 128, 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 128, 128, 1)
    
    return img

# --- API Endpoint ---
@app.post("/predict_defect")
async def predict_defect(file: UploadFile = File(...)):
    """
    Receives an image, predicts the defect type, and returns
    a binary "Good" or "Bad" result.
    """
    try:
        # 1. Read the uploaded file
        image_bytes = await file.read()
        
        # 2. Preprocess the image
        input_data = preprocess_image(image_bytes)
        
        # 3. Make Prediction
        predictions = model.predict(input_data)
        predicted_index = np.argmax(predictions, axis=1)[0]
        
        # 4. Decode Prediction
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        
        # 5. Determine Binary Output ("Good" or "Bad")
        if predicted_index in defect_indices:
            result = "Bad (Defect Detected)"
        else:
            # We assume any non-defect class is "good"
            result = "Good (No Defect)"
        
        # Optional: Get confidence for the predicted class
        confidence = float(np.max(predictions)) * 100
        
        # 6. Return the result
        return JSONResponse(content={
            "filename": file.filename,
            "prediction_binary": result,
            "prediction_detail": predicted_label,
            "confidence": f"{confidence:.2f}%"
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch any unexpected error
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# --- Run the API ---
# To run this locally, execute this in your terminal:
# uvicorn app:app --reload

# For Colab/Jupyter (optional, often requires external tools like ngrok for external access)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)