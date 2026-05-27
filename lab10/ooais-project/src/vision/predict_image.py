from pathlib import Path
from PIL import Image
import joblib
from src.vision.feature_extractor import extract_features
import matplotlib.pyplot as plt

MODEL_PATH = Path("models/image_model.joblib")

def load_model():
    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        raise SystemExit(1)
    
    model = joblib.load(MODEL_PATH)
    print("Model loaded.")
    return model


def predict_image(model, image_path):
    path = Path(image_path)
    
    if not path.exists():
        print(f"Error: file not found: {image_path}")
        return
    
    with Image.open(path) as image:
        features = extract_features(image)
        image_for_plot = image.copy()

    prediction = model.predict([features])[0]

    print("=== Prediction ===")
    print(f"Image: {image_path}")
    print(f"Predicted class: {prediction}")

    plt.imshow(image_for_plot)
    plt.title(f"Prediction: {prediction}")
    plt.axis("off")
    plt.show()


def main():
    model = load_model()
    image_path = "data/processed/images/test/forest/forest_0000.jpg"
    image_path = "data/raw/eurosat/2750/Highway/Highway_1.jpg"
    image_path = "data/inference_samples/noise.jpg"
    predict_image(model, image_path)

if __name__ == "__main__":
    main()
