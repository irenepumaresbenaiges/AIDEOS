import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from src.vision.feature_extractor import extract_features

MODELS_DIR = Path("models")
IMAGE_PATH = Path("data/processed/images/test/forest/forest_0000.jpg")

def predict_with_all():
    image = Image.open(IMAGE_PATH)
    features = extract_features(image).reshape(1, -1)
    
    print(f"Image: {IMAGE_PATH}")

    model_names = ["random_forest", "knn", "logistic_regression", "svm"]
    predictions = {}

    for name in model_names:
        model_path = MODELS_DIR / f"{name}.joblib"

        if model_path.exists():
            model = joblib.load(model_path)
            pred = model.predict(features)[0]
            display_name = name.replace('_', ' ').title()
            if display_name == "Knn": display_name = "KNN"
            if display_name == "Svm": display_name = "SVM"
            predictions[display_name] = pred
            print(f"{display_name}: {pred}")


    title = " | ".join([f"{n}: {p}" for n, p in predictions.items()])
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.title(title, fontsize=10)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    predict_with_all()