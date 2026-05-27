from pathlib import Path
from PIL import Image
import numpy as np
from src.vision.feature_extractor import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import time
import matplotlib.pyplot as plt

DATASET_DIR = Path("data/processed/images")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def load_image_split(split_dir):
    X = []
    y = []
    
    class_dirs = sorted([
        path for path in split_dir.iterdir()
        if path.is_dir()
    ])
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        image_files = sorted([
            path for path in class_dir.iterdir()
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
        
        for image_path in image_files:
            with Image.open(image_path) as image:
                features = extract_features(image)
                X.append(features)
                y.append(class_name)

    X = np.array(X)
    y = np.array(y)
    return X, y


def load_training_and_test_data():
    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"
    
    X_train, y_train = load_image_split(train_dir)
    X_test, y_test = load_image_split(test_dir)
    
    print("=== Image ML Dataset ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    print("=== Training Image Classifier ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Number of features per image: {X_train.shape[1]}")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Model trained.")

    return model

def evaluate_model(model, X_test, y_test):
    print("=== Model Evaluation ===")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Number of test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


MODEL_PATH = Path("models/image_model.joblib")

def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("=== Saving Model ===")
    print(f"Saved model: {MODEL_PATH}")


def train_and_compare(X_train, X_test, y_train, y_test):
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC()
    }
    
    results = []
    
    print("\n=== Training and Evaluating Models ===")
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start_time
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"Model: {name}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Training time: {duration:.4f} s")
        
        results.append({
            "name": name,
            "accuracy": accuracy,
            "time": duration,
            "model": model
        })
        
        joblib.dump(model, MODELS_DIR / f"{name.replace(' ', '_').lower()}.joblib")

    names = [r["name"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    times = [r["time"] for r in results]

    plt.figure(figsize=(10, 6))
    for i, name in enumerate(names):
        plt.scatter(times[i], accuracies[i], s=100)
        plt.text(times[i], accuracies[i], f" {name}")

    plt.title("Accuracy vs Training Time")
    plt.xlabel("Training Time (seconds)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("models/comparison_plot.png")
    print("\nPlot saved as models/comparison_plot.png")


def main():
    X_train, X_test, y_train, y_test = load_training_and_test_data()
    train_and_compare(X_train, X_test, y_train, y_test)
    # model = train_model(X_train, y_train)
    # evaluate_model(model, X_test, y_test)
    # save_model(model)


if __name__ == "__main__":
    main()

