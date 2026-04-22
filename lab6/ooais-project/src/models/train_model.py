import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

if __name__ == "__main__":

    print("=== Machine Learning: Loading Feature Dataset ===")
    
    features_path = "data/processed/model_features.csv"
    if not os.path.exists(features_path):
        print(f"ERROR: The file {features_path} doesn't exist.")
    df_features = pd.read_csv(features_path)
    
    print(f"Input file: {features_path}")
    print(f"Records loaded: {len(df_features)}")
    print(f"Columns: {list(df_features.columns)}")

    print("\n=== Machine Learning: Preparing Features and Target ===")

    labels_path = "data/processed/model_labels.csv"
    if not os.path.exists(labels_path):
        print(f"ERROR: The file {labels_path} doesn't exist.")
    df_labels = pd.read_csv(labels_path)

    X = df_features.values.tolist()
    y = df_labels['anomaly_flag'].tolist()
    
    print(f"Number of samples in X: {len(X)}")
    print(f"Number of labels in y: {len(y)}")
    
    target_values = sorted(list(set(y)))
    print(f"Target values detected: {target_values}")

    print("\n=== Machine Learning: Train/Test Split ===")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    print("\n=== Machine Learning: Model Training ===")
    
    model = DecisionTreeClassifier()
    
    model.fit(X_train, y_train)
    
    print(f"Model: DecisionTreeClassifier")
    print("Training completed successfully.") 

    print("\n=== Machine Learning: Prediction ===")
    
    predictions = model.predict(X_test)
    predictions = [int(p) for p in predictions]

    print("Predictions generated for test set.")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Example predictions:\n {list(predictions[:5])}")

    print("\n=== Machine Learning: Evaluation ===")
    
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    print("\n=== Machine Learning: Saving and Inspecting Model ===")
    
    model_path = "results/decision_tree_model.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model: {model_path}")
    print(f"Model type: {type(model).__name__}")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    
    feature_names = list(df_features.columns)
    tree_rules = export_text(model, feature_names=feature_names)
    
    print("\nDecision Tree Rules:")
    print(tree_rules)

    print("\n=== Machine Learning: Saving Evaluation Results ===")
    
    evaluation_path = "results/model_evaluation.txt"
    with open(evaluation_path, "w") as f:
        f.write("OOAIS Model Evaluation\n")
        f.write("======================\n")
        f.write(f"Model: {type(model).__name__}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n")
        print(f"Saved file: {evaluation_path}")

    print("\n=== Machine Learning: Saving Training Report ===")
    
    report_path = "reports/model_training_summary.txt"
    with open(report_path, "w") as f:
        f.write("OOAIS Model Training Summary\n")
        f.write("============================\n")
        f.write("Input datasets\n")
        f.write("--------------\n")
        f.write(f"{features_path}\n")
        f.write(f"{labels_path}\n\n")
        
        f.write("Dataset statistics\n")
        f.write("------------------\n")
        f.write(f"Number of samples: {len(X)}\n")
        f.write(f"Number of features: {len(df_features.columns)}\n\n")
        
        f.write("Model\n")
        f.write("-----\n")
        f.write(f"{type(model).__name__}\n\n")
        
        f.write("Train/Test split\n")
        f.write("----------------\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        
        f.write("Evaluation summary\n")
        f.write("------------------\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{conf_matrix}\n")
    
    print(f"Saved file: {report_path}")