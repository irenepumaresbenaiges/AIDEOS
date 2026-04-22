from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def validate_input_files():
    
    required_files = [
        "data/processed/model_features.csv",
        "data/processed/model_labels.csv"
    ]
    
    missing_files = []

    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("Error: missing required input file(s):")
        for missing in missing_files:
            print(f" - {missing}")
        raise SystemExit(1)


def load_data():
    
    features_path = "data/processed/model_features.csv"
    labels_path = "data/processed/model_labels.csv"
    
    print("=== Model Playground: Loading Data ===")
    print(f"Feature file: {features_path}")
    print(f"Label file: {labels_path}")
    
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    
    return features_df, labels_df


def inspect_data(features_df, labels_df):

    if features_df.empty:
        print("Error: Feature dataset is empty.") 
        raise SystemExit(1)
        
    if labels_df.empty:
        print("Error: Label dataset is empty.") 
        raise SystemExit(1)

    if len(features_df) != len(labels_df):
        print(f"Error: Row count mismatch. Features: {len(features_df)}, Labels: {len(labels_df)}") 
        raise SystemExit(1)
        
    if "anomaly_flag" not in labels_df.columns: 
        print("Error: 'anomaly_flag' column missing in labels dataset.") 
        raise SystemExit(1)

    num_samples = len(features_df)
    num_features = features_df.shape[1]
    feature_names = list(features_df.columns) 
    target_values = labels_df["anomaly_flag"].unique()
    
    print("\n=== Model Playground: Data Inspection ===")
    print(f"Number of samples: {num_samples}") 
    print(f"Number of features: {num_features}")
    print(f"Feature columns: {feature_names}") 
    print(f"Target values detected: {target_values.tolist()}")


def prepare_features_and_labels(features_df, labels_df):

    X = features_df.values
    y = labels_df["anomaly_flag"].astype(int).values
    
    print("\n=== Model Playground: Preparing Features and Labels ===") 
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return X, y


def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42
    )
    
    print("\n=== Model Playground: Train/Test Split ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def define_models():

    models = {
        "Decision Tree (baseline)": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    
    return models


def train_models(models, X_train, y_train):
    
    print("\n=== Model Playground: Training Models ===")
    
    trained_models = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        print(f"{model_name}: trained")
        
    return trained_models


def generate_predictions(trained_models, X_test):
    
    results = []
    
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        result = {
            "name": model_name,
            "model": model,
            "y_pred": y_pred
        }
        results.append(result)
        
    return results


def print_example_predictions(prediction_results, y_test, num_examples=5):

    print("\n=== Model Playground: Example Predictions ===")
    
    for i in range(num_examples):
        line = f"True: {y_test[i]}"
        for result in prediction_results:
            model_name = result["name"]
            y_pred = result["y_pred"]
            line += f" | {model_name}: {y_pred[i]}"
        print(line)


def compute_accuracy(prediction_results, y_test):

    print("\n=== Model Playground: Accuracy Comparison ===")
    
    for result in prediction_results:
        y_pred = result["y_pred"]
        accuracy = accuracy_score(y_test, y_pred)
        result["accuracy"] = accuracy
        print(f"{result['name']}: {accuracy:.4f}")
        
    return prediction_results


def compute_detailed_metrics(prediction_results, y_test):

    print("\n=== Model Playground: Detailed Evaluation ===")
    
    for result in prediction_results:
        y_pred = result["y_pred"]
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        result["confusion_matrix"] = cm
        result["classification_report"] = report
        
        print(f"\nModel: {result['name']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        
        print("\nConfusion Matrix:")
        print(cm)
        
        print("\nClass labels:")
        print("0 -> normal observation")
        print("1 -> anomaly")
        
        print("\nClassification Report:")
        print("------------------------------------------------------------")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<10}")
        print("------------------------------------------------------------")
        
        print(
            f"0 (normal)      "
            f"{report['0']['precision']:.2f}       "
            f"{report['0']['recall']:.2f}     "
            f"{report['0']['f1-score']:.2f}      "
            f"{int(report['0']['support'])}"
        )
        
        print(
            f"1 (anomaly)     "
            f"{report['1']['precision']:.2f}       "
            f"{report['1']['recall']:.2f}     "
            f"{report['1']['f1-score']:.2f}      "
            f"{int(report['1']['support'])}"
        )
        
        print("------------------------------------------------------------")
        
        print(
            f"Macro average   "
            f"{report['macro avg']['precision']:.2f}       "
            f"{report['macro avg']['recall']:.2f}     "
            f"{report['macro avg']['f1-score']:.2f}      "
            f"{int(report['macro avg']['support'])}"
        )
        print(
            f"Weighted average"
            f"{report['weighted avg']['precision']:.2f}       "
            f"{report['weighted avg']['recall']:.2f}     "
            f"{report['weighted avg']['f1-score']:.2f}      "
            f"{int(report['weighted avg']['support'])}"
        )
        
    return prediction_results


def rank_models(evaluation_results):
    
    sorted_results = sorted(
        evaluation_results,
        key=lambda result: result["accuracy"],
        reverse=True
    )
    
    print("\n=== Model Playground: Ranking ===")
    for index, result in enumerate(sorted_results, start=1):
        print(f"{index}. {result['name']} - {result['accuracy']:.4f}")
    
    return sorted_results


def run_experiments(X_train, X_test, y_train, y_test):

    print("\n=== Model Playground: Controlled Experiments ===")

    dt_accuracies = []
    for d in [2, 3, 5]:
        model = DecisionTreeClassifier(max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        dt_accuracies.append(acc)
        print(f"Decision Tree (max_depth={d}): {acc:.4f}")

    rf_accuracies = []
    for n in [5, 10, 50]:
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        rf_accuracies.append(acc)
        print(f"Random Forest (n_estimators={n}): {acc:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot([5, 10, 50], dt_accuracies, marker='o', color='b')
    plt.title("Decision Tree: Accuracy vs Depth")
    plt.xlabel("Max Depth")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot([5, 10, 50], rf_accuracies, marker='s', color='r')
    plt.title("Random Forest: Accuracy vs n_estimators")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {"dt_depths": dt_accuracies, "rf_estimators": rf_accuracies}


def save_experiment_summary(features_path, labels_path, X, X_train, X_test, ranked_models, experiment_results):

    print("\n=== Model Playground: Saving Summary ===")

    output_path = "reports/model_playground_summary.txt"
    with open(output_path, "w") as f:
        f.write("OOAIS Model Playground Summary\n")
        f.write("================================\n\n")

        f.write("Input datasets\n--------------\n")
        f.write(f"- {features_path}\n")
        f.write(f"- {labels_path}\n\n")

        f.write("Dataset statistics\n------------------\n")
        f.write(f"Number of samples: {X.shape[0]}\n")
        f.write(f"Number of features: {X.shape[1]}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Testing samples: {len(X_test)}\n\n")

        f.write("Compared models\n---------------\n")
        for result in ranked_models:
            f.write(f"- {result['name']}: {result['accuracy']:.4f}\n")
        f.write("\n")

        best_model = ranked_models[0]
        f.write("Best model\n----------\n")
        f.write(f"{best_model['name']} achieved the highest accuracy: {best_model['accuracy']:.4f}\n\n")

        f.write("Additional experiments\n----------------------\n")
        for d, acc in zip([2, 3, 5], experiment_results["dt_depths"]):
            f.write(f"- Decision Tree (max_depth={d}): {acc:.4f}\n")
            
        for n, acc in zip([5, 10, 50], experiment_results["rf_estimators"]):
            f.write(f"- Random Forest (n_estimators={n}): {acc:.4f}\n")
        f.write("\n")

        f.write("Conclusion\n----------\n")
        f.write(f"The best candidate for further experiments is {best_model['name']},\n")
        f.write("because it achieved the highest accuracy on the current test set.\n")

    print(f"Saved file: {output_path}")


def create_metric_plots(ranked_models):

    print("\n=== Model Playground: Saving Visualizations ===")
    
    model_names = [result["name"] for result in ranked_models]
    accuracies = [result["accuracy"] for result in ranked_models]
    
    precisions = [result["classification_report"]["1"]["precision"] for result in ranked_models]
    recalls = [result["classification_report"]["1"]["recall"] for result in ranked_models]
    f1_scores = [result["classification_report"]["1"]["f1-score"] for result in ranked_models]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].bar(model_names, accuracies)
    axes[0, 0].set_title("Accuracy")

    axes[0, 1].bar(model_names, precisions)
    axes[0, 1].set_title("Precision (Anomaly)")

    axes[1, 0].bar(model_names, recalls)
    axes[1, 0].set_title("Recall (Anomaly)")

    axes[1, 1].bar(model_names, f1_scores)
    axes[1, 1].set_title("F1-score (Anomaly)")

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel("Score")

    plt.tight_layout()
    plt.savefig("reports/model_comparison_panel.png")
    plt.close()
    
    print("Saved file: reports/model_comparison_panel.png")


if __name__ == "__main__":
    validate_input_files()
    features_df, labels_df = load_data()
    inspect_data(features_df, labels_df)
    X, y = prepare_features_and_labels(features_df, labels_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = define_models()
    trained_models = train_models(models, X_train, y_train)
    prediction_results = generate_predictions(trained_models, X_test)
    print_example_predictions(prediction_results, y_test)
    prediction_results = compute_accuracy(prediction_results, y_test)
    prediction_results = compute_detailed_metrics(prediction_results, y_test)
    sorted_results = rank_models(prediction_results)
    experiment_results = run_experiments(X_train, X_test, y_train, y_test)
    save_experiment_summary(
        "data/processed/model_features.csv",
        "data/processed/model_labels.csv",
        X, X_train, X_test, 
        sorted_results, 
        experiment_results
    )
    create_metric_plots(sorted_results)
