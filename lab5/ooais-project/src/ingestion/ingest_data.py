import json
import csv
import os

DATASET_PATH = "data/raw/orbital_observations.csv"
METADATA_PATH = "data/raw/metadata.json"

if __name__ == "__main__":
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    rows = []
    dataset_columns = []
    with open(DATASET_PATH, 'r') as f:
        reader = csv.DictReader(f)
        dataset_columns = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"Dataset name: {metadata['dataset_name']}")
    print(f"Records loaded: {len(rows)}")
    print(f"Dataset columns: {dataset_columns}")
    metadata_columns = metadata["columns"]
    print(f"Metadata columns: {metadata_columns}")

    if dataset_columns == metadata_columns:
        print("Column validation: OK")
        column_validation_status = "OK"
    else:
        print("Column validation: MISMATCH")
        print(f"Expected: {metadata_columns}")
        print(f"Actual: {dataset_columns}")

    actual_count = len(rows)
    expected_count = metadata["num_records"]
    if actual_count == expected_count:
        print("Record count: OK")
    else:
        print("Record count: MISMATCH")
        print(f"Expected: {expected_count}")
        print(f"Actual: {actual_count}")

    valid_records = []
    invalid_records = []
    for row in rows:
        if row['temperature'] == 'INVALID':
            invalid_records.append(row)
        else:
            valid_records.append(row)

    expected_invalid = metadata["invalid_records"]
    if len(invalid_records) == expected_invalid:
        print("Invalid records count: OK")
    else:
        print("Invalid records count: MISMATCH")
        
    print(f"Valid records: {len(valid_records)}")
    print(f"Invalid records: {len(invalid_records)}")

    PROCESSED_DIR = 'data/processed/'
    valid_file_path = os.path.join(PROCESSED_DIR, 'observations_valid.csv')
    invalid_file_path = os.path.join(PROCESSED_DIR, 'observations_invalid.csv')
    
    with open(valid_file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=dataset_columns)
        writer.writeheader()
        writer.writerows(valid_records)
        
    with open(invalid_file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=dataset_columns)
        writer.writeheader()
        writer.writerows(invalid_records)

    model_input_path = os.path.join(PROCESSED_DIR, 'model_input.csv')
    
    feature_cols = metadata["feature_columns"]
    model_data = []
    for record in valid_records:
        filtered_row = {key: record[key] for key in feature_cols}
        model_data.append(filtered_row)
        
    with open(model_input_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=feature_cols)
        writer.writeheader()
        writer.writerows(model_data)

    REPORTS_DIR = 'reports/'
    summary_path = os.path.join(REPORTS_DIR, 'ingestion_summary.txt')

    column_validation_status = "OK" if dataset_columns == metadata_columns else "MISMATCH"
    record_count_status = "OK" if len(rows) == metadata["num_records"] else "MISMATCH"

    with open(summary_path, 'w') as f:
        f.write(f"Dataset: {metadata['dataset_name']}\n")
        f.write(f"Records loaded: {len(rows)}\n")
        f.write(f"Expected records: {metadata['num_records']}\n\n")
        f.write(f"Column validation: {column_validation_status}\n")
        f.write(f"Record count validation: {record_count_status}\n\n")
        f.write(f"Valid records: {len(valid_records)}\n")
        f.write(f"Invalid records: {len(invalid_records)}\n\n")
        f.write("Generated files:\n")
        f.write(f"- {valid_file_path}\n")
        f.write(f"- {invalid_file_path}\n")
        f.write(f"- {model_input_path}\n")

    feature_cols_metadata = metadata.get("feature_columns", [])
    target_col_metadata = metadata.get("target_column", "")
    missing_features = [col for col in feature_cols_metadata if col not in dataset_columns]
    if not missing_features:
        print("Feature validation: OK")
    else:
        print(f"Missing feature columns: {missing_features}")

    if target_col_metadata in dataset_columns:
        print("Target validation: OK")
    else:
        print(f"Target column missing: {target_col_metadata}")