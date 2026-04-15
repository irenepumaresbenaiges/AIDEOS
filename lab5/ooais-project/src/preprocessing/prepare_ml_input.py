import pandas as pd

if __name__ == "__main__":

    input_path = "data/processed/observations_valid.csv"
    df = pd.read_csv(input_path)

    initial_count = len(df)
    numeric_cols = ['temperature', 'velocity', 'altitude', 'signal_strength']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=numeric_cols)
    df = df[df['altitude'] >= 0]

    accepted_count = len(df)
    rejected_count = initial_count - accepted_count

    print("=== ML Input Preparation: Loading and Conversion ===")
    print(f"Input file: {input_path}")
    print(f"Records loaded: {initial_count}")
    print(f"Records accepted: {accepted_count}")
    print(f"Records rejected: {rejected_count}")

    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_max > col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[col] = 0.0

    print("\n=== ML Input Preparation: Normalization ===")
    print("Normalization completed successfully.")
    print("All selected numerical features are in range [0,1].")

    df['temperature_velocity_interaction'] = df['temperature'] * df['velocity']
    df['altitude_signal_ratio'] = df['altitude'] / (df['signal_strength'] + 0.0001)

    print("\n=== ML Input Preparation: Derived Features ===")
    print("New features added:")
    print("- temperature_velocity_interaction")
    print("- altitude_signal_ratio")
    
    example_row = df.iloc[0].to_dict()
    print("Example record (extended):")
    print(example_row)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_normalized'] = df['timestamp'].dt.hour / 24

    print("\n=== ML Input Preparation: Temporal Features ===")
    print("New feature added:")
    print("- hour_normalized")
    
    example_row_temporal = df.iloc[0].to_dict()
    example_row_temporal['timestamp'] = example_row_temporal['timestamp'].isoformat()
    print("Example record (extended):")
    print(example_row_temporal)

    selected_columns = [
        'temperature',
        'velocity',
        'altitude',
        'signal_strength',
        'temperature_velocity_interaction',
        'altitude_signal_ratio',
        'hour_normalized'
    ]
    df_features = df[selected_columns].copy()

    print("\n=== ML Input Preparation: Feature Selection ===")
    print("Selected features:")
    for col in selected_columns:
        print(f"- {col}")
    
    print("Example record (final):")
    print(df_features.iloc[0].to_dict())

    df_labels = df[['anomaly_flag']].copy()

    features = "data/processed/model_features.csv"
    labels = "data/processed/model_labels.csv"
    df_features.to_csv(features, index=False)
    df_labels.to_csv(labels, index=False)

    print("\n=== ML Input Preparation: Saving Outputs ===")
    print(f"Saved file: {features}")
    print(f"Saved file: {labels}")
    print(f"Number of records: {len(df_features)}")
    print(f"Number of features: {len(df_features.columns)}")
    
    print("Example label record:")
    print(df_labels.iloc[0].to_dict())