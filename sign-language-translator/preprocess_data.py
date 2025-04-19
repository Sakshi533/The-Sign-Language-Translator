# sign-language-translator/preprocess_data.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import os
import argparse

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Default data directory is 'data' relative to the script
default_base_data_dir = os.path.join(script_dir, "data")
# Default output directory is 'preprocessed' inside the default data directory
default_output_dir = os.path.join(default_base_data_dir, "preprocessed")


def preprocess_data(base_data_dir, output_dir):
    """
    Loads raw sign language MNIST data, preprocesses it, and saves the
    processed arrays to a compressed NPZ file.

    Args:
        base_data_dir (str): Absolute path to the directory containing the CSV files.
        output_dir (str): Absolute path to the directory where the
                          'sign_mnist_preprocessed.npz' file will be saved.
    """
    print("--- Starting Data Preprocessing ---")
    print(f"Using data directory: {base_data_dir}")
    print(f"Output directory: {output_dir}")

    train_csv_path = os.path.join(base_data_dir, "sign_mnist_train.csv")
    test_csv_path = os.path.join(base_data_dir, "sign_mnist_test.csv")

    # Load the datasets
    try:
        print(f"Loading datasets...")
        print(f"Training data path: {train_csv_path}")
        print(f"Testing data path: {test_csv_path}")
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
        print("Datasets loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        print(f"Please ensure the following files exist:")
        print(f"- {train_csv_path}")
        print(f"- {test_csv_path}")
        return # Exit the function on error

    print("Raw Training data shape:", train_df.shape)
    print("Raw Testing data shape:", test_df.shape)

    # Separate labels and features
    y_train_raw = train_df['label'].values
    y_test_raw = test_df['label'].values
    x_train_raw = train_df.drop(['label'], axis=1).values
    x_test_raw = test_df.drop(['label'], axis=1).values
    print("Features and labels separated.")

    # Normalize pixel values
    x_train = x_train_raw / 255.0
    x_test = x_test_raw / 255.0
    print("Pixel values normalized.")

    # Reshape data for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    print(f"Data reshaped to: x_train={x_train.shape}, x_test={x_test.shape}")

    # Binarize labels
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train_raw)
    y_test = label_binarizer.transform(y_test_raw)
    print("Labels binarized (one-hot encoded).")
    print("Number of classes:", len(label_binarizer.classes_))

    # --- Save preprocessed data ---
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_file_path = os.path.join(output_dir, "sign_mnist_preprocessed.npz")

    print(f"\nSaving preprocessed data to: {preprocessed_file_path} ...")
    np.savez_compressed(
        preprocessed_file_path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        classes=label_binarizer.classes_
    )
    print("Preprocessed data saved successfully.")
    print("\n--- Data Preprocessing Complete ---")


if __name__ == "__main__":
    # Allows running the script from the command line
    parser = argparse.ArgumentParser(description="Preprocess Sign Language MNIST data.")
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default=default_base_data_dir, # Use calculated default absolute path
        help="Directory containing raw train/test CSV data."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir, # Use calculated default absolute path
        help="Directory to save the processed .npz file."
    )
    args = parser.parse_args()

    # Pass the potentially overridden absolute paths to the function
    preprocess_data(args.base_data_dir, args.output_dir)
