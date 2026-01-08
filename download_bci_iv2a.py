#!/usr/bin/env python3
"""
Download and preprocess BCI Competition IV 2a dataset for STEEGFormer-vICML
Uses MOABB to download the data and converts it to the expected format
"""

import numpy as np
import pickle
import os
from pathlib import Path
from moabb.datasets import BNCI2014_001  # BCI Competition IV 2a
from moabb.paradigms import MotorImagery
from scipy.signal import resample

# Configuration
OUTPUT_DIR = "/teamspace/studios/this_studio/STEEGFormer-vICML/data/bci_iv2a"
TARGET_FS = 256  # Target sampling rate for STEEGFormer
EPOCH_LENGTH = 4  # seconds

def download_and_preprocess_bci_iv2a():
    """Download BCI Competition IV 2a and preprocess for STEEGFormer"""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize MOABB dataset (BCI Competition IV 2a = BNCI2014_001)
    print("Downloading BCI Competition IV 2a dataset...")
    dataset = BNCI2014_001()

    # Initialize paradigm for motor imagery
    paradigm = MotorImagery(
        events=['left_hand', 'right_hand', 'feet', 'tongue'],  # 4 classes
        n_classes=4,
        fmin=0.5,  # Will bandpass the data
        fmax=100,
        channels=None,  # Use all available channels
        resample=TARGET_FS
    )

    # Process each subject
    subjects = dataset.subject_list
    print(f"Processing {len(subjects)} subjects...")

    for subject_id in subjects:
        print(f"\nProcessing subject {subject_id}...")

        try:
            # Get data for this subject
            X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])

            # X shape: (n_trials, n_channels, n_times)
            # labels: class labels
            # metadata: DataFrame with session and run information

            # Split into train and test based on session
            # Session 'session_T' is training, 'session_E' is evaluation/test
            train_mask = metadata['session'] == 'session_T'
            test_mask = metadata['session'] == 'session_E'

            trainX = X[train_mask]
            trainY = labels[train_mask]
            testX = X[test_mask]
            testY = labels[test_mask]

            # Convert labels to 0-indexed (MOABB might use different encoding)
            # Map: left_hand=0, right_hand=1, feet=2, tongue=3
            unique_labels = np.unique(np.concatenate([trainY, testY]))
            label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            trainY = np.array([label_map[label] for label in trainY])
            testY = np.array([label_map[label] for label in testY])

            # Prepare data dict
            subject_data = {
                'trainX': trainX,
                'trainY': trainY,
                'testX': testX,
                'testY': testY
            }

            # Save as pickle
            output_file = os.path.join(OUTPUT_DIR, f"sub-{subject_id:02d}.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(subject_data, f)

            print(f"  Saved: {output_file}")
            print(f"  Train shape: {trainX.shape}, Test shape: {testX.shape}")
            print(f"  Train labels: {np.unique(trainY)}, Test labels: {np.unique(testY)}")

        except Exception as e:
            print(f"  Error processing subject {subject_id}: {e}")
            continue

    print(f"\n✓ Dataset downloaded and preprocessed!")
    print(f"✓ Saved to: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"1. Update the 'data_dir' in benchmark/neural_networks/util/dataset_specs.yaml")
    print(f"   Change line 107 to: data_dir: {OUTPUT_DIR}")
    print(f"2. Run the benchmark with: python benchmark/neural_networks/wandb_downstream_evaluation.py")

if __name__ == "__main__":
    download_and_preprocess_bci_iv2a()
