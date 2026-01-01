"""
Hand Gesture Model Training Script

This script trains a Random Forest classifier to recognize hand gestures using
MediaPipe hand landmark detection. It processes video files or images from a 
dataset directory structure where each subdirectory represents a gesture class.

Usage:
    python train_model.py --data_dir <path_to_dataset> --output <model_path>
    
Example:
    python train_model.py --data_dir ../data/handgestures --output gesture_model.pkl
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Initialize MediaPipe
mp_hands = mp.solutions.hands


def extract_landmarks(image):
    """
    Extract hand landmarks from an image using MediaPipe.
    
    Args:
        image: BGR image from OpenCV
        
    Returns:
        numpy array of shape (84,) containing x,y coordinates for 21 landmarks × 2 hands
        Returns zeros if no hands detected
    """
    with mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=2, 
        min_detection_confidence=0.7
    ) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
            
            # If only one hand is detected, pad with zeros
            if len(results.multi_hand_landmarks) == 1:
                landmarks.extend([0] * (21 * 2))  # 21 landmarks, each with x and y
            
            return np.array(landmarks).flatten()
    
    # If no hands are detected, return a zero array
    return np.zeros(21 * 2 * 2)  # 21 landmarks × 2 coords × 2 hands = 84 features


def load_dataset_from_videos(data_dir, frame_skip=1, verbose=True):
    """
    Load dataset from video files organized in subdirectories by gesture class.
    
    Args:
        data_dir: Root directory containing subdirectories for each gesture
        frame_skip: Process every Nth frame (1 = all frames, 5 = every 5th frame)
        verbose: Print progress information
        
    Returns:
        X: numpy array of features (n_samples, 84)
        y: numpy array of labels (n_samples,)
    """
    landmarks_list = []
    labels_list = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    gesture_classes = sorted(os.listdir(data_dir))
    
    if verbose:
        print(f"\nFound {len(gesture_classes)} gesture classes: {gesture_classes}")
        print(f"Processing videos with frame_skip={frame_skip}...\n")
    
    for label in gesture_classes:
        label_dir = os.path.join(data_dir, label)
        
        if not os.path.isdir(label_dir):
            continue
        
        if verbose:
            print(f"Processing gesture: {label}")
        
        video_files = [f for f in os.listdir(label_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        video_count = 0
        frame_total = 0
        
        for video_file in video_files:
            video_path = os.path.join(label_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"  Warning: Could not open video {video_file}")
                continue
            
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for speed if requested
                if frame_count % frame_skip != 0:
                    continue
                
                landmarks = extract_landmarks(frame)
                
                # Only add samples where landmarks were detected (non-zero)
                if np.any(landmarks):
                    landmarks_list.append(landmarks)
                    labels_list.append(label)
                    frame_total += 1
            
            cap.release()
            video_count += 1
        
        if verbose:
            print(f"  Processed {video_count} videos, extracted {frame_total} frames\n")
    
    X = np.array(landmarks_list)
    y = np.array(labels_list)
    
    if verbose:
        print(f"Dataset loaded successfully!")
        print(f"Total frames processed: {len(X)}")
        print(f"Feature shape: {X.shape}")
        print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}\n")
    
    return X, y


def train_model(X_train, y_train, n_estimators=100, random_state=42, verbose=True):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels (encoded)
        n_estimators: Number of trees in the forest
        random_state: Random seed for reproducibility
        verbose: Print training information
        
    Returns:
        Trained RandomForestClassifier model
    """
    if verbose:
        print(f"Training Random Forest with {n_estimators} estimators...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=random_state,
        verbose=1 if verbose else 0
    )
    model.fit(X_train, y_train)
    
    if verbose:
        print("Training completed!\n")
    
    return model


def evaluate_model(model, X_test, y_test, label_encoder, verbose=True):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels (encoded)
        label_encoder: LabelEncoder to decode class names
        verbose: Print detailed evaluation metrics
        
    Returns:
        accuracy: Test accuracy score
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if verbose:
        print("=" * 60)
        print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("=" * 60)
        print("\nClassification Report:")
        print(classification_report(
            y_test, 
            y_pred, 
            target_names=label_encoder.classes_
        ))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print()
    
    return accuracy


def save_model(model, label_encoder, output_path, verbose=True):
    """
    Save the trained model and label encoder to disk.
    
    Args:
        model: Trained classifier
        label_encoder: LabelEncoder used for labels
        output_path: Path to save the model file
        verbose: Print save information
    """
    # Save both model and label encoder together
    model_data = {
        'model': model,
        'label_encoder': label_encoder
    }
    
    joblib.dump(model_data, output_path)
    
    if verbose:
        print(f"Model and label encoder saved to: {output_path}")
        print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train a hand gesture recognition model using Random Forest',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to dataset directory containing subdirectories for each gesture class'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='gesture_model.pkl',
        help='Output path for the trained model file'
    )
    
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of dataset to use for testing (0.0 to 1.0)'
    )
    
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help='Number of trees in the Random Forest'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--frame_skip',
        type=int,
        default=1,
        help='Process every Nth frame from videos (e.g., 5 = every 5th frame)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if verbose:
        print("\n" + "=" * 60)
        print("Hand Gesture Model Training")
        print("=" * 60)
        print(f"Data directory: {args.data_dir}")
        print(f"Output model: {args.output}")
        print(f"Test size: {args.test_size}")
        print(f"N estimators: {args.n_estimators}")
        print(f"Random state: {args.random_state}")
        print(f"Frame skip: {args.frame_skip}")
        print("=" * 60)
    
    try:
        # Step 1: Load dataset
        X, y = load_dataset_from_videos(
            args.data_dir, 
            frame_skip=args.frame_skip,
            verbose=verbose
        )
        
        if len(X) == 0:
            print("Error: No valid samples extracted from dataset!")
            sys.exit(1)
        
        # Step 2: Encode labels
        if verbose:
            print("Encoding labels...")
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Step 3: Split dataset
        if verbose:
            print(f"Splitting dataset (train={1-args.test_size:.0%}, test={args.test_size:.0%})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=args.test_size, 
            random_state=args.random_state,
            stratify=y_encoded
        )
        
        if verbose:
            print(f"Training set size: {len(X_train)}")
            print(f"Test set size: {len(X_test)}\n")
        
        # Step 4: Train model
        model = train_model(
            X_train, y_train,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            verbose=verbose
        )
        
        # Step 5: Evaluate model
        accuracy = evaluate_model(
            model, X_test, y_test, 
            label_encoder,
            verbose=verbose
        )
        
        # Step 6: Save model
        save_model(model, label_encoder, args.output, verbose=verbose)
        
        if verbose:
            print("=" * 60)
            print("Training completed successfully!")
            print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
