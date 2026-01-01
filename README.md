# Hand Gesture Music Controller

A Python-based application that uses computer vision to control music synthesis via hand gestures. This project integrates MediaPipe for hand tracking, scikit-learn for gesture recognition, and SuperCollider for sound synthesis.

## Features

- **Real-time Hand Gesture Recognition**: Detects and classifies hand gestures using a trained Random Forest model.
- **Music Control**: Maps hand gestures and positions to musical parameters (pitch, volume, etc.).
- **Multiple Modes**:
  - **Continuous Pitch Mode**: Control pitch continuously with hand movement.
  - **Discrete Pitch Mode**: Play discrete MIDI notes.
  - **Track Overlay Mode**: Layer different sounds.
- **GUI Interface**: Visual feedback of the camera feed and current mode.
- **OSC Communication**: Sends control data to SuperCollider via Open Sound Control (OSC).

## Prerequisites

- Python 3.8+
- [SuperCollider](https://supercollider.github.io/) (for sound synthesis)
- A webcam

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd hand_gesture
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure SuperCollider is installed and running.

## Usage

### 1. Sound Synthesis Setup

Before running the Python application, you need to start the SuperCollider server and load the synth definitions.
- Open `src/sound_syn.scd` in SuperCollider.
- Execute the code to boot the server and define the synths.

### 2. Running the Application

To start the main application with the GUI:

```bash
python src/RUNME.py
```
Or directly:
```bash
python src/gui_version.py
```

### 3. Training the Model (Optional)

If you want to retrain the gesture recognition model with your own dataset:

1. Organize your dataset in a directory where each subdirectory is a gesture class containing video files.
2. Run the training script:

```bash
python src/train_model.py --data_dir path/to/your/dataset --output src/gesture_model.pkl
```

Arguments:
- `--data_dir`: Path to the dataset directory (required).
- `--output`: Path to save the trained model (default: `gesture_model.pkl`).
- `--test_size`: Fraction of data to use for testing (default: 0.2).
- `--n_estimators`: Number of trees in the Random Forest (default: 100).
- `--frame_skip`: Process every Nth frame to speed up training (default: 1).

## Project Structure

- `src/`: Source code directory.
  - `train_model.py`: Script for training the gesture recognition model.
  - `gui_version.py`: Main application with GUI.
  - `RUNME.py`: Entry point script.
  - `sound_syn.py`: Python module for OSC communication.
  - `sound_syn.scd`: SuperCollider code for sound synthesis.
  - `global_variable.py`: Configuration and global variables.
  - `music_source/`: Contains MIDI files and related scripts.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## Gestures

The model is trained to recognize the following gestures:
- Open Hand
- Fist
- Cross
- One
- Two
- Three
- Thumb
