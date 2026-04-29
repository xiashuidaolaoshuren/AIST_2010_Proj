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

- **Python 3.12** (`requires-python` in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) (recommended)
- [SuperCollider](https://supercollider.github.io/) (for sound synthesis)
- A webcam

**Note:** `mediapipe` is pinned to `0.10.14` so the classic `mp.solutions.hands` API used by this project remains available. Newer MediaPipe wheels dropped `solutions` in favor of `tasks`.

## Trained model

The repository does **not** include `gesture_model.pkl`. Without it, the app still runs: you get the camera and hand landmarks; the gesture label shows `(No model)` until you add a model.

Train your own (one subdirectory per gesture class, videos inside each):

```bash
uv sync
uv run python src/train_model.py --data_dir path/to/your/dataset --output src/gesture_model.pkl
```

Or copy an existing `gesture_model.pkl` to `src/gesture_model.pkl`.

## Installation

```bash
git clone <repository-url>
cd hand_gesture
uv sync
```

Optional pip install aligned with `pyproject.toml`:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Sound synthesis setup

Open `src/sound_syn.scd` in SuperCollider and execute it to boot the server and load synths.

### 2. Running the application

```bash
uv run hand-gesture-gui
```

Alternatives:

```bash
uv run python src/RUNME.py
uv run python src/gui_version.py
```

### 3. Training

```bash
uv run python src/train_model.py --help
```

Common arguments: `--data_dir` (required), `--output` (default `gesture_model.pkl`), `--test_size`, `--n_estimators`, `--frame_skip`.

## Project structure

- `src/gui_version.py` — main GUI and inference loop
- `src/train_model.py` — training CLI
- `src/RUNME.py` — launches `gui_version.py`
- `src/hand_gesture_app/` — `hand-gesture-gui` entrypoint
- `src/sound_syn.py`, `src/sound_syn.scd` — OSC / SuperCollider
- `pyproject.toml`, `uv.lock` — project and locked dependencies
- `requirements.txt` — pip-friendly constraint list

## Gestures

Open Hand, Fist, Cross, One, Two, Three, Thumb
