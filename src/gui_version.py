import cv2
import mediapipe as mp
import os
import sys
import joblib
import numpy as np
from pythonosc import udp_client
import sounddevice as sd
import wavio as wv
import time
import warnings
import tkinter as tk
from sound_syn import sound_synth
from global_variable import *
from PIL import Image, ImageTk  # For displaying OpenCV frames in Tkinter
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def load_gesture_model_bundle(path: str):
    """Load (model, label_encoder) from joblib files in supported layouts."""
    payload = joblib.load(path)
    if isinstance(payload, dict):
        if "model" in payload and "label_encoder" in payload:
            return payload["model"], payload["label_encoder"]
        raise ValueError(
            "Model dict must contain 'model' and 'label_encoder' "
            f"(found keys: {list(payload.keys())!r}). "
            "Use output from src/train_model.py or the training notebook save cell."
        )
    if isinstance(payload, (list, tuple)):
        if len(payload) == 2:
            return payload[0], payload[1]
        raise ValueError(
            "Expected a 2-element list/tuple [model, label_encoder] "
            f"(got length {len(payload)})."
        )
    if hasattr(payload, "predict") and hasattr(payload, "classes_"):
        from sklearn.preprocessing import LabelEncoder

        m = payload
        class_vals = np.asarray(m.classes_)
        if np.issubdtype(class_vals.dtype, np.number) and class_vals.dtype != object:
            raise ValueError(
                "This pickle contains only the classifier (no LabelEncoder), "
                "and it was trained on encoded integers, so gesture names cannot be recovered. "
                "Re-save with both objects, e.g. "
                "joblib.dump({'model': model, 'label_encoder': label_encoder}, path) "
                "or run src/train_model.py to write gesture_model.pkl."
            )
        le = LabelEncoder()
        le.classes_ = np.asarray(class_vals, dtype=object)
        return m, le
    raise ValueError(f"Unsupported gesture_model.pkl contents: {type(payload)!r}")


# Global variables
midi_interval = 0.05  # Value of y to increase/decrease 1 MIDI note
mode_id = ["Frequency Mode", "Discrete Midi Mode", "Composing Mode"]
instrument = ["Sine oscillator", "Hand flute"]

# Initialize the OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", 57120)

# Load the gesture model (optional — repo does not ship gesture_model.pkl)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(script_dir, "gesture_model.pkl")
model = None
label_encoder = None
if os.path.isfile(model_file_path):
    try:
        model, label_encoder = load_gesture_model_bundle(model_file_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Warning: could not load gesture model ({exc}). "
            "Running without gesture control; fix or remove the file and retrain if needed.",
            file=sys.stderr,
        )
        model, label_encoder = None, None
else:
    print(
        f"No model at {model_file_path}. "
        "Camera and hand preview will work; add gesture_model.pkl to enable control. "
        "See README (train with src/train_model.py).",
    )
warnings.simplefilter("ignore")

# Mediapipe variables
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Tkinter GUI variables
current_mode = 0
current_instrument = 0


def _open_working_capture():
    """Return a VideoCapture that delivers at least one good frame, or None."""

    def _try_open(idx, api):
        try:
            c = cv2.VideoCapture(idx, api) if api is not None else cv2.VideoCapture(idx)
            if not c.isOpened():
                return None
            for _ in range(5):
                ok, fr = c.read()
                if ok and fr is not None and fr.size > 0:
                    return c
                time.sleep(0.05)
            c.release()
        except Exception:
            pass
        return None

    tries = []
    if sys.platform == "win32":
        tries.append((0, cv2.CAP_DSHOW))
    tries.append((0, cv2.CAP_ANY))
    for idx, api in tries:
        cap_local = _try_open(idx, api)
        if cap_local is not None:
            return cap_local
    return _try_open(0, None)


cap = _open_working_capture()
if cap is None or not cap.isOpened():
    print(
        "Warning: no working webcam; GUI will open without a live preview. "
        "Connect a camera or close other apps using it, then restart.",
        file=sys.stderr,
    )
    cap = None
gesture_text = "(No model)" if model is None else "None"
midinote = 69  # Default MIDI note
mcp_y = None  # Middle finger MCP Y position for tracking movement

recording = False
recording_start_time = None
recording_file_path = None
try:
    input_device_info = sd.query_devices(kind="input")
    input_channels = input_device_info["max_input_channels"]
except Exception as exc:  # noqa: BLE001
    print(
        f"Warning: could not query default audio input ({exc}). "
        "Composing-mode recording may not work.",
        file=sys.stderr,
    )
    input_device_info = None
    input_channels = 1
recording_data = None

os.makedirs(os.path.join(script_dir, "recording"), exist_ok=True)

init = False
key_pressed_s = False

CANVAS_WIDTH = 800
CANVAS_HEIGHT = 480


def switch_mode():
    """Switch between modes."""
    global current_mode
    current_mode = (current_mode + 1) % len(mode_id)
    mode_label.config(text=f"Mode: {mode_id[current_mode]}")


def switch_instrument():
    """Switch between instruments."""
    global current_instrument
    current_instrument = (current_instrument + 1) % len(instrument)
    instrument_label.config(text=f"Instrument: {instrument[current_instrument]}")


def calculate_distance(thumb_tip, index_finger_tip, middle_finger_tip):
    if middle_finger_tip == 0:
        return np.sqrt(
            (thumb_tip[0] - index_finger_tip[0]) ** 2
            + (thumb_tip[1] - index_finger_tip[1]) ** 2
        )
    return np.sqrt(
        (thumb_tip[0] - middle_finger_tip[0]) ** 2
        + (thumb_tip[1] - middle_finger_tip[1]) ** 2
    )


def update_frame():
    """Update the video frame and process gestures."""
    global gesture_text, recording, current_mode, midinote, mcp_y
    global recording_start_time, recording_file_path
    global input_device_info, input_channels
    global recording_data, init, key_pressed_s

    if cap is None:
        video_canvas.delete("all")
        video_canvas.create_rectangle(
            0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, fill="#1a1a1a", outline=""
        )
        video_canvas.create_text(
            CANVAS_WIDTH // 2,
            CANVAS_HEIGHT // 2,
            text="No camera\n\nOpen without gesture_model.pkl is OK;\nadd src/gesture_model.pkl after training.",
            fill="#eeeeee",
            font=("Arial", 14),
            justify=tk.CENTER,
        )
        gesture_label.config(text=f"Gesture: {gesture_text}")
        root.after(200, update_frame)
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video.")
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks_list = []
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            landmarks = np.array(
                [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            ).flatten()
            hand_landmarks_list.append(landmarks)

        if model is None:
            gesture_text = "(No model)"
        else:
            if len(hand_landmarks_list) == 2:
                combined_landmarks = np.concatenate(hand_landmarks_list).reshape(
                    1, -1
                )
            else:
                combined_landmarks = np.concatenate(
                    [hand_landmarks_list[0], np.zeros(42)]
                ).reshape(1, -1)

            raw_pred = model.predict(combined_landmarks)[0]
            if isinstance(raw_pred, str | np.str_):
                gesture_text = str(raw_pred)
            else:
                gesture_text = label_encoder.inverse_transform([int(raw_pred)])[0]
            middle_mcp = results.multi_hand_landmarks[0].landmark[
                mp_hands.HandLandmark.MIDDLE_FINGER_MCP
            ]

            if current_mode == 0:  # Frequency Mode
                middle_mcp = results.multi_hand_landmarks[0].landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                ]
                freq = np.clip((1 - middle_mcp.y) * 1980 + 20, 20, 2000)
                sound_synth(
                    0 if gesture_text != "Open Hand" else freq,
                    current_mode,
                    current_instrument,
                )
            elif current_mode == 1:  # Discrete Midi Mode
                thumb_tip = results.multi_hand_landmarks[0].landmark[
                    mp_hands.HandLandmark.THUMB_TIP
                ]
                thumb_tip_coords = (
                    int(thumb_tip.x * frame.shape[1]),
                    int(thumb_tip.y * frame.shape[0]),
                )
                middle_finger_tip = results.multi_hand_landmarks[0].landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                ]
                middle_finger_tip_coords = (
                    int(middle_finger_tip.x * frame.shape[1]),
                    int(middle_finger_tip.y * frame.shape[0]),
                )

                thumb_middle_distance = calculate_distance(
                    thumb_tip_coords, 0, middle_finger_tip_coords
                )

                if thumb_middle_distance < 40:
                    if not key_pressed_s:
                        mcp_y = middle_mcp.y
                        midinote = 69
                        print("Position Initialized")
                        init = True
                        key_pressed_s = True
                    else:
                        key_pressed_s = False

                if init:
                    change_of_y = -(middle_mcp.y - mcp_y)
                    change_of_midi = int(change_of_y / midi_interval)
                    if abs(change_of_midi) > 0:
                        midinote += change_of_midi
                        mcp_y = middle_mcp.y
                    match gesture_text:
                        case "Fist":
                            sound_synth(0, current_mode, current_instrument)
                        case "Open Hand":
                            index_finger_tip = results.multi_hand_landmarks[
                                0
                            ].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                            index_finger_tip_coords = (
                                int(index_finger_tip.x * frame.shape[1]),
                                int(index_finger_tip.y * frame.shape[0]),
                            )

                            thumb_index_distance = calculate_distance(
                                thumb_tip_coords, index_finger_tip_coords, 0
                            )
                            if thumb_index_distance < 40:
                                sound_synth(
                                    midinote, current_mode, current_instrument
                                )
                            else:
                                sound_synth(0, current_mode, current_instrument)

            elif current_mode == 2:  # Composing Mode
                if gesture_text == "One":
                    if not recording:
                        recording = True
                        recording_start_time = time.time()
                        recording_file_path = os.path.join(
                            script_dir,
                            "recording",
                            f"recording_{int(recording_start_time)}.wav",
                        )
                        print(f"Recording started: {recording_file_path}")
                        recording_data = sd.rec(
                            int(10 * 44100),
                            samplerate=44100,
                            channels=1,
                            dtype="float32",
                        )
                else:
                    if recording:
                        recording = False
                        sd.stop()
                        recording_duration = time.time() - recording_start_time
                        print(f"Recording stopped: {recording_file_path}")

                        wv.write(
                            recording_file_path,
                            recording_data[
                                : int((recording_duration - 0.5) * 44100)
                            ],
                            44100,
                            sampwidth=2,
                        )
                        time.sleep(0.5)
                        client.send_message("/playfile", recording_file_path)

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    video_canvas.imgtk = imgtk

    gesture_label.config(text=f"Gesture: {gesture_text}")

    root.after(10, update_frame)


hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
)

root = tk.Tk()
root.title("Hand Gesture Music Controller")

video_canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
video_canvas.pack()

mode_label = tk.Label(root, text=f"Mode: {mode_id[current_mode]}", font=("Arial", 14))
mode_label.pack()

instrument_label = tk.Label(
    root, text=f"Instrument: {instrument[current_instrument]}", font=("Arial", 14)
)
instrument_label.pack()

gesture_label = tk.Label(root, text="Gesture: None", font=("Arial", 14))
gesture_label.pack()

mode_button = tk.Button(root, text="Switch Mode", command=switch_mode)
mode_button.pack(side=tk.LEFT, padx=10)

instrument_button = tk.Button(
    root, text="Switch Instrument", command=switch_instrument
)
instrument_button.pack(side=tk.LEFT, padx=10)

update_frame()

root.mainloop()

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
