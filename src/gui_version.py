import cv2
import mediapipe as mp
import os
import joblib
import numpy as np
from pythonosc import udp_client
import sounddevice as sd
import wavio as wv
import time
import warnings
import tkinter as tk
import soundfile as sf
from scipy.io.wavfile import write
from sound_syn import sound_synth
from global_variable import *
from PIL import Image, ImageTk  # For displaying OpenCV frames in Tkinter

# Global variables
midi_interval = 0.05  # Value of y to increase/decrease 1 MIDI note
mode_id = ["Frequency Mode", "Discrete Midi Mode", "Composing Mode"]
instrument = ["Sine oscillator", "Hand flute"]

# Initialize the OSC client
client = udp_client.SimpleUDPClient("127.0.0.1", 57120)

# Load the gesture model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(script_dir, "gesture_model.pkl")
model_data = joblib.load(model_file_path)
model = model_data['model']
label_encoder = model_data['label_encoder']
warnings.simplefilter("ignore")

# Mediapipe variables
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Tkinter GUI variables
current_mode = 0
current_instrument = 0
cap = cv2.VideoCapture(0)  # OpenCV video capture
gesture_text = "None"
midinote = 69  # Default MIDI note
mcp_y = None  # Middle finger MCP Y position for tracking movement

recording = False
recording_start_time = None
recording_file_path = None
# Get the default input device's information
input_device_info = sd.query_devices(kind="input")
input_channels = input_device_info["max_input_channels"]
recording_data = None

init = False
key_pressed_s = False

# Canvas size for displaying the webcam feed
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 480


def sound_synth(midi_freq, mode, instrument_id):
    """Send MIDI frequency or note via OSC."""
    print(
        f"Sending Midinote Number: {midi_freq}, Mode: {mode}, Instrument: {instrument_id}"
    )
    client.send_message("/from_python", [midi_freq, mode, instrument_id])


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

# def start_recording():
#     """Start recording audio."""
#     global recording, recording_start_time, recording_file_path, recording_data
#     if not recording:
#         recording = True
#         recording_start_time = time.time()
#         recording_file_path = os.path.join(
#             script_dir, "recording", f"recording_{int(recording_start_time)}.wav"
#         )
#         print(f"Recording started: {recording_file_path}")
#         recording_data = sd.rec(
#             int(10 * 44100),  # Buffer size for 10 seconds
#             samplerate=44100,
#             channels=1,
#             dtype="float32",
#         )


# def stop_recording():
#     """Stop recording audio and save the file."""
#     global recording, recording_file_path, recording_start_time, recording_data
#     if recording:
#         recording = False
#         sd.stop()
#         recording_duration = time.time() - recording_start_time
#         print(f"Recording stopped: {recording_file_path}")

#         # Save the recorded data
#         wv.write(
#             recording_file_path,
#             recording_data[: int(recording_duration * 44100)],
#             44100,
#             sampwidth=2,
#         )
#         # Play the recorded file via OSC
#         client.send_message("/playfile", recording_file_path)

def calculate_distance(thumb_tip, index_finger_tip, middle_finger_tip):
    if middle_finger_tip == 0:
        return np.sqrt(
            (thumb_tip[0] - index_finger_tip[0]) ** 2
            + (thumb_tip[1] - index_finger_tip[1]) ** 2
        )
    else:
        return np.sqrt(
            (thumb_tip[0] - middle_finger_tip[0]) ** 2
            + (thumb_tip[1] - middle_finger_tip[1]) ** 2
        )

def update_frame():
    """Update the video frame and process gestures."""
    global gesture_text, recording, current_mode, midinote, mcp_y,recording, recording_start_time, recording_file_path
    global input_device_info, input_channels
    global recording_data, init, key_pressed_s
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video.")
        root.after(10, update_frame)
        return

    # Resize the frame to fit the canvas
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks_list = []
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array(
                [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            ).flatten()
            hand_landmarks_list.append(landmarks)

        if len(hand_landmarks_list) == 2:
            combined_landmarks = np.concatenate(hand_landmarks_list).reshape(1, -1)
        else:
            combined_landmarks = np.concatenate(
                [hand_landmarks_list[0], np.zeros(42)]
            ).reshape(1, -1)

        gesture_id = model.predict(combined_landmarks)[0]
        gesture_text = label_encoder.inverse_transform([gesture_id])[0]
        middle_mcp = results.multi_hand_landmarks[0].landmark[
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
        ]

            # Update the mode behavior
        if current_mode == 0:  # Frequency Mode
            if results.multi_hand_landmarks:
                middle_mcp = results.multi_hand_landmarks[0].landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                ]
                freq = np.clip((1 - middle_mcp.y) * 1980 + 20, 20, 2000)
                sound_synth(0 if gesture_text != "Open Hand" else freq, current_mode, current_instrument)
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
            
            #Replace the 's' button
            if thumb_middle_distance < 40:  # If the distance between thumb and middle finger is less than 40 pixels
                if not key_pressed_s: 
                    mcp_y = middle_mcp.y
                    midinote = 69
                    print("Position Initialized")
                    init = True
                    key_pressed_s = True
                else: key_pressed_s = False

            if init:
                change_of_y = -(middle_mcp.y - mcp_y)
                change_of_midi = int(change_of_y / midi_interval)
                if abs(change_of_midi) > 0:
                    midinote += change_of_midi
                    mcp_y = middle_mcp.y
                match gesture_text:
                    case "Fist":
                        sound_synth(0, current_mode, current_instrument)
                    case 'Open Hand':
                        index_finger_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_TIP
                        ]

                        index_finger_tip_coords = (
                            int(index_finger_tip.x * frame.shape[1]),
                            int(index_finger_tip.y * frame.shape[0]),
                        )

                        thumb_index_distance = calculate_distance(
                        thumb_tip_coords, index_finger_tip_coords, 0
                                    )
                        if thumb_index_distance < 40:  # If thumb and index finger are close
                            sound_synth(midinote, current_mode, current_instrument)  # Play sound
                        else:
                            sound_synth(0, current_mode, current_instrument)  # Stop sound

        elif current_mode == 2:  # Composing Mode
            # if gesture_text == "One" and not recording:
            #     start_recording()
            # elif gesture_text != "One" and recording:
            #     stop_recording()

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
                        int(
                            10 * 44100
                        ),  # Initial buffer size, will be adjusted later
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

    # Convert frame to ImageTk format for Tkinter
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    video_canvas.imgtk = imgtk

    # Update gesture label
    gesture_label.config(text=f"Gesture: {gesture_text}")

    root.after(10, update_frame)


# Initialize Mediapipe Hands
hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
)

# Tkinter GUI
root = tk.Tk()
root.title("Hand Gesture Music Controller")

# Video canvas
video_canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
video_canvas.pack()

# Mode and instrument labels
mode_label = tk.Label(root, text=f"Mode: {mode_id[current_mode]}", font=("Arial", 14))
mode_label.pack()

instrument_label = tk.Label(
    root, text=f"Instrument: {instrument[current_instrument]}", font=("Arial", 14)
)
instrument_label.pack()

gesture_label = tk.Label(root, text="Gesture: None", font=("Arial", 14))
gesture_label.pack()

# Buttons
mode_button = tk.Button(root, text="Switch Mode", command=switch_mode)
mode_button.pack(side=tk.LEFT, padx=10)

instrument_button = tk.Button(root, text="Switch Instrument", command=switch_instrument)
instrument_button.pack(side=tk.LEFT, padx=10)

# Start video loop
update_frame()

# Start Tkinter main loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()