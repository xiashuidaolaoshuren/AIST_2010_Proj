import mido
import os

# Load the MIDI file
script_dir = os.path.dirname(os.path.abspath(__file__))
midi_file_path = os.path.join(script_dir, '【Animenz】鸟之诗 (Project CHE Special)-(Av8319626,P1).mid')
midi_file = mido.MidiFile(midi_file_path)
# Iterate through the MIDI messages and print note numbers
midi_notes = []
for track in midi_file.tracks:
    for msg in track:
        if msg.type == 'note_on':  # Note on event
            midi_notes.append(msg.note)

# Print the MIDI note numbers
print("MIDI Note Numbers:", midi_notes)
