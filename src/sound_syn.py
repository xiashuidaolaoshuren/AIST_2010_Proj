### Boot the SuperCollier server first before running sound_synth()!!!
from global_variable import client

def sound_synth(midi_freq, mode, instrument_id):
    print(
        f"Sending Midinote Number: {midi_freq}, Mode: {mode}, Instrument: {instrument_id}"
    )
    client.send_message("/from_python", [midi_freq, mode, instrument_id])
    if mode == 2 and midi_freq == 0:
        client.send_message("/stop_recording", [0])












