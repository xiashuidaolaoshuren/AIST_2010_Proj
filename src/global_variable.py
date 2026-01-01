from pythonosc import udp_client
import pygame

midi_interval = 0.026 #Value of y to increase / decrease 1 midinote
respond_time = 1
mode_id = ['Continuous Pitch Mode', 'Discrete Pitch Mode', "Track Overlay Mode"]
instrument = ['Sine Oscillator', 'Hand Flute']
gestures = [
    "Open Hand",
    "Fist",
    "Cross",
    "One",
    "Two",
    "Three",
    "Thumb",
]
client = udp_client.SimpleUDPClient("127.0.0.1", 57120)
window_width = 1000
window_height = 500


