To start the program, run RUNME.py.

Wait for a while until the pygame window finish initialization.

The sound synthesizer consists of three mode: Continuous pitch Mode, Discrete Pitch Mode and Track Overlay Mode.
In addition, we provide two instrument: Sine Oscillator and Hand Flute.

The important parameters are shown on the right hand side of the screen.
You may change the mode and instrument by clicking the corresponding buttons on the right.

Here're the procedure to perform sound synthesis for different modes:

    First of all, open sound_synth.scd. Boot the Supercollider server and run the Supercollider code. 
    Otherwise, no sound will be given out 

    For Continuous pitch mode, you may change the frequency by moving your hand up and down, pitch will be changed according to the 
    change of y-coordinate of your hand.

    For Discrete pitch mode, you need to touch your index finger tip and thumb tip like an 'OK' hand gesture 👌. Then, perform the same
    hand gesture to perform sound synthesis once for each time. If you want to let the current position of your hand to be the new reference
    y-coordinate, you can perform the hand gesture which similar to the 'OK' hand gesture but touching the thumb tip with middle finger instaed.

    You can mute the sound by performing a 'fist' hand gesture 👊, or just move your hand outside the camera (i.e. hand gesture is 'No Hand').

    For track overlay mode, perform hand gesture 'one' ☝️ to start the recording. Then, make some voice and it will be recorded automatically. Perform 
    'Open Hand' to stop the recording. Then, the recorded audio will be add as a new soundtrack. Repeat the above procedure to add the another
    soundtrack. The recorded soundtracks will be store as a stack which obey the 'Last in-First out' (LIFO) rule. You can remove the last soundtrack
    added ony by performing a 'fist' hand gesture 👊 in this mode.

To exit the program, just click the 'Quit' button

