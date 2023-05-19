# Embedded-Challenge



Welcome to our Arduino Motion Sequence Lock project. 
This project uses the Adafruit Circuit Playground Classic to record a motion sequence using the on-board accelerometer, 
and then uses that sequence as a "lock". The correct sequence "unlocks" the device, indicated by LED lights and sound. 

## How to Use

1. Record the Sequence:  
   - Press the left button (or #4, the record button) to start recording the sequence. 
   - During this time, the lights on the Circuit Playground Classic will turn yellow, indicating that the device 
   is in the recording mode.
   - Perform your unique motion sequence while the device is in this mode. 
   - Press the record button again to save the sequence.

2. Check the Sequence: 
   - To attempt to unlock the device, press the right button (or #19, the check sequence button).
   - Make sure to fully press and hold down the button. The lights on the Circuit Playground Classic will turn purple, 
   indicating that the device is now in the sequence checking mode. Perform the same motion sequence that you recorded earlier.

3. Unlocking Feedback:
   - If the correct sequence is performed, the lights will turn green and the device will output a sound, 
     indicating that the device has been successfully unlocked.
   - If the incorrect sequence is performed, the lights will turn red, indicating an unsuccessful unlock attempt. 
     Multiple attempts can be made to unlock the device.

4. Reset the Sequence: If you wish to record a new sequence, simply press the left button (or #4, the record button) again. 
   This will overwrite the old sequence. A new sequence checking process can commence for the newly recorded sequence.

Enjoy!
