#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_CircuitPlayground.h>

#define SEQ_LENGTH 50
float X_seq[SEQ_LENGTH], Y_seq[SEQ_LENGTH], Z_seq[SEQ_LENGTH];
float check_X_seq[SEQ_LENGTH], check_Y_seq[SEQ_LENGTH], check_Z_seq[SEQ_LENGTH];

int seq_index = 0;
bool sequence_recorded = false;
bool check_sequence_recorded = false;

void indicate_success();
void indicate_failure(int start_index, int end_index);

void setup() {
  Serial.begin(9600);
  CircuitPlayground.begin();
  CircuitPlayground.setBrightness(10);
}

void record_sequence() {
  Serial.println("Recording sequence...");

  // Wait for the record button to be released
  while (CircuitPlayground.leftButton()) {
    delay(10);
  }

  if (sequence_recorded) {
    // Stop recording if already recorded
    sequence_recorded = false;
    Serial.println("Recording stopped.");
  } else {
    // Initialize sequence index
    seq_index = 0;

    // Light up LEDs in yellow while recording
    for (int i = 0; i < 10; i++) {
      CircuitPlayground.setPixelColor(i, 255, 255, 0);
    }

    // Record accelerometer values until the record button is pressed again
    while (!CircuitPlayground.leftButton()) {
      X_seq[seq_index] = CircuitPlayground.motionX();
      Y_seq[seq_index] = CircuitPlayground.motionY();
      Z_seq[seq_index] = CircuitPlayground.motionZ();
     // Print accelerometer values
      Serial.print("X: ");
      Serial.print(X_seq[seq_index]);
      Serial.print("\tY: ");
      Serial.print(Y_seq[seq_index]);
      Serial.print("\tZ: ");
      Serial.println(Z_seq[seq_index]);

      // Increment seq_index by 1 and wrap it back to 0 if it reaches SEQ_LENGTH, 
      // making sure that the index stays within the bounds of the sequence array.
      seq_index = (seq_index + 1) % SEQ_LENGTH;


      delay(100);
    }

    sequence_recorded = true;
    Serial.println("Sequence Successfully Recorded!");
  }
}


    
bool check_sequence() {
  Serial.println("\nCheck Recording sequence...");

  // Wait for the record button to be released
  while (CircuitPlayground.rightButton()) {
    delay(10);
  }

  if (check_sequence_recorded) {
    // Stop recording if already recorded
    check_sequence_recorded = false;
    Serial.println("Check Recording stopped.");
  } 
  else 
  {
    // Initialize sequence index
    seq_index = 0;

    // Light up LEDs in Purple while recording
    for (int i = 0; i < 10; i++) {
      CircuitPlayground.setPixelColor(i, 255, 0, 255);
    }

    // Record accelerometer values until the record button is pressed again
    while (!CircuitPlayground.rightButton()) {
      check_X_seq[seq_index] = CircuitPlayground.motionX();
      check_Y_seq[seq_index] = CircuitPlayground.motionY();
      check_Z_seq[seq_index] = CircuitPlayground.motionZ();
      // Print accelerometer values
      Serial.print("check X: ");
      Serial.print(check_X_seq[seq_index]);
      Serial.print("\tcheck Y: ");
      Serial.print(check_Y_seq[seq_index]);
      Serial.print("\tcheck Z: ");
      Serial.println(check_Z_seq[seq_index]);

      // Increment seq_index by 1 and wrap it back to 0 if it reaches SEQ_LENGTH, 
      // making sure that the index stays within the bounds of the sequence array.
      seq_index = (seq_index + 1) % SEQ_LENGTH;

      delay(100);
    }

    check_sequence_recorded = true;
    Serial.println("\tCheck Sequence Successfully Recorded!");


    // Define start_index and end_index for sequence checking
    int start_index = 0;
    int end_index = seq_index - 1;

    float tolerance = 30.0; // change tolerance if necessary
     
     // Forumula for Euclidean Distance
    //https://www.cuemath.com/euclidean-distance-formula/

    for (int i = start_index; i <= end_index; i++) {
        float diffX = (check_X_seq[i] - X_seq[i]) * (check_X_seq[i] - X_seq[i]);
        float diffY = (check_Y_seq[i] - Y_seq[i])  *  (check_Y_seq[i] - Y_seq[i]) ;
        float diffZ = (check_Z_seq[i] - Z_seq[i])   * (check_Z_seq[i] - Z_seq[i]) ;

    float euclideanDistance = sqrt(diffX + diffY + diffZ);

    if (euclideanDistance > tolerance) {
        indicate_failure(start_index, end_index);
        return false;
    }
    }
   
    // Wait for the right button to be released before ending the function
    while (CircuitPlayground.rightButton()) {
      delay(10);
    }
 

    // If we reached here, the sequence was correct
    indicate_success();
    return true;
  }
  
}


void indicate_success() 
{
  // Light up LEDs in green and play a tone to indicate successful unlock
  for (int i = 0; i < 10; i++) 
    {
        CircuitPlayground.setPixelColor(i, 0, 255, 0);
    }
  CircuitPlayground.playTone(880, 400);
  delay(100);
  Serial.println("Success! You're in!");
}

void indicate_failure(int start_index, int end_index) 
{
  // Light up LEDs in red to indicate unsuccessful unlock
  for (int i = 0; i < 10; i++) 
    {
        CircuitPlayground.setPixelColor(i, 255, 0, 0);
    }
  delay(100);
  Serial.println("Incorrect Sequence...Try Again");
  Serial.println("\nExpected Sequence:");
  for (int i = start_index; i <= end_index; i++) 
    {
        Serial.print("X: ");
        Serial.print(X_seq[i]);
        Serial.print("\tY: ");
        Serial.print(Y_seq[i]);
        Serial.print("\tZ: ");
        Serial.println(Z_seq[i]);
    }
  Serial.println("\nSequence that was attempted:");

  for (int i = start_index; i <= end_index; i++)
   {
    Serial.print("Check Value X: ");
    Serial.print(check_X_seq[i]);
    Serial.print("\tCheck Value Y: ");
    Serial.print(check_Y_seq[i]);
    Serial.print("\tCheck Value Z: ");
    Serial.println(check_Z_seq[i]);
    }
}


void loop() {
  if (CircuitPlayground.leftButton()) {
    // Record sequence when the record (left) button is pressed
    record_sequence();
  } 
  else if (CircuitPlayground.rightButton()) {
    // Check sequence when the right button is pressed and then released
    while (CircuitPlayground.rightButton()) {
      // do nothing until the button is released
      delay(10);
    }
    check_sequence();
  }

  // Delay for stability
  delay(10);
}

