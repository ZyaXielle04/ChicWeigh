#include <TM1637Display.h>
#include <math.h>  // For isnan() (optional, but good practice)

#define CLK 2
#define DIO 3
TM1637Display display(CLK, DIO);

String inputString = "";
boolean stringComplete = false;
float lastValidWeight = 0.0;  // Track the last valid weight

void setup() {
  Serial.begin(9600);
  display.setBrightness(0x0f); // Max brightness
  display.showNumberDecEx(0, 0b00000010, true); // Show "0.00" (DP on tenths place)
  Serial.println("ChicWeigh Ready! Waiting for weight from Python (or manual input).");
}

void loop() {
  if (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }

  if (stringComplete) {
    // Trim whitespace if any
    inputString.trim();
    
    if (inputString.length() > 0) {
      float weight = inputString.toFloat();
      if (!isnan(weight) && weight >= 0) {  // Valid non-negative number
        lastValidWeight = weight;
        Serial.print("Valid weight: ");
        Serial.print(weight, 2);  // Echo back
        Serial.println(" kg - Display updated.");
      } else {
        Serial.print("Invalid input ('");
        Serial.print(inputString);
        Serial.println("')! Keeping previous value.");
      }
    } else {
      Serial.println("Empty input! Keeping previous value.");
    }
    
    // Scale and clamp for display (0.00 to 99.99)
    int displayWeight = (int)(lastValidWeight * 100);
    displayWeight = constrain(displayWeight, 0, 9999);  // Prevent overflow
    display.showNumberDecEx(displayWeight, 0b00000010, true); // Show with DP on tenths
    
    inputString = "";
    stringComplete = false;
  }
}