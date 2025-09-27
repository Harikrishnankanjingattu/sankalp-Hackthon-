#include <DHT.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#define DHTPIN D2
#define DHTTYPE DHT11
#define RAIN_SENSOR_PIN A0
#define LED_HIGH_TEMP D5
#define LED_HIGH_RISK D4
#define LED_DISASTER D3

DHT dht(DHTPIN, DHTTYPE);

// Set LCD address 0x27 (common) and 16x2 display
LiquidCrystal_I2C lcd(0x27, 16, 2);

void setup() {
  Serial.begin(9600);
  dht.begin();

  pinMode(LED_HIGH_TEMP, OUTPUT);
  pinMode(LED_HIGH_RISK, OUTPUT);
  pinMode(LED_DISASTER, OUTPUT);

  digitalWrite(LED_HIGH_TEMP, LOW);
  digitalWrite(LED_HIGH_RISK, LOW);
  digitalWrite(LED_DISASTER, LOW);

  // Initialize LCD
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Arduino Ready");

  Serial.println("Arduino Ready");
}

void loop() {
  float tempC = dht.readTemperature();
  float humidity = dht.readHumidity();
  int rainValue = analogRead(RAIN_SENSOR_PIN);
  int rainPercentage = map(rainValue, 1023, 0, 0, 100);  // dry->0%, wet->100%
  rainPercentage = constrain(rainPercentage, 0, 100);

  // Serial Output
  if (!isnan(tempC) && !isnan(humidity)) {
    Serial.print("SENSOR:");
    Serial.print(tempC);
    Serial.print(",");
    Serial.print(humidity);
    Serial.print(",");
    Serial.println(rainPercentage);
  } else {
    Serial.println("SENSOR:NaN,NaN,NaN");
  }

  // LCD Display
  lcd.clear();
  if (!isnan(tempC) && !isnan(humidity)) {
    lcd.setCursor(0,0);
    lcd.print("T:");
    lcd.print(tempC,1);
    lcd.print("C H:");
    lcd.print(humidity,0);
    lcd.print("%");

    lcd.setCursor(0,1);
    lcd.print("Rain:");
    lcd.print(rainPercentage);
    lcd.print("%");
  } else {
    lcd.setCursor(0,0);
    lcd.print("Sensor Error");
  }

  // Check for commands from Serial
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    Serial.print("Received: ");
    Serial.println(command);

    if (command.startsWith("ALERT")) {
      int tempAlert = command.substring(6,7).toInt();
      int riskAlert = command.substring(8,9).toInt();
      int disasterAlert = command.substring(10,11).toInt();

      digitalWrite(LED_HIGH_TEMP, tempAlert);
      digitalWrite(LED_HIGH_RISK, riskAlert);
      digitalWrite(LED_DISASTER, disasterAlert);

      Serial.print("LEDs: Temp=");
      Serial.print(tempAlert);
      Serial.print(", Risk=");
      Serial.print(riskAlert);
      Serial.print(", Disaster=");
      Serial.println(disasterAlert);
    }
  }

  delay(3000);  // Update every 3 seconds
}
