/*
 * ECHORA haptic wristband - BLE test firmware (ESP32 / ESP32-S3)
 * ----------------------------------------------------------------
 * This sketch turns the ESP32 into a BLE device that the laptop test
 * script (ble_test/test_ble_wristband.py) can connect to.
 *
 * It receives a 20-byte packet (one byte per electrode, 0-255) and,
 * for this TEST, just prints what it got over the USB serial monitor
 * and blinks the onboard LED so you can SEE that data arrived. Driving
 * the real 30 electrodes comes later (see the TODO near onWrite()).
 *
 * The BLE name + UUIDs below MUST match the Python script and the
 * echora project config. Don't change them unless you change both.
 *
 * HOW TO UPLOAD (Arduino IDE):
 *   1. Tools -> Board -> install "esp32" boards if you haven't
 *      (Boards Manager -> search "esp32" -> install).
 *   2. Tools -> Board -> pick your ESP32 (e.g. "ESP32S3 Dev Module"
 *      or "ESP32 Dev Module").
 *   3. Plug the ESP32 in by USB. Tools -> Port -> pick its COM port.
 *   4. Click Upload (the round arrow button).
 *   5. Open Tools -> Serial Monitor, set baud to 115200.
 *   6. On the laptop run:  python ble_test/test_ble_wristband.py
 *
 * Needs the standard ESP32 BLE libraries that ship with the esp32
 * board package — no extra library install required.
 */

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ---- must match the Python script + echora config -------------------------
#define DEVICE_NAME          "ECHORA-Wristband"
#define SERVICE_UUID         "0000ffe0-0000-1000-8000-00805f9b34fb"
#define CHARACTERISTIC_UUID  "0000ffe1-0000-1000-8000-00805f9b34fb"

#define N_ELECTRODES 20          // 4 rows x 5 cols

// Onboard LED pin. Many ESP32 boards use GPIO 2; some ESP32-S3 boards
// don't have one wired — if it never blinks, that's fine, watch serial.
#define LED_PIN 2

BLECharacteristic *pCharacteristic;
bool deviceConnected = false;

// Called when the central (laptop) connects or disconnects.
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer) {
    deviceConnected = true;
    Serial.println(">> Central connected");
  }
  void onDisconnect(BLEServer *pServer) {
    deviceConnected = false;
    Serial.println(">> Central disconnected, advertising again");
    pServer->getAdvertising()->start();   // allow reconnects
  }
};

// Called every time the laptop writes a packet to the characteristic.
class CharCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pChar) {
    String value = pChar->getValue();
    int len = value.length();

    Serial.printf("Got %d bytes: ", len);

    int activeCount = 0;
    for (int i = 0; i < len; i++) {
      uint8_t v = (uint8_t)value[i];
      Serial.printf("%3d ", v);
      if (v > 0) activeCount++;

      // TODO (later, for real hardware):
      //   set electrode i to intensity v here, e.g. with a PWM driver
      //   such as a PCA9685, or ledcWrite() on a real GPIO.
    }
    Serial.printf("  (%d active)\n", activeCount);

    // Visible proof data arrived: LED on if any electrode is active.
    digitalWrite(LED_PIN, activeCount > 0 ? HIGH : LOW);
  }
};

void setup() {
  Serial.begin(115200);
  delay(300);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.println("\nECHORA wristband BLE test firmware starting...");

  BLEDevice::init(DEVICE_NAME);
  // Allow a bigger packet so all 30 bytes fit in one write (default is 23).
  BLEDevice::setMTU(64);

  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
      CHARACTERISTIC_UUID,
      BLECharacteristic::PROPERTY_WRITE |
      BLECharacteristic::PROPERTY_WRITE_NR    // write without response (fast path)
  );
  pCharacteristic->setCallbacks(new CharCallbacks());
  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  BLEDevice::startAdvertising();

  Serial.printf("Advertising as '%s'. Waiting for the laptop...\n", DEVICE_NAME);
}

void loop() {
  // Nothing to do here — everything happens in the BLE callbacks.
  delay(1000);
}
