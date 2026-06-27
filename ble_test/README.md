# BLE wristband test (isolated — does not touch the main app)

Goal: prove the laptop can talk to the ESP32 haptic wristband over Bluetooth
(BLE) before changing anything in `src/`. If this passes, we wire the same
settings into `src/hardware/haptic_feedback.py`.

## What's here
- `echora_wristband_ble/echora_wristband_ble.ino` — firmware to **upload to the ESP32**.
- `test_ble_wristband.py` — script to **run on the laptop** that connects and sends test patterns.

The data format is simple: each packet is **30 bytes**, one per electrode
(5 rows × 6 cols), value `0` (off) to `255` (full). Both files already agree
on this, plus the BLE name and UUIDs.

## Steps

### 1. Upload firmware to the ESP32
Open `echora_wristband_ble/echora_wristband_ble.ino` in the Arduino IDE and
follow the upload instructions in the comment at the top of the file. Then
open the Serial Monitor at **115200 baud** — you should see
`Advertising as 'ECHORA-Wristband'. Waiting for the laptop...`.

### 2. Install the BLE library on the laptop
```
pip install bleak
```

### 3. Run the test
```
python ble_test/test_ble_wristband.py
```
You should see the script connect, then the ESP32 Serial Monitor should print
each packet it receives (and the onboard LED blinks when electrodes are
"active"). That means BLE works.

If it can't find the device, first list everything nearby:
```
python ble_test/test_ble_wristband.py --scan
```
and confirm `ECHORA-Wristband` is in the list.

## After it works
Once the test passes, we implement `_connect_ble()` / `_send_ble()` in
`src/hardware/haptic_feedback.py` using the exact same name/UUIDs and set
`HAPTIC_PROTOCOL = "BLE"`. The project's `config.py` already has these UUIDs.
