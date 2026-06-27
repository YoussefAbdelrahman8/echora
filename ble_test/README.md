# BLE wristband tester (isolated — does not touch the main app)

Goal: talk to the ESP32 haptic wristband over Bluetooth (BLE) from one
simple window, before changing anything in `src/`.

Grid is **4 rows × 5 cols = 20 electrodes**. Each BLE packet is **20 bytes**,
one per electrode, value `0` (off) to `255` (full).

## Files
- `echora_wristband_ble/echora_wristband_ble.ino` — firmware to **upload to the ESP32**.
- `wristband_gui.py` — the **one app you run on the laptop**. It scans, connects, and lets you click electrodes/patterns live.

## Steps

### 1. Upload firmware to the ESP32 (one time)
Open `echora_wristband_ble/echora_wristband_ble.ino` in the Arduino IDE and
follow the upload instructions in the comment at the top of the file. Then open
the Serial Monitor at **115200 baud** — you should see
`Advertising as 'ECHORA-Wristband'. Waiting for the laptop...`.

### 2. Install the BLE library (one time)
```
pip install bleak
```
(Tkinter, the GUI, already ships with Python on Windows.)

### 3. Run the app
```
python ble_test/wristband_gui.py
```
- Click **Scan & Connect** — it finds `ECHORA-Wristband` and connects (retries 3×).
- Click electrode buttons to toggle them, or use the pattern buttons.
- Each click sends instantly; watch the ESP32 Serial Monitor + onboard LED.

## If "Scan & Connect" fails
Most common fix on Windows:
1. Settings → Bluetooth & devices → if `ECHORA-Wristband` is listed, **Remove device**.
2. Toggle Bluetooth **off/on**.
3. Press **EN/reset** on the ESP32 so it advertises fresh.
4. Click Scan & Connect again. The real error prints in the terminal.

## After it works
Once this works, we implement `_connect_ble()` / `_send_ble()` in
`src/hardware/haptic_feedback.py` using the same name/UUIDs and set
`HAPTIC_PROTOCOL = "BLE"`. The project's `config.py` already has these UUIDs.
