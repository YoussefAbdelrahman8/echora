"""
Standalone BLE test for the ECHORA haptic wristband (ESP32).

This script is SELF-CONTAINED — it does NOT import anything from the
echora project, so running it cannot affect the main app. Use it to
prove the laptop <-> ESP32 BLE link works before we wire BLE into
src/hardware/haptic_feedback.py.

What it does:
    1. Scans for the wristband by BLE name.
    2. Connects to it.
    3. Sends a few test patterns (all-on, all-off, left, right, a
       moving dot) as 30-byte packets — one byte per electrode, 0-255.

Requirements:
    pip install bleak

Run:
    python ble_test/test_ble_wristband.py

If you don't know the device name yet, run with --scan to just list
every BLE device nearby, then copy the right name into DEVICE_NAME.
"""

import asyncio
import argparse
import sys

from bleak import BleakScanner, BleakClient

# ---------------------------------------------------------------------------
# These three values MUST match the ESP32 firmware (the .ino sketch).
# They are the same values already configured in the echora project, so if
# the test passes, the project will use the exact same settings.
# ---------------------------------------------------------------------------
DEVICE_NAME         = "ECHORA-Wristband"
SERVICE_UUID        = "0000ffe0-0000-1000-8000-00805f9b34fb"
CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

# The wristband grid: 5 rows x 6 cols = 30 electrodes.
ROWS, COLS = 5, 6
N_ELECTRODES = ROWS * COLS


def grid_all(value: int) -> bytes:
    """30 bytes, every electrode set to `value` (0-255)."""
    return bytes([value] * N_ELECTRODES)


def grid_left(value: int = 255) -> bytes:
    """Left two columns on — 'guide hand left'."""
    out = [0] * N_ELECTRODES
    for r in range(ROWS):
        out[r * COLS + 0] = value
        out[r * COLS + 1] = value
    return bytes(out)


def grid_right(value: int = 255) -> bytes:
    """Right two columns on — 'guide hand right'."""
    out = [0] * N_ELECTRODES
    for r in range(ROWS):
        out[r * COLS + 4] = value
        out[r * COLS + 5] = value
    return bytes(out)


def grid_single(index: int, value: int = 255) -> bytes:
    """One electrode on (index 0..29) — used for the 'moving dot' test."""
    out = [0] * N_ELECTRODES
    out[index] = value
    return bytes(out)


async def just_scan():
    """List every BLE device nearby so you can find the wristband's name."""
    print("Scanning for 8 seconds...\n")
    devices = await BleakScanner.discover(timeout=8.0)
    if not devices:
        print("No BLE devices found. Is Bluetooth on? Is the ESP32 powered?")
        return
    for d in devices:
        print(f"  name={d.name!r:30}  address={d.address}")
    print("\nCopy the wristband's name into DEVICE_NAME at the top of this file.")


async def run_test():
    print(f"Looking for '{DEVICE_NAME}'...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)
    if device is None:
        print(f"\nCould not find '{DEVICE_NAME}'.")
        print("  - Make sure the ESP32 is powered and running the .ino sketch.")
        print("  - Run with --scan to see what names are advertising.")
        sys.exit(1)

    print(f"Found {device.address}. Connecting...")
    async with BleakClient(device) as client:
        print(f"Connected: {client.is_connected}")
        print(f"Negotiated MTU: {client.mtu_size} bytes "
              f"(need >= {N_ELECTRODES + 3} to send all 30 in one write)\n")

        async def send(name: str, packet: bytes):
            # response=True asks the ESP32 to acknowledge each write — slower
            # but gives a clear pass/fail during testing.
            await client.write_gatt_char(CHARACTERISTIC_UUID, packet, response=True)
            print(f"  sent {name:16} ({len(packet)} bytes)")
            await asyncio.sleep(0.6)

        print("Running test sequence — watch the wristband / ESP32 serial monitor:\n")
        await send("ALL ON",  grid_all(255))
        await send("ALL OFF", grid_all(0))
        await send("LEFT",    grid_left())
        await send("RIGHT",   grid_right())

        print("  moving dot across all 30 electrodes...")
        for i in range(N_ELECTRODES):
            await client.write_gatt_char(CHARACTERISTIC_UUID, grid_single(i), response=False)
            await asyncio.sleep(0.08)

        await send("ALL OFF", grid_all(0))
        print("\nTest complete. If the ESP32 logged the packets, BLE works.")


def main():
    parser = argparse.ArgumentParser(description="BLE test for ECHORA wristband")
    parser.add_argument("--scan", action="store_true",
                        help="Just list nearby BLE devices and exit")
    args = parser.parse_args()

    try:
        asyncio.run(just_scan() if args.scan else run_test())
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
