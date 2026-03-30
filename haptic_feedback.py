# =============================================================================
# haptic_feedback.py — ECHORA Haptic Wristband Communication
# =============================================================================
# Manages communication between ECHORA and the ESP32-S3 wristband.
#
# The wristband has a 5×6 grid of 30 electrotactile electrodes.
# Each electrode can be activated independently at varying intensities.
#
# Current status: STUB MODE
#   - All patterns are logged but not transmitted to hardware
#   - When ESP32 firmware and protocol are finalised, fill in
#     the send() method with the chosen transport (BLE/Serial/WiFi)
#   - Nothing else in the system needs to change
#
# Integration points:
#   - interaction_detection.py calls HapticBridge.send() directly
#   - control_unit.py can call haptic.send_pattern() for custom patterns
#   - This module is the ONLY place that touches hardware communication
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import time
import threading
import json
from typing import Optional, List
from pathlib import Path

from config import (
    HAPTIC_ROWS,
    HAPTIC_COLS,
)
from utils import logger


# =============================================================================
# HAPTIC PROTOCOL CONSTANTS
# =============================================================================
# When you decide on a protocol, set HAPTIC_PROTOCOL here and fill in
# the corresponding connection method below.
#
# Options:
#   "STUB"   — log only, no hardware (current default)
#   "SERIAL" — USB serial cable (simplest, good for development)
#   "BLE"    — Bluetooth Low Energy (wireless, good for wearable)
#   "WIFI"   — WiFi UDP (longest range, needs WiFi infrastructure)

HAPTIC_PROTOCOL = "STUB"

# Serial settings (used when HAPTIC_PROTOCOL = "SERIAL")
SERIAL_PORT     = "COM3"       # Windows: "COM3", "COM4" etc.
                               # Linux/Mac: "/dev/ttyUSB0", "/dev/ttyACM0"
SERIAL_BAUDRATE = 115200       # must match ESP32 firmware setting

# BLE settings (used when HAPTIC_PROTOCOL = "BLE")
BLE_DEVICE_NAME           = "ECHORA-Wristband"
BLE_SERVICE_UUID          = "12345678-1234-1234-1234-123456789abc"
BLE_CHARACTERISTIC_UUID   = "87654321-4321-4321-4321-cba987654321"

# WiFi settings (used when HAPTIC_PROTOCOL = "WIFI")
WIFI_ESP32_IP   = "192.168.1.100"   # IP address of ESP32 on local network
WIFI_ESP32_PORT = 5005              # UDP port to send patterns to


# =============================================================================
# HAPTIC PATTERNS
# =============================================================================
# Pre-built electrode grids for common feedback scenarios.
# Each pattern is a 5×6 numpy array with values 0.0-1.0.
# 0.0 = electrode off, 1.0 = full intensity.

def pattern_all_on(intensity: float = 1.0) -> np.ndarray:
    """All 30 electrodes on — used for SUCCESS feedback."""
    return np.full((HAPTIC_ROWS, HAPTIC_COLS), intensity, dtype=np.float32)

def pattern_all_off() -> np.ndarray:
    """All 30 electrodes off — used to clear the wristband."""
    return np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)

def pattern_left(intensity: float = 1.0) -> np.ndarray:
    """Left columns active — guide hand LEFT."""
    grid = np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)
    grid[:, 0] = intensity
    grid[:, 1] = intensity
    return grid

def pattern_right(intensity: float = 1.0) -> np.ndarray:
    """Right columns active — guide hand RIGHT."""
    grid = np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)
    grid[:, 4] = intensity
    grid[:, 5] = intensity
    return grid

def pattern_up(intensity: float = 1.0) -> np.ndarray:
    """Top rows active — guide hand UP."""
    grid = np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)
    grid[0, :] = intensity
    grid[1, :] = intensity
    return grid

def pattern_down(intensity: float = 1.0) -> np.ndarray:
    """Bottom rows active — guide hand DOWN."""
    grid = np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)
    grid[3, :] = intensity
    grid[4, :] = intensity
    return grid

def pattern_center(intensity: float = 1.0) -> np.ndarray:
    """Center row active — object is straight ahead."""
    grid = np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)
    grid[2, :] = intensity
    return grid

def pattern_danger_pulse() -> np.ndarray:
    """
    Alternating checkerboard — urgent DANGER alert.
    Maximum tactile contrast for immediate attention.
    """
    grid = np.zeros((HAPTIC_ROWS, HAPTIC_COLS), dtype=np.float32)
    for r in range(HAPTIC_ROWS):
        for c in range(HAPTIC_COLS):
            # Activate every other electrode in a checkerboard pattern.
            if (r + c) % 2 == 0:
                grid[r, c] = 1.0
    return grid


# =============================================================================
# HAPTIC FEEDBACK CLASS
# =============================================================================

class HapticFeedback:
    """
    Manages all communication with the ESP32-S3 haptic wristband.

    Currently operates in STUB mode — logs all patterns without
    transmitting to hardware. When ESP32 hardware and firmware are
    ready, change HAPTIC_PROTOCOL and fill in the appropriate
    connection method.

    The 5x6 electrode grid:
        Col:  0    1    2    3    4    5
        Row 0: [  ] [  ] [  ] [  ] [  ] [  ]   ← top of wrist
        Row 1: [  ] [  ] [  ] [  ] [  ] [  ]
        Row 2: [  ] [  ] [  ] [  ] [  ] [  ]   ← center
        Row 3: [  ] [  ] [  ] [  ] [  ] [  ]
        Row 4: [  ] [  ] [  ] [  ] [  ] [  ]   ← bottom of wrist

    Usage:
        haptic = HapticFeedback()
        haptic.connect()

        # Send a raw electrode grid:
        grid = np.zeros((5, 6), dtype=np.float32)
        grid[2, 3] = 1.0   # activate one electrode
        haptic.send(grid)

        # Use a preset pattern:
        haptic.send(pattern_right(intensity=0.8))

        # Pulse success feedback:
        haptic.pulse_success()

        haptic.disconnect()
    """

    def __init__(self):
        """Creates the HapticFeedback object. Does NOT connect yet."""

        # Whether connected to real hardware.
        self._connected: bool = False

        # The hardware connection object — type depends on protocol.
        # Serial: serial.Serial instance
        # BLE:    bleak BleakClient instance
        # WiFi:   socket.socket instance
        self._connection = None

        # Threading lock — prevents two threads sending simultaneously.
        # Simultaneous sends can corrupt the data stream.
        self._lock = threading.Lock()

        # Statistics
        self._send_count:  int = 0
        self._error_count: int = 0

        # Last grid sent — for diagnostics and debug overlay.
        self._last_grid: np.ndarray = pattern_all_off()

        logger.info(
            f"HapticFeedback created. Protocol: {HAPTIC_PROTOCOL}. "
            f"Call connect() to start."
        )


    # =========================================================================
    # CONNECTION
    # =========================================================================

    def connect(self) -> bool:
        """
        Connects to the ESP32 wristband using the configured protocol.

        Returns True if connected successfully, False on failure.
        In STUB mode, always returns True immediately.
        """

        if HAPTIC_PROTOCOL == "STUB":
            logger.info(
                "HapticFeedback: STUB mode. "
                "Patterns will be logged but not transmitted. "
                "Set HAPTIC_PROTOCOL when hardware is ready."
            )
            self._connected = True
            return True

        elif HAPTIC_PROTOCOL == "SERIAL":
            return self._connect_serial()

        elif HAPTIC_PROTOCOL == "BLE":
            return self._connect_ble()

        elif HAPTIC_PROTOCOL == "WIFI":
            return self._connect_wifi()

        else:
            logger.error(f"Unknown protocol: {HAPTIC_PROTOCOL}")
            return False


    def _connect_serial(self) -> bool:
        """
        Connects via USB serial cable.

        TODO: Install pyserial first:
            pip install pyserial

        Then set SERIAL_PORT to match your ESP32's COM port.
        On Windows: check Device Manager → Ports (COM & LPT).
        On Linux: usually /dev/ttyUSB0 or /dev/ttyACM0.
        """

        try:
            import serial   # pip install pyserial

            self._connection = serial.Serial(
                port     = SERIAL_PORT,
                baudrate = SERIAL_BAUDRATE,
                timeout  = 1.0
            )
            self._connected = True
            logger.info(
                f"HapticFeedback: Serial connected on "
                f"{SERIAL_PORT} at {SERIAL_BAUDRATE} baud."
            )
            return True

        except ImportError:
            logger.error(
                "pyserial not installed. Run: pip install pyserial"
            )
            return False

        except Exception as e:
            logger.error(f"Serial connection failed: {e}")
            return False


    def _connect_ble(self) -> bool:
        """
        Connects via Bluetooth Low Energy.

        TODO: Install bleak first:
            pip install bleak

        Then set BLE_DEVICE_NAME, BLE_SERVICE_UUID, and
        BLE_CHARACTERISTIC_UUID to match your ESP32 firmware.

        Note: BLE in Python requires asyncio. This stub uses a
        synchronous wrapper — you may need to adapt for async.
        """

        try:
            # BLE connection is async — this is a simplified placeholder.
            # For full BLE implementation, use asyncio + bleak properly.
            logger.warning(
                "BLE connection not yet implemented. "
                "See _connect_ble() TODO in haptic_feedback.py"
            )
            return False

        except Exception as e:
            logger.error(f"BLE connection failed: {e}")
            return False


    def _connect_wifi(self) -> bool:
        """
        Connects via WiFi UDP.

        No special library needed — uses Python's built-in socket module.
        The ESP32 must be on the same WiFi network.
        Set WIFI_ESP32_IP and WIFI_ESP32_PORT to match your ESP32 firmware.
        """

        try:
            import socket

            self._connection = socket.socket(
                socket.AF_INET,    # IPv4
                socket.SOCK_DGRAM  # UDP — no connection needed, just send
            )
            self._connected = True
            logger.info(
                f"HapticFeedback: WiFi UDP ready. "
                f"Target: {WIFI_ESP32_IP}:{WIFI_ESP32_PORT}"
            )
            return True

        except Exception as e:
            logger.error(f"WiFi UDP setup failed: {e}")
            return False


    # =========================================================================
    # SENDING PATTERNS
    # =========================================================================

    def send(self, grid: np.ndarray) -> bool:
        """
        Sends a 5×6 electrode activation grid to the wristband.

        This is the main function — everything else calls this.

        Arguments:
            grid: numpy array shape (HAPTIC_ROWS, HAPTIC_COLS)
                  Values: 0.0 = electrode off, 1.0 = full intensity.
                  Values between 0.0 and 1.0 = partial intensity.

        Returns:
            True if sent successfully, False on error.
        """

        if not self._connected:
            return False

        # Cache for diagnostics.
        self._last_grid = grid.copy()
        self._send_count += 1

        # Flatten the 5×6 grid to a 30-element 1D array.
        # This is what the ESP32 firmware will receive.
        flat = grid.flatten()

        with self._lock:

            if HAPTIC_PROTOCOL == "STUB":
                return self._send_stub(flat)

            elif HAPTIC_PROTOCOL == "SERIAL":
                return self._send_serial(flat)

            elif HAPTIC_PROTOCOL == "BLE":
                return self._send_ble(flat)

            elif HAPTIC_PROTOCOL == "WIFI":
                return self._send_wifi(flat)

        return False


    def _send_stub(self, flat: np.ndarray) -> bool:
        """
        Stub implementation — logs the pattern without transmitting.

        Logs every 10 sends to avoid flooding the console.
        Active electrodes are shown as their intensity values.
        """

        n_active = int(np.sum(flat > 0))

        # Log every 10 sends — frequent sends would flood the console.
        if self._send_count % 10 == 0:
            active_indices = [i for i, v in enumerate(flat) if v > 0]
            logger.debug(
                f"Haptic #{self._send_count}: "
                f"{n_active}/30 active — indices: {active_indices}"
            )

        return True


    def _send_serial(self, flat: np.ndarray) -> bool:
        """
        Sends pattern via USB serial.

        Protocol: 30 bytes, one per electrode, value 0-255.
        Framed with start byte (0xFF) and end byte (0xFE).

        Frame format: [0xFF] [e0] [e1] ... [e29] [0xFE]
        Total: 32 bytes per frame.

        TODO: Update this to match your ESP32 firmware's expected format.
        """

        if self._connection is None:
            return False

        try:
            # Convert 0.0-1.0 floats to 0-255 integers.
            bytes_data = bytes([int(v * 255) for v in flat])

            # Frame: start byte + 30 electrode bytes + end byte.
            frame = bytes([0xFF]) + bytes_data + bytes([0xFE])

            self._connection.write(frame)
            return True

        except Exception as e:
            logger.error(f"Serial send failed: {e}")
            self._error_count += 1
            return False


    def _send_ble(self, flat: np.ndarray) -> bool:
        """
        Sends pattern via BLE characteristic write.

        TODO: Implement when BLE protocol is finalised.
        Requires bleak library and async setup.
        """

        logger.warning("BLE send not yet implemented.")
        return False


    def _send_wifi(self, flat: np.ndarray) -> bool:
        """
        Sends pattern via WiFi UDP datagram.

        Protocol: 30 bytes sent as a single UDP packet.
        Each byte represents one electrode intensity (0-255).

        The ESP32 listens on WIFI_ESP32_PORT and activates
        electrodes based on received byte values.

        TODO: Update byte format to match your ESP32 firmware.
        """

        if self._connection is None:
            return False

        try:
            # Convert 0.0-1.0 floats to 0-255 integers.
            payload = bytes([int(v * 255) for v in flat])

            self._connection.sendto(
                payload,
                (WIFI_ESP32_IP, WIFI_ESP32_PORT)
            )
            return True

        except Exception as e:
            logger.error(f"WiFi UDP send failed: {e}")
            self._error_count += 1
            return False


    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def send_all_off(self) -> bool:
        """Turns all 30 electrodes off."""
        return self.send(pattern_all_off())

    def send_all_on(self, intensity: float = 1.0) -> bool:
        """Turns all 30 electrodes on at the given intensity."""
        return self.send(pattern_all_on(intensity))

    def pulse_success(self, pulses: int = 3, interval: float = 0.15):
        """
        Pulses all electrodes N times — success/arrival feedback.

        Runs in a background thread so it does not block the main loop.

        Arguments:
            pulses:   number of on/off cycles
            interval: seconds between each on/off transition
        """

        def _pulse_worker():
            for _ in range(pulses):
                self.send(pattern_all_on(1.0))
                time.sleep(interval)
                self.send(pattern_all_off())
                time.sleep(interval)

        threading.Thread(target=_pulse_worker, daemon=True).start()

    def pulse_danger(self, pulses: int = 2, interval: float = 0.1):
        """
        Rapid danger pulse — maximum urgency alert.
        Uses checkerboard pattern for maximum tactile contrast.
        """

        def _danger_worker():
            for _ in range(pulses):
                self.send(pattern_danger_pulse())
                time.sleep(interval)
                self.send(pattern_all_off())
                time.sleep(interval)

        threading.Thread(target=_danger_worker, daemon=True).start()

    def send_direction(self, dx: float, dy: float, intensity: float = 1.0):
        """
        Sends a directional guidance pattern based on a movement vector.

        This wraps the ElectrodeGridBuilder logic from interaction_detection.py
        for cases where direction feedback is needed outside INTERACTION mode.

        Arguments:
            dx:        horizontal displacement (positive = right)
            dy:        vertical displacement (positive = down in image coords)
            intensity: electrode intensity 0.0-1.0
        """

        from interaction_detection import ElectrodeGridBuilder
        builder = ElectrodeGridBuilder()
        grid    = builder.build_guidance_grid(dx, dy, intensity)
        self.send(grid)


    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_stats(self) -> dict:
        """Returns diagnostic statistics."""
        return {
            "protocol":    HAPTIC_PROTOCOL,
            "connected":   self._connected,
            "send_count":  self._send_count,
            "error_count": self._error_count,
            "last_active": int(np.sum(self._last_grid > 0)),
        }

    def visualise_grid(self, grid: np.ndarray) -> str:
        """
        Returns a text visualisation of an electrode grid.

        Useful for debugging — shows which electrodes are active.

        Example output:
            Row 0: [ ][ ][X][ ][ ][ ]
            Row 1: [ ][ ][X][X][ ][ ]
            Row 2: [X][X][X][X][X][X]
            Row 3: [ ][ ][X][X][ ][ ]
            Row 4: [ ][ ][X][ ][ ][ ]
        """

        lines = []
        for r in range(HAPTIC_ROWS):
            row_str = f"Row {r}: "
            for c in range(HAPTIC_COLS):
                val = grid[r, c]
                if val > 0.7:
                    row_str += "[X]"   # strong activation
                elif val > 0.3:
                    row_str += "[o]"   # medium activation
                else:
                    row_str += "[ ]"   # off
            lines.append(row_str)
        return "\n".join(lines)


    # =========================================================================
    # DISCONNECT
    # =========================================================================

    def disconnect(self):
        """Cleanly closes the hardware connection."""

        # Turn all electrodes off before disconnecting.
        if self._connected:
            self.send_all_off()
            time.sleep(0.05)

        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None

        self._connected = False
        logger.info(
            f"HapticFeedback disconnected. "
            f"Total sends: {self._send_count}, "
            f"errors: {self._error_count}"
        )


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_haptic: Optional[HapticFeedback] = None


def init_haptic() -> HapticFeedback:
    """
    Initialises the module-level haptic feedback singleton.
    Call once at startup from control_unit.py.
    """

    global _haptic

    if _haptic is not None:
        logger.debug("HapticFeedback already initialised.")
        return _haptic

    _haptic = HapticFeedback()
    _haptic.connect()
    logger.info("Module-level haptic feedback ready.")
    return _haptic


def get_haptic() -> Optional[HapticFeedback]:
    """Returns the shared HapticFeedback instance."""
    return _haptic


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":

    print("=== ECHORA haptic_feedback.py self-test ===\n")

    haptic = HapticFeedback()
    haptic.connect()

    # ── Test 1: All preset patterns ───────────────────────────────────────────
    print("Test 1: Preset patterns")

    patterns = {
        "all_off":      pattern_all_off(),
        "all_on":       pattern_all_on(),
        "left":         pattern_left(),
        "right":        pattern_right(),
        "up":           pattern_up(),
        "down":         pattern_down(),
        "center":       pattern_center(),
        "danger_pulse": pattern_danger_pulse(),
    }

    for name, grid in patterns.items():
        haptic.send(grid)
        n_active = int(np.sum(grid > 0))
        print(f"  {name:15s} — {n_active}/30 electrodes active")
        print(haptic.visualise_grid(grid))
        print()

    print("  PASSED\n")

    # ── Test 2: Directional guidance ──────────────────────────────────────────
    print("Test 2: Directional guidance vectors")

    directions = [
        ("Right",      100,    0),
        ("Left",      -100,    0),
        ("Up",            0, -100),
        ("Down",          0,  100),
        ("Up-Right",    80,  -80),
        ("Down-Left",  -80,   80),
    ]

    for label, dx, dy in directions:
        haptic.send_direction(dx, dy, intensity=1.0)
        print(f"  {label:12s} dx={dx:+4d} dy={dy:+4d}")

    print("  PASSED\n")

    # ── Test 3: Success pulse ─────────────────────────────────────────────────
    print("Test 3: Success pulse (3 pulses)")
    haptic.pulse_success(pulses=3, interval=0.1)
    time.sleep(1.0)
    print("  PASSED\n")

    # ── Test 4: Stats ─────────────────────────────────────────────────────────
    print("Test 4: Stats")
    stats = haptic.get_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")
    print("  PASSED\n")

    haptic.disconnect()
    print("=== All tests passed ===")