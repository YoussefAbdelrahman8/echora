
import numpy as np
import time
import threading
import json
from typing import Optional, List
from pathlib import Path

from src.core.config import settings
from src.core.utils import logger

HAPTIC_PROTOCOL = "STUB"

SERIAL_PORT     = "COM3"       # Windows: "COM3", "COM4" etc.
SERIAL_BAUDRATE = 115200       # must match ESP32 firmware setting

BLE_DEVICE_NAME           = "ECHORA-Wristband"
settings.BLE_SERVICE_UUID          = "12345678-1234-1234-1234-123456789abc"
settings.BLE_CHARACTERISTIC_UUID   = "87654321-4321-4321-4321-cba987654321"

WIFI_ESP32_IP   = "192.168.1.100"   # IP address of ESP32 on local network
WIFI_ESP32_PORT = 5005              # UDP port to send patterns to

def pattern_all_on(intensity: float = 1.0) -> np.ndarray:
    """All 30 electrodes on — used for SUCCESS feedback."""
    return np.full((settings.HAPTIC_ROWS, settings.HAPTIC_COLS), intensity, dtype=np.float32)

def pattern_all_off() -> np.ndarray:
    """All 30 electrodes off — used to clear the wristband."""
    return np.zeros((settings.HAPTIC_ROWS, settings.HAPTIC_COLS), dtype=np.float32)

def pattern_left(intensity: float = 1.0) -> np.ndarray:
    """Left columns active — guide hand LEFT."""
    grid = np.zeros((settings.HAPTIC_ROWS, settings.HAPTIC_COLS), dtype=np.float32)
    grid[:, 0] = intensity
    grid[:, 1] = intensity
    return grid

def pattern_right(intensity: float = 1.0) -> np.ndarray:
    """Right columns active — guide hand RIGHT."""
    grid = np.zeros((settings.HAPTIC_ROWS, settings.HAPTIC_COLS), dtype=np.float32)
    grid[:, 4] = intensity
    grid[:, 5] = intensity
    return grid

def pattern_up(intensity: float = 1.0) -> np.ndarray:
    """Top rows active — guide hand UP."""
    grid = np.zeros((settings.HAPTIC_ROWS, settings.HAPTIC_COLS), dtype=np.float32)
    grid[0, :] = intensity
    grid[1, :] = intensity
    return grid

def pattern_down(intensity: float = 1.0) -> np.ndarray:
    """Bottom rows active — guide hand DOWN."""
    grid = np.zeros((settings.HAPTIC_ROWS, settings.HAPTIC_COLS), dtype=np.float32)
    grid[3, :] = intensity
    grid[4, :] = intensity
    return grid

def pattern_center(intensity: float = 1.0) -> np.ndarray:
    """Center row active — object is straight ahead."""
    grid = np.zeros((settings.HAPTIC_ROWS, settings.HAPTIC_COLS), dtype=np.float32)
    grid[2, :] = intensity
    return grid

def pattern_danger_pulse() -> np.ndarray:
    """
    Alternating checkerboard — urgent DANGER alert.
    Maximum tactile contrast for immediate attention.
    """
    grid = np.zeros((settings.HAPTIC_ROWS, settings.HAPTIC_COLS), dtype=np.float32)
    for r in range(settings.HAPTIC_ROWS):
        for c in range(settings.HAPTIC_COLS):
            if (r + c) % 2 == 0:
                grid[r, c] = 1.0
    return grid

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

        grid = np.zeros((5, 6), dtype=np.float32)
        grid[2, 3] = 1.0   # activate one electrode
        haptic.send(grid)

        haptic.send(pattern_right(intensity=0.8))

        haptic.pulse_success()

        haptic.disconnect()
    """

    def __init__(self):
        """Creates the HapticFeedback object. Does NOT connect yet."""

        self._connected: bool = False

        self._connection = None

        self._lock = threading.Lock()

        self._send_count:  int = 0
        self._error_count: int = 0

        self._last_grid: np.ndarray = pattern_all_off()

        logger.info(
            f"HapticFeedback created. Protocol: {HAPTIC_PROTOCOL}. "
            f"Call connect() to start."
        )

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

        Then set BLE_DEVICE_NAME, settings.BLE_SERVICE_UUID, and
        settings.BLE_CHARACTERISTIC_UUID to match your ESP32 firmware.

        Note: BLE in Python requires asyncio. This stub uses a
        synchronous wrapper — you may need to adapt for async.
        """

        try:
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

    def send(self, grid: np.ndarray) -> bool:
        """
        Sends a 5×6 electrode activation grid to the wristband.

        This is the main function — everything else calls this.

        Arguments:
            grid: numpy array shape (settings.HAPTIC_ROWS, settings.HAPTIC_COLS)
                  Values: 0.0 = electrode off, 1.0 = full intensity.
                  Values between 0.0 and 1.0 = partial intensity.

        Returns:
            True if sent successfully, False on error.
        """

        if not self._connected:
            return False

        self._last_grid = grid.copy()
        self._send_count += 1

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
            bytes_data = bytes([int(v * 255) for v in flat])

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
        for r in range(settings.HAPTIC_ROWS):
            row_str = f"Row {r}: "
            for c in range(settings.HAPTIC_COLS):
                val = grid[r, c]
                if val > 0.7:
                    row_str += "[X]"   # strong activation
                elif val > 0.3:
                    row_str += "[o]"   # medium activation
                else:
                    row_str += "[ ]"   # off
            lines.append(row_str)
        return "\n".join(lines)

    def disconnect(self):
        """Cleanly closes the hardware connection."""

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

