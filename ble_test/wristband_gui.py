
"""
Simple GUI to interact with the ECHORA haptic wristband over BLE.

Self-contained — imports nothing from the echora project, so it cannot
affect the main app. A small window shows the 5x6 electrode grid as
clickable buttons plus preset patterns. Clicking sends live to the ESP32.

Requirements:
    pip install bleak           (Tkinter ships with Python on Windows)

Run:
    python ble_test/wristband_gui.py

Flow:
    1. Upload the .ino firmware to the ESP32 first.
    2. Run this script, click "Connect".
    3. Click electrode buttons or preset patterns — they send instantly.
"""

import asyncio
import threading
import tkinter as tk
from tkinter import ttk

from bleak import BleakScanner, BleakClient

# ---- must match the firmware (.ino) and the echora project config --------
DEVICE_NAME         = "ECHORA-Wristband"
CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

ROWS, COLS = 4, 5
N = ROWS * COLS   # 20

# ---------------------------------------------------------------------------
# BLE runs on its own asyncio loop in a background thread, because Tkinter is
# synchronous and bleak is async. The GUI schedules BLE work onto that loop.
# ---------------------------------------------------------------------------
_loop = asyncio.new_event_loop()
_client: BleakClient | None = None


def _start_loop():
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


threading.Thread(target=_start_loop, daemon=True).start()


def run_async(coro):
    """Schedule a coroutine on the BLE loop from the GUI thread."""
    return asyncio.run_coroutine_threadsafe(coro, _loop)


class WristbandGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("ECHORA Wristband — BLE tester")
        root.resizable(False, False)

        # grid[r][c] holds the current intensity 0-255 for that electrode
        self.grid = [[0] * COLS for _ in range(ROWS)]
        self.buttons: list[list[tk.Button]] = []

        # --- top bar: connect + status + intensity ------------------------
        top = ttk.Frame(root, padding=10)
        top.grid(row=0, column=0, sticky="ew")

        self.connect_btn = ttk.Button(top, text="Scan & Connect", command=self.on_connect)
        self.connect_btn.grid(row=0, column=0, padx=(0, 10))

        self.status = tk.StringVar(value="Disconnected")
        ttk.Label(top, textvariable=self.status).grid(row=0, column=1, padx=(0, 20))

        ttk.Label(top, text="Intensity").grid(row=0, column=2)
        self.intensity = tk.IntVar(value=255)
        ttk.Scale(top, from_=0, to=255, variable=self.intensity,
                  orient="horizontal", length=120).grid(row=0, column=3, padx=5)

        # --- electrode grid ----------------------------------------------
        grid_frame = ttk.LabelFrame(root, text="Electrodes (click to toggle)", padding=10)
        grid_frame.grid(row=1, column=0, padx=10, pady=5)

        for r in range(ROWS):
            row_btns = []
            for c in range(COLS):
                b = tk.Button(grid_frame, width=4, height=2, bg="gray85",
                              command=lambda rr=r, cc=c: self.toggle(rr, cc))
                b.grid(row=r, column=c, padx=2, pady=2)
                row_btns.append(b)
            self.buttons.append(row_btns)

        # --- preset patterns ---------------------------------------------
        presets = ttk.LabelFrame(root, text="Patterns", padding=10)
        presets.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")

        actions = [
            ("All On", self.all_on), ("All Off", self.all_off),
            ("Left", self.left), ("Right", self.right),
            ("Up", self.up), ("Down", self.down),
            ("Center", self.center), ("Danger", self.danger),
        ]
        for i, (label, fn) in enumerate(actions):
            ttk.Button(presets, text=label, command=fn).grid(
                row=i // 4, column=i % 4, padx=3, pady=3, sticky="ew")

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---- BLE actions -----------------------------------------------------
    def set_status(self, text: str):
        # Always update Tk vars from the GUI thread.
        self.root.after(0, lambda: self.status.set(text))

    def on_connect(self):
        self.set_status("Connecting...")
        run_async(self._connect())

    async def _connect(self):
        global _client
        try:
            device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)
            if device is None:
                self.set_status(f"Not found: {DEVICE_NAME}")
                return
            # The first connect after a scan often fails on Windows; retry a few times.
            last_err = None
            for attempt in range(1, 4):
                try:
                    self.set_status(f"Connecting... (try {attempt}/3)")
                    _client = BleakClient(device)
                    await _client.connect()
                    self.set_status(f"Connected ({_client.mtu_size}B MTU)")
                    return
                except Exception as e:
                    last_err = e
                    print(f"connect attempt {attempt} failed: {e!r}")
                    await asyncio.sleep(1.5)
            self.set_status(f"Failed: {last_err}")
        except Exception as e:
            self.set_status(f"Error: {e}")

    async def _send(self):
        if _client is None or not _client.is_connected:
            self.set_status("Not connected")
            return
        flat = bytes(self.grid[r][c] for r in range(ROWS) for c in range(COLS))
        try:
            await _client.write_gatt_char(CHARACTERISTIC_UUID, flat, response=False)
        except Exception as e:
            self.set_status(f"Send failed: {e}")

    def send(self):
        run_async(self._send())

    # ---- grid manipulation ----------------------------------------------
    def refresh_buttons(self):
        for r in range(ROWS):
            for c in range(COLS):
                on = self.grid[r][c] > 0
                self.buttons[r][c].configure(bg="tomato" if on else "gray85")

    def toggle(self, r: int, c: int):
        self.grid[r][c] = 0 if self.grid[r][c] > 0 else self.intensity.get()
        self.refresh_buttons()
        self.send()

    def set_all(self, value_fn):
        for r in range(ROWS):
            for c in range(COLS):
                self.grid[r][c] = value_fn(r, c)
        self.refresh_buttons()
        self.send()

    # ---- presets ---------------------------------------------------------
    def _v(self):
        return self.intensity.get()

    def all_on(self):  self.set_all(lambda r, c: self._v())
    def all_off(self): self.set_all(lambda r, c: 0)
    def left(self):    self.set_all(lambda r, c: self._v() if c in (0, 1) else 0)
    def right(self):   self.set_all(lambda r, c: self._v() if c in (COLS - 2, COLS - 1) else 0)
    def up(self):      self.set_all(lambda r, c: self._v() if r in (0, 1) else 0)
    def down(self):    self.set_all(lambda r, c: self._v() if r in (ROWS - 2, ROWS - 1) else 0)
    def center(self):  self.set_all(lambda r, c: self._v() if r in (1, 2) else 0)
    def danger(self):  self.set_all(lambda r, c: self._v() if (r + c) % 2 == 0 else 0)

    # ---- shutdown --------------------------------------------------------
    def on_close(self):
        async def _disc():
            if _client is not None and _client.is_connected:
                try:
                    await _client.write_gatt_char(CHARACTERISTIC_UUID, bytes(N), response=False)
                    await _client.disconnect()
                except Exception:
                    pass
        fut = run_async(_disc())
        try:
            fut.result(timeout=3)
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    WristbandGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
