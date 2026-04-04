# ECHORA AI
AI-Powered Sensory Substitution System for everyday navigation and interaction.

## Architecture & Project Structure

ECHORA uses a deeply modular architecture designed to make adding or replacing features incredibly simple. The application is logically split into:

```text
echora/
├── src/
│   ├── main.py                  # CLI application entry point
│   ├── core/
│   │   ├── config.py            # Global settings & configuration
│   │   ├── utils.py             # Math, bounding boxes, logging, and helpers
│   │   ├── state_machine.py     # System state orchestration
│   │   └── control_unit.py      # Core brain/game-loop module
│   ├── perception/
│   │   ├── obstacle_detection.py# YOLO-based obstacle detection logic
│   │   ├── interaction_detection.py # Hands/Objects scanning for interaction
│   │   ├── ocr.py               # Optical Character Recognition logic
│   │   ├── banknote.py          # YOLO Banknote classification model
│   │   ├── echora_face.py       # Face embedding recognition engine
│   │   └── kalman_tracker.py    # Temporal smoothing for bounding boxes
│   ├── hardware/
│   │   ├── camera.py            # DepthAI / OAK-D interaction wrapping
│   │   ├── audio_feedback.py    # pyttsx3 Text-To-Speech queues and panning
│   │   └── haptic_feedback.py   # ESP32 BLE signal translation code
│   └── storage/
│       ├── database.py          # SQLite interactions
│       └── register_face.py     # Face registration script
├── tests/                       # Standalone regression and function tests
├── models/                      # ML models (.pt, .blob)
├── database/                    # Local DB storage (.db)
├── sounds/                      # Audial .wav alerts
├── requirements.txt
├── run_tests.sh
└── Dockerfile
```

## System Flow & Pipeline

ECHORA operates within an infinite loop defined in **`control_unit.py`**. The overall pipeline flows as follows on a frame-by-frame basis:

1. **Information Capture**: It begins in `src/hardware/camera.py`, where RGB frames and synchronized spatial Depth-Maps are acquired directly from the connected hardware camera.
2. **Analysis Routing**: Based on the active mode (defined by the `state_machine.py`), `control_unit.py` delegates this unified data bundle to various sub-engines within `src/perception/`.
    * **Obstacle Detection** triggers via YOLO on every frame, classifying dynamic distance zones via the `kalman_tracker`. 
    * **Faces, Banknotes, and Interactions** are evaluated at defined sub-frame tick rates (e.g., once every 5-15 frames) to offset operational load.
3. **Decisions State**: Based on the perception results, `state_machine.py` decides if the system needs to switch modes. For instance, if an OCR-readable billboard crosses a specific depth threshold, the state machine fires a callback forcing the system into `OCR` mode.
4. **Output Dispatch**: Finally, actions are fed into `src/hardware/`. 
    * `audio_feedback.py` uses threaded speech synthesis to announce objects and directions relative to camera panning values.
    * `haptic_feedback.py` emits Bluetooth Low Energy requests out to a wearable wristband based on proximity intensity values.

## Run via Docker
A Dockerfile is provided to establish a consistent testing and execution environment. You must have the Docker daemon running to initiate:

```bash
# Build the image dependencies
docker build -t echora .

# Spin up a container
docker run --rm echora
```

## Run Local Tests
We have separated individual module self-tests into `tests/`.

You can automatically run these sequentially across your operating system from the root workspace directory with:
```bash
./run_tests.sh
```
