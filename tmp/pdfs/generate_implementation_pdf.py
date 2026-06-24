from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "echora_implementation_workflow.pdf"

NAVY = colors.HexColor("#123047")
BLUE = colors.HexColor("#226D9B")
TEAL = colors.HexColor("#007C78")
PALE = colors.HexColor("#EAF3F8")
INK = colors.HexColor("#1E2932")
MUTED = colors.HexColor("#51606C")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=27, leading=32, textColor=NAVY, alignment=TA_CENTER, spaceAfter=9,
))
styles.add(ParagraphStyle(
    name="Subtitle", parent=styles["Normal"], fontName="Helvetica",
    fontSize=11, leading=15, textColor=MUTED, alignment=TA_CENTER, spaceAfter=22,
))
styles.add(ParagraphStyle(
    name="H1Custom", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=17, leading=22, textColor=NAVY, spaceBefore=10, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="H2Custom", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=12.5, leading=16, textColor=TEAL, spaceBefore=9, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9.0, leading=13, textColor=INK, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="Small", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=8.3, leading=11.5, textColor=INK,
))
styles.add(ParagraphStyle(
    name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold",
    fontSize=9.5, leading=14, textColor=NAVY, backColor=PALE,
    borderColor=colors.HexColor("#B8D6E6"), borderWidth=0.5, borderPadding=8,
    spaceBefore=6, spaceAfter=10,
))

def p(text, style="BodyCustom"):
    return Paragraph(text, styles[style])

def bullets(items):
    return [p(f"- {item}") for item in items]

def feature_table(rows):
    data = [[p("Feature", "Small"), p("Primary technologies", "Small"), p("Result", "Small")]]
    data += [[p(a, "Small"), p(b, "Small"), p(c, "Small")] for a, b, c in rows]
    table = Table(data, colWidths=[36*mm, 68*mm, 65*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D5DD")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F9FB")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#B8D6E6"))
    canvas.line(doc.leftMargin, 13*mm, A4[0] - doc.rightMargin, 13*mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 8.5*mm, "ECHORA - Implementation Workflow")
    canvas.drawRightString(A4[0] - doc.rightMargin, 8.5*mm, f"Page {doc.page}")
    canvas.restoreState()

story = []
story += [Spacer(1, 32*mm), p("ECHORA", "TitleCustom"),
          p("Implementation Workflow and Technology Summary", "Subtitle")]
story.append(p(
    "ECHORA is a real-time sensory-substitution system designed to help blind and visually impaired users navigate, read text, identify people and banknotes, and reach nearby objects.",
    "Callout"))
story.append(p("Document purpose", "H1Custom"))
story.append(p(
    "This document summarizes the implemented architecture, execution flow, feature-level technologies, and feedback mechanisms. It is written as a ready-to-use implementation section for project documentation."))
story.append(Spacer(1, 6*mm))
story.append(p("System at a glance", "H1Custom"))
story.append(feature_table([
    ("Navigation", "YOLOv8, stereo depth, ByteTrack", "Obstacle distance, direction, safety alerts"),
    ("Text reading", "PaddleOCR PP-OCRv4, OpenCV", "Spoken Arabic/English text"),
    ("Face identification", "InsightFace, ArcFace, SQLite", "Known or unknown person announcement"),
    ("Banknotes", "Custom YOLOv8 model", "Spoken Egyptian Pound denomination"),
    ("Object reaching", "MediaPipe Hands, depth, haptic grid", "Directional tactile guidance"),
]))
story.append(PageBreak())

story.append(p("1. Overall architecture and control flow", "H1Custom"))
story.append(p(
    "The application is a Python command-line program. <b>main.py</b> starts the system and creates a central ControlUnit. The ControlUnit initializes the camera, AI models, database, audio, haptic services, and state machine, then executes a continuous frame loop."))
story.append(p("Per-frame execution sequence", "H2Custom"))
story += bullets([
    "The OAK-D camera produces synchronized RGB, stereo-depth, and IMU data.",
    "The obstacle detector analyzes the RGB frame and combines detections with depth measurements.",
    "Background probes check for text, faces, banknotes, and interactable objects.",
    "The state machine selects the appropriate operating mode in automatic operation.",
    "The active mode generates speech, spatial sound, haptic feedback, or a database update.",
    "The loop repeats continuously while timing metrics and logs are collected.",
])
story.append(p("Performance approach", "H2Custom"))
story.append(p(
    "Expensive OCR and face-identification operations execute in background threads. OCR calls are protected by a lock because its inference engine is not thread-safe. Detection results are cached where possible, and models select CUDA, Apple MPS, or CPU based on available hardware."))
story.append(p("Automatic state transitions", "H2Custom"))
story.append(feature_table([
    ("Navigation", "Default mode", "Continuously detects and announces hazards"),
    ("OCR", "Text within 2 m plus low motion", "Reads stable detected text"),
    ("Interaction", "Interactable object within 0.8 m", "Guides the user's hand to the object"),
    ("Face ID", "Repeated high-confidence face detection", "Identifies a registered person"),
    ("Banknote", "Repeated banknote detection plus low motion", "Classifies the denomination"),
]))
story.append(p(
    "Transitions require consecutive confirming frames and minimum dwell times. This prevents noisy single-frame detections from causing unwanted mode switches. A danger obstacle has higher priority and can force a return to Navigation mode.", "Callout"))

story.append(p("2. Camera and sensor acquisition", "H1Custom"))
story.append(p("Technologies: DepthAI, Luxonis OAK-D, OpenCV, NumPy."))
story.append(p(
    "The camera module configures an OAK-D pipeline with a 1280 x 800 RGB camera, left and right mono cameras, stereo-depth processing, and an IMU. StereoDepth aligns depth with the RGB stream, allowing each detected object to be assigned a distance in millimetres. A dedicated IMU thread stores current acceleration and gyroscope data; acceleration magnitude is used to estimate motion and suppress still-image features while the user is moving."))
story.append(PageBreak())

story.append(p("3. Navigation and obstacle awareness", "H1Custom"))
story.append(p("Technologies: Ultralytics YOLOv8, PyTorch, OpenCV, stereo depth, ByteTrack, optional Ollama/Gemma VLM."))
story.append(p("Feature flow", "H2Custom"))
story += bullets([
    "The RGB frame is analyzed by a custom accessibility YOLOv8 model and, when enabled, a second COCO YOLOv8 model.",
    "Only relevant classes such as people, furniture, doors, stairs, vehicles, bottles, phones, and switches are kept.",
    "For each bounding box, the detector samples the depth map, calculates distance, and derives a horizontal viewing angle.",
    "ByteTrack-style tracking stabilizes identities across frames and supports approach-speed estimates.",
    "Objects are categorized as Danger (under 1.5 m), Warning (1.5 to 3 m), Safe, or Unknown if depth is unavailable.",
])
story.append(p("Feedback behavior", "H2Custom"))
story += bullets([
    "The closest danger object receives urgent spoken feedback, a spatialized sound, and a danger haptic pulse.",
    "Warning objects receive speech and spatial sound, with less urgency.",
    "When no obstacle is present, the system announces that the path is clear.",
    "When navigation is clear, an optional Gemma model through Ollama can create a short scene description for context.",
])

story.append(p("4. OCR text reading", "H1Custom"))
story.append(p("Technologies: PaddleOCR PP-OCRv4, PaddlePaddle, OpenCV, optional CUDA acceleration."))
story += bullets([
    "A background probe estimates whether readable text is present and measures the nearest text region with the depth map.",
    "OCR mode starts when text is sufficiently close and the user is stable.",
    "Preprocessing includes brightness-aware CLAHE, gamma correction for overexposure, Hough-line skew correction, unsharp masking, and bilateral denoising.",
    "PaddleOCR detects and recognizes text. It supports English, Arabic, or both according to configuration.",
    "Low-confidence, very small, or non-text-like detections are rejected. Arabic and English detections are overlap-deduplicated.",
    "Stable, cleaned text is spoken. After five seconds with no reading, the user is asked to adjust distance or camera angle.",
])
story.append(PageBreak())

story.append(p("5. Face identification and registration", "H1Custom"))
story.append(p("Technologies: InsightFace buffalo_sc, RetinaFace, ArcFace 512-dimensional embeddings, cosine similarity, SQLAlchemy, SQLite."))
story.append(p("Recognition flow", "H2Custom"))
story += bullets([
    "A low-resolution face probe runs during navigation to reduce compute cost.",
    "After stable detection, full-frame InsightFace processing detects the strongest face and produces an ArcFace embedding.",
    "The embedding is compared with stored face embeddings using cosine similarity.",
    "A match needs both a minimum similarity score and a margin above the second-best candidate.",
    "The result must remain stable across multiple frames before speech output.",
    "For a known person, the database updates the last-seen time, meeting count, and event log. Otherwise the user hears that the person is unknown.",
])
story.append(p("Registration flow", "H2Custom"))
story += bullets([
    "The user presses 6, types a name in the camera window, and confirms it with Enter.",
    "The system captures a short burst of live frames over approximately two seconds.",
    "Only clear, large, high-confidence face samples are retained.",
    "The best normalized embeddings are averaged and saved in SQLite.",
    "The recognition memory is refreshed immediately, so the newly registered face is available in the current session.",
])

story.append(p("6. Egyptian banknote recognition", "H1Custom"))
story.append(p("Technologies: custom YOLOv8 classification/detection model, PyTorch, temporal majority voting."))
story += bullets([
    "A periodic detector checks whether a banknote is visible.",
    "When a note is repeatedly visible and user motion is sufficiently low, the application enters Banknote mode.",
    "The custom YOLOv8 model detects and classifies the note denomination above a confidence threshold.",
    "A short history buffer confirms stable classifications before speech output.",
    "Each denomination is announced once, then detection pauses while speech completes and a cooldown expires.",
])

story.append(p("7. Hand-to-object interaction guidance", "H1Custom"))
story.append(p("Technologies: MediaPipe Hands, YOLOv8 detection reuse, stereo depth, NumPy geometry, HTTP requests."))
story += bullets([
    "Navigation detections are filtered for interactable objects such as cups, bottles, phones, remotes, switches, and door handles.",
    "In Interaction mode, MediaPipe Hands provides the dominant hand's index-fingertip location.",
    "The application calculates horizontal, vertical, and depth offsets from the fingertip to the target-object center.",
    "A small interaction state machine chooses IDLE, GUIDANCE, EDGE, or SUCCESS.",
    "The direction is encoded as a 4 x 5 haptic grid, converted to a 20-bit pattern, and sent asynchronously to a local HTTP endpoint.",
    "Once the hand remains within the success threshold for the required dwell time, the wristband pulses and the system says 'Object reached.'",
])
story.append(PageBreak())

story.append(p("8. Audio, haptics, storage, and reliability", "H1Custom"))
story.append(p("Audio feedback", "H2Custom"))
story.append(p(
    "Audio uses pyttsx3, Windows SAPI where available, and pygame. Speech is queued by priority so urgent obstacle warnings can be heard before lower-priority notifications. Requests expire after a time-to-live period to prevent stale speech. Pygame supplies danger, warning, and chime sounds, and object angle is converted into left/right stereo panning."))
story.append(p("Haptic feedback", "H2Custom"))
story.append(p(
    "The interaction feature sends 20-bit patterns to a local HTTP bridge at /api/pattern. If the bridge is unavailable, the module falls back to a log-only stub. The project also includes a separate generic ESP32 haptic abstraction with stub, serial, BLE placeholder, and Wi-Fi UDP options; its current default is stub mode."))
story.append(p("Persistent storage", "H2Custom"))
story.append(p(
    "SQLAlchemy manages the local SQLite database. The database stores people, face embeddings, registration metadata, last-seen information, preferences, and event logs. Face embeddings are stored as 512-dimensional float32 vectors compatible with the InsightFace model."))
story.append(p("Configuration and operational controls", "H2Custom"))
story.append(p(
    "Central configuration defines model paths, confidence thresholds, camera resolution, safety distances, OCR languages, state dwell times, audio settings, and hardware constants. Pydantic Settings loads values from a .env file. The command line supports automatic or manual operation, headless execution, audio disabling, debug logging, session logs, OCR language selection, and face-tolerance adjustment."))
story.append(p("Implementation conclusion", "H1Custom"))
story.append(p(
    "ECHORA follows a modular, real-time pipeline: capture the user's environment, interpret it with specialized AI models, select the most useful task through a safety-first state machine, and return the result through accessible audio and tactile channels. This design keeps the system extensible while prioritizing timely, understandable feedback for real-world use.", "Callout"))

doc = SimpleDocTemplate(
    str(OUTPUT), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm,
    topMargin=18*mm, bottomMargin=20*mm, title="ECHORA Implementation Workflow",
    author="ECHORA Project",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUTPUT)
