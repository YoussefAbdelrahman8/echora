from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf" / "echora_real_system_analysis_and_tests.pdf"

NAVY = colors.HexColor("#123047")
TEAL = colors.HexColor("#007C78")
RED = colors.HexColor("#A82525")
AMBER = colors.HexColor("#8A5700")
PALE = colors.HexColor("#EAF3F8")
INK = colors.HexColor("#1E2932")
MUTED = colors.HexColor("#51606C")

s = getSampleStyleSheet()
s.add(ParagraphStyle(name="TitleX", parent=s["Title"], fontName="Helvetica-Bold", fontSize=25, leading=30, textColor=NAVY, alignment=TA_CENTER, spaceAfter=8))
s.add(ParagraphStyle(name="SubX", parent=s["Normal"], fontName="Helvetica", fontSize=10.5, leading=14, textColor=MUTED, alignment=TA_CENTER, spaceAfter=18))
s.add(ParagraphStyle(name="H1X", parent=s["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=NAVY, spaceBefore=9, spaceAfter=7))
s.add(ParagraphStyle(name="H2X", parent=s["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=TEAL, spaceBefore=7, spaceAfter=4))
s.add(ParagraphStyle(name="BodyX", parent=s["BodyText"], fontName="Helvetica", fontSize=8.9, leading=12.6, textColor=INK, spaceAfter=4))
s.add(ParagraphStyle(name="SmallX", parent=s["BodyText"], fontName="Helvetica", fontSize=8.1, leading=10.5, textColor=INK))
s.add(ParagraphStyle(name="CalloutX", parent=s["BodyText"], fontName="Helvetica-Bold", fontSize=9.1, leading=13, textColor=NAVY, backColor=PALE, borderColor=colors.HexColor("#B8D6E6"), borderWidth=.5, borderPadding=7, spaceBefore=5, spaceAfter=8))
s.add(ParagraphStyle(name="RiskX", parent=s["BodyText"], fontName="Helvetica-Bold", fontSize=8.9, leading=12.6, textColor=RED, backColor=colors.HexColor("#FDECEC"), borderColor=colors.HexColor("#F2BABA"), borderWidth=.5, borderPadding=7, spaceBefore=5, spaceAfter=8))

def p(text, style="BodyX"):
    return Paragraph(text, s[style])

def bullets(items):
    return [p("- " + item) for item in items]

def grid(headers, rows, widths):
    data = [[p(h, "SmallX") for h in headers]] + [[p(x, "SmallX") for x in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#C7D5DD")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F9FB")]),
        ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    return t

def footer(c, doc):
    c.saveState(); c.setStrokeColor(colors.HexColor("#B8D6E6"))
    c.line(doc.leftMargin, 13*mm, A4[0]-doc.rightMargin, 13*mm)
    c.setFont("Helvetica", 8); c.setFillColor(MUTED)
    c.drawString(doc.leftMargin, 8.5*mm, "ECHORA - Real System Analysis and Test Evidence")
    c.drawRightString(A4[0]-doc.rightMargin, 8.5*mm, f"Page {doc.page}")
    c.restoreState()

story = [Spacer(1, 24*mm), p("ECHORA", "TitleX"), p("Real Implementation Analysis and Test Evidence", "SubX")]
story.append(p("This report is evidence-based: statements about behaviour are traced to the current codebase, while test status records the actual outcome of the test command executed in this workspace.", "CalloutX"))
story.append(p("1. Scope and evidence", "H1X"))
story.append(p("The analysis reviewed the live source code under src/, the active configuration, the runtime tests under tests/, and the result of running pytest. It does not claim that camera hardware, ML inference, Ollama, an ESP32 wristband, or audio hardware were validated because those integrations were not runnable in the available test environment."))
story.append(grid(["Evidence type", "What it establishes", "Status"], [
    ("Source inspection", "Current data flow, thresholds, module dependencies, and integration paths", "Completed"),
    ("Test inventory", "23 pytest runtime test functions across OCR, obstacles, interaction, banknotes, and audio", "Completed"),
    ("Automated test execution", "pytest collection attempted using bundled Python", "Blocked by missing dependencies"),
    ("Hardware/model validation", "OAK-D, trained models, TTS, VLM, ESP32", "Not executed"),
], [43*mm, 88*mm, 38*mm]))
story.append(p("Architecture in one sentence", "H2X"))
story.append(p("ECHORA is a Python real-time control loop that turns synchronized RGB, depth, and motion data into feature-specific AI results, then selects the safest feedback mode and communicates through speech, spatial sound, and haptics."))
story.append(PageBreak())

story.append(p("2. Real runtime architecture", "H1X"))
story.append(p("The application starts in main.py and constructs ControlUnit. Startup initializes the OAK-D pipeline, PaddleOCR, dual YOLO obstacle detection, MediaPipe Hands, a banknote model, SQLite, InsightFace, haptic services, audio, and the StateMachine. The ControlUnit then loops over synchronized camera bundles."))
story.append(p("Actual per-frame ordering", "H2X"))
story += bullets([
    "Read RGB frame, aligned depth map, IMU sample, and timestamp from the OAK-D pipeline.",
    "Run the obstacle detector first and obtain tracked obstacle and interaction-object detections.",
    "Run the state machine using probe results retained from previous sampling ticks.",
    "Launch or update periodic probes for OCR, face presence, banknote presence, and interactable-object distance.",
    "Execute the active mode handler: Navigation, OCR, Interaction, Face ID, or Banknote.",
    "Draw a debug overlay, collect frame timing, and process user keys when a display is enabled.",
])
story.append(p("Mode selection is safety-first", "H2X"))
story.append(p("Navigation is the default. OCR requires text within 2 m and low motion; Interaction requires an interactable object within 0.8 m; Face ID requires repeated confident face detection; Banknote requires repeated note detection and low motion. The StateMachine uses consecutive-frame counters and dwell times. A danger obstacle forces a return to Navigation in all specialized modes except the person-only Face ID exception."))
story.append(p("Observed implementation detail", "H2X"))
story.append(p("The state machine is evaluated before the current iteration updates its probes. Therefore a new OCR, face, banknote, or interaction signal is normally consumed on a later frame. This is a small, deliberate-looking latency trade-off that keeps heavyweight probes asynchronous, but it should be described as sampled rather than instantaneous behaviour."))
story.append(PageBreak())

story.append(p("3. Feature-level analysis", "H1X"))
story.append(grid(["Feature", "Actual implementation", "Feedback/output", "Test coverage"], [
    ("Navigation", "Custom accessibility YOLOv8 plus optional COCO YOLOv8; depth sampled per bounding box; ByteTracker confirms tracks and estimates movement.", "Closest danger/warning object is spoken with direction; danger adds spatial sound and checkerboard haptic pulse.", "7 runtime tests cover corridor promotion, unknown depth, dual-model merge, track confirmation/expiry, and VLM failure disabling."),
    ("OCR", "PaddleOCR PP-OCRv4 with optional Arabic and English models. Uses contrast correction, skew correction, sharpening, denoising, caching, ordering, and stability history.", "Stable text is spoken. After about 5 s with no text, a repositioning prompt is spoken.", "9 runtime tests cover language choice, GPU fallback, cache behaviour, normalization, distance priority, and document order."),
    ("Face ID", "InsightFace buffalo_sc: RetinaFace detection and ArcFace 512-D embeddings. Cosine similarity plus runner-up margin; SQLite stores profiles and events.", "Known person or unknown-person speech; successful identification updates seen count and event log.", "No active pytest runtime test file. Legacy self-test is excluded by conftest."),
    ("Banknote", "Dedicated custom YOLOv8 model. Highest-confidence label enters a small temporal history before confirmation.", "Stable Egyptian Pound denomination is spoken once, followed by speech-completion cooldown.", "2 runtime tests cover one announcement per scan and reset behaviour."),
    ("Interaction", "Reuses obstacle detections for targets, then MediaPipe Hands finds the index fingertip. RGB/depth geometry calculates dx, dy, dz to target.", "4 x 5 electrode grid becomes a 20-bit HTTP pattern. Success causes a pulse and speech.", "4 runtime tests cover grid size, phase thresholds, dwell time, and invalid-depth filtering."),
    ("Audio", "pyttsx3 or Windows SAPI worker with a priority queue; pygame provides alert tones and stereo panning.", "Asynchronous prioritized speech; expiry avoids late messages.", "1 runtime test covers banknote speech completion event and label wording."),
], [25*mm, 55*mm, 45*mm, 44*mm]))
story.append(PageBreak())

story.append(p("4. Detailed operational analysis", "H1X"))
story.append(p("Navigation", "H2X"))
story.append(p("Detected objects are enriched with depth, viewing angle, urgency, and tracking data. Danger is nominally below 1.5 m; warning spans 1.5-3 m. A warning obstacle can be promoted to danger when it overlaps the collision corridor and is close enough. Interaction-only classes are deliberately removed from user-facing obstacle alerts but retained for object-reaching guidance. The VLM scene description is started only when there are no danger or warning obstacles, preventing conversational context from competing with safety alerts."))
story.append(p("OCR", "H2X"))
story.append(p("OCR is architected to avoid stalling the camera loop. The ControlUnit launches the reading operation in a daemon thread, while OCRReader itself uses an inference lock. Results are filtered by confidence, text height, and alphanumeric density, then ordered into readable lines. Arabic output reverses block order when it is predominantly Arabic, allowing text-to-speech to follow right-to-left reading order."))
story.append(p("Face ID and storage", "H2X"))
story.append(p("Face registration captures frames for roughly two seconds, filters weak, small, or blurry detections, averages the best compatible embeddings, normalizes the result, and persists it under a unique name. Identification compares a live embedding against all stored embeddings. This is a local-first design: no face image or embedding is sent to a remote service by this code path."))
story.append(p("Interaction and haptics", "H2X"))
story.append(p("The interaction pipeline computes a 3-D fingertip-to-target offset. It has four phases: IDLE when a hand or target is missing, GUIDANCE for general direction, EDGE when image/depth alignment is close, and SUCCESS when both image-space and depth thresholds are met. HTTP sends are asynchronous and duplicate haptic patterns are suppressed to reduce local-server traffic."))
story.append(PageBreak())

story.append(p("5. Test execution results", "H1X"))
story.append(p("Command executed", "H2X"))
story.append(p("python -m pytest -q, using the bundled Python 3.12 environment after the project virtual environment was found to reference a missing Python 3.10 installation."))
story.append(grid(["Metric", "Actual result"], [
    ("Discovered active runtime tests", "23 test functions in five files"),
    ("Executed tests", "0 - pytest stopped during test collection"),
    ("Collection errors", "5"),
    ("Elapsed time", "0.27 seconds"),
    ("Overall result", "NOT PASSING in the available environment; no unit-test pass claim can be made"),
], [58*mm, 112*mm]))
story.append(p("Collection blockers", "H2X"))
story.append(grid(["Test area", "Blocking missing module"], [
    ("Audio runtime", "pyttsx3"),
    ("Banknote runtime", "cv2"),
    ("Interaction runtime", "pydantic_settings"),
    ("Obstacle runtime", "pydantic_settings"),
    ("OCR runtime", "cv2"),
], [70*mm, 100*mm]))
story.append(p("Interpretation", "H2X"))
story.append(p("This outcome is an environment failure, not evidence that the five features are logically failing. The test suite could not import the application modules, so no assertions were executed. The stale .venv is also a reproducibility problem: its interpreter path points to a removed Python 3.10 installation. Full test results require rebuilding the environment from requirements.txt with a supported interpreter, then rerunning pytest."))
story.append(p("What the tests are designed to validate", "H2X"))
story += bullets([
    "OCR language routing, GPU fallback, caching, stable-text normalization, depth prioritization, and reading order.",
    "Obstacle urgency promotion, invalid-depth handling, merging two detector outputs, tracker confirmation, and VLM failure shutdown.",
    "Interaction grid dimensions, phase boundaries, success dwell time, and invalid-depth rejection.",
    "Banknote one-time announcement and reset behaviour; audio speech-completion signalling.",
])
story.append(PageBreak())

story.append(p("6. Findings and implementation risks", "H1X"))
story.append(p("Finding 1 - Haptic implementations are inconsistent", "H2X"))
story.append(p("The active interaction path uses a safe 4 x 5 grid and a 20-bit HTTP bridge. A separate generic haptic_feedback module also starts at application startup, but it is configured for a 4 x 5 grid while its pattern_right function writes column 5 and pattern_down writes row 4. Both indices are out of range for a 4 x 5 array. The generic module defaults to stub mode, which reduces immediate runtime impact, but its directional patterns will fail if those functions are used.", "RiskX"))
story.append(p("Finding 2 - Banknote distance requirement is not enforced in the active mode flow", "H2X"))
story.append(p("The configuration defines BANKNOTE_MAX_DIST_MM = 500 and the BanknoteDetector exposes is_note_in_range(). However, the ControlUnit's banknote probe calls detect_banknote() and the Banknote handler calls classify_denomination(); neither calls is_note_in_range(). Consequently, banknote mode can be entered and a denomination announced without the configured distance gate being applied.", "RiskX"))
story.append(p("Finding 3 - Test infrastructure is currently non-reproducible", "H2X"))
story.append(p("The checked-in virtual environment cannot launch because it targets a missing Python 3.10 executable. The alternative bundled interpreter lacked runtime dependencies. Automated tests therefore cannot currently prove correctness until the environment is rebuilt.", "RiskX"))
story.append(p("Finding 4 - Test scope is incomplete for key integrations", "H2X"))
story.append(p("The active pytest suite has strong logic coverage for OCR, obstacles, interaction, banknotes, and audio, but it has no active runtime tests for Face ID, the ControlUnit orchestration loop, camera acquisition, actual model loading/inference, the local haptic endpoint, or real audio playback. Several older standalone self-tests are explicitly excluded from pytest collection."))
story.append(p("Recommended next validation steps", "H2X"))
story += bullets([
    "Recreate .venv with a supported Python version and install requirements.txt; rerun pytest and record pass/fail counts.",
    "Fix or consolidate the two haptic implementations, then add tests for right/down directional patterns and endpoint payloads.",
    "Call is_note_in_range() before entering or processing Banknote mode if close-range scanning is a product requirement.",
    "Add mocked ControlUnit integration tests for priority transitions and hardware-adapter failure paths.",
    "Run a hardware acceptance test with OAK-D, audio output, haptic bridge, trained models, and Ollama; capture FPS, latency, recognition accuracy, and safety-alert response time.",
])
story.append(p("Conclusion", "H1X"))
story.append(p("The codebase implements a credible modular sensory-assistance pipeline with thoughtful asynchronous processing, safety-prioritized mode selection, and useful unit-level test design. Its current limitation is verification readiness rather than a lack of feature implementation: the environment blocks test execution, and the haptic and banknote-distance paths contain concrete implementation issues that should be resolved before claiming system-level validation.", "CalloutX"))

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=18*mm, bottomMargin=20*mm, title="ECHORA Real System Analysis and Test Evidence", author="ECHORA Project")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
