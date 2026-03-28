# =============================================================================
# kalman_tracker.py — ECHORA Multi-Object Tracker
# =============================================================================
# Manages one Kalman filter per detected object.
# Handles the full lifecycle: detected → confirmed → lost → deleted.
# Matches new YOLO detections to existing tracks using IoU.
# Produces clean track dictionaries consumed by obstacle_detection.py
# and control_unit.py.
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

# numpy for matrix operations inside each Kalman filter.
import numpy as np

# cv2 for the KalmanFilter class — OpenCV has a built-in implementation.
import cv2

# time for timestamping when each track was last seen.
import time

# Type hints — make the code easier to read.
from typing import List, Dict, Optional, Tuple

# Our constants and helpers.
from config import (
    KALMAN_PROCESS_NOISE,
    KALMAN_MEASUREMENT_NOISE,
    KALMAN_MAX_MISSED_FRAMES,
)
from utils import (
    logger,
    bbox_center,
    angle_from_x,
    classify_urgency,
    get_timestamp_ms,
)


# =============================================================================
# TRACK STATE CONSTANTS
# =============================================================================

class TrackState:
    """
    Constants for the four possible states of a tracked object.

    Using a class instead of plain strings means:
      - You write TrackState.CONFIRMED instead of "CONFIRMED"
      - If you make a typo, Python tells you immediately
      - All valid states are documented in one place

    These states form a one-way lifecycle:
      DETECTED → CONFIRMED → LOST → DELETED
    (A track can also go CONFIRMED → LOST → CONFIRMED if it reappears)
    """

    # Object was just seen for the first time this session.
    # Not yet reported to the rest of ECHORA — might be a false positive.
    # Becomes CONFIRMED after MIN_HITS consecutive detections.
    DETECTED  = "DETECTED"

    # Object has been seen reliably for enough consecutive frames.
    # Now reported to obstacle_detection, control_unit, audio_feedback.
    CONFIRMED = "CONFIRMED"

    # Object was not detected this frame.
    # Kalman filter is predicting its position.
    # Will be deleted after KALMAN_MAX_MISSED_FRAMES consecutive misses.
    LOST      = "LOST"

    # Object has been missing too long. Track will be removed.
    # Once DELETED, the track object is discarded on the next cleanup.
    DELETED   = "DELETED"


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# How many consecutive frames a detection must appear before we confirm it.
# 2 means: seen frame 1 AND frame 2 → confirmed. Prevents false positives.
MIN_HITS_TO_CONFIRM = 2

# Minimum IoU score to consider a detection and a track as the same object.
# 0.3 means at least 30% overlap. Below this = different objects.
IOU_THRESHOLD = 0.3

# Maximum distance in pixels between detection center and track center
# to even attempt IoU matching. Prevents matching objects on opposite
# sides of the frame just because IoU happened to be above threshold.
MAX_CENTER_DISTANCE_PX = 200


# =============================================================================
# TRACK CLASS
# =============================================================================

class Track:
    """
    Represents one tracked object with its own Kalman filter.

    Created when YOLO first detects an object.
    Updated every frame with new detections (or marked as lost).
    Deleted when the object has been missing too long.

    Each Track has a unique ID like "person_001", "chair_002".
    """

    def __init__(
        self,
        track_id: str,
        bbox: Tuple[int, int, int, int],
        label: str,
        confidence: float,
        distance_mm: float,
        frame_width: int
    ):
        """
        Creates a new Track for a freshly detected object.

        Arguments:
            track_id:     unique string ID, e.g. "person_001"
            bbox:         bounding box (x1, y1, x2, y2) in pixels
            label:        YOLO class name, e.g. "person", "chair"
            confidence:   YOLO detection confidence 0.0 to 1.0
            distance_mm:  depth reading at this object's position in mm
            frame_width:  width of the camera frame in pixels (for angle calc)
        """

        # ── Identity ──────────────────────────────────────────────────────────
        # Store the unique ID for this track.
        self.track_id    = track_id

        # The YOLO class label — what kind of object this is.
        self.label       = label

        # How confident YOLO was about this detection (0.0 to 1.0).
        self.confidence  = confidence

        # ── State ─────────────────────────────────────────────────────────────
        # New tracks start as DETECTED, not immediately CONFIRMED.
        # This prevents false positives from being announced.
        self.state = TrackState.DETECTED

        # ── Position ─────────────────────────────────────────────────────────
        # Current bounding box in pixel coordinates.
        self.bbox = bbox

        # Depth at this object's location in millimetres.
        self.distance_mm = distance_mm

        # Width of the camera frame — needed to calculate the angle.
        self.frame_width = frame_width

        # ── Counters ──────────────────────────────────────────────────────────
        # How many consecutive frames this track has been successfully detected.
        # When this reaches MIN_HITS_TO_CONFIRM, state becomes CONFIRMED.
        self.hits = 1   # starts at 1 because we just detected it

        # How many consecutive frames this track has been MISSED (not detected).
        # When this exceeds KALMAN_MAX_MISSED_FRAMES, state becomes DELETED.
        self.missed = 0

        # Total number of frames this track has existed (for diagnostics).
        self.age = 1

        # When this track was first created — Unix timestamp in seconds.
        self.created_at = time.time()

        # When this track was last updated with a real detection.
        self.last_seen_ms = get_timestamp_ms()

        # ── Kalman Filter ─────────────────────────────────────────────────────
        # Each track gets its OWN Kalman filter instance.
        # This is key — 5 tracked objects = 5 independent Kalman filters.
        self._kalman = self._init_kalman(bbox)

        logger.debug(f"Track created: {track_id} ({label}) at bbox={bbox}")


    def _init_kalman(self, bbox: Tuple[int, int, int, int]) -> cv2.KalmanFilter:
        """
        Creates and initialises a Kalman filter for this specific track.

        The underscore prefix means this is private — only called internally.

        State vector:   [cx, cy, vx, vy]
          cx, cy = center x and y of the bounding box in pixels
          vx, vy = velocity in pixels per frame

        Measurement:    [cx, cy]
          We can only observe position, not velocity.

        The initial state is set from the first detection's bounding box.
        """

        # Create a Kalman filter with:
        #   4 dynamic parameters (state: cx, cy, vx, vy)
        #   2 measurement parameters (observation: cx, cy)
        kf = cv2.KalmanFilter(4, 2)

        # Transition matrix — how state evolves between frames.
        # Encodes: position_new = position_old + velocity × 1_frame
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],   # cx_new = cx + vx
            [0, 1, 0, 1],   # cy_new = cy + vy
            [0, 0, 1, 0],   # vx_new = vx (constant velocity)
            [0, 0, 0, 1],   # vy_new = vy (constant velocity)
        ], dtype=np.float32)

        # Measurement matrix — maps state [cx,cy,vx,vy] to observation [cx,cy].
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],   # we observe cx
            [0, 1, 0, 0],   # we observe cy
        ], dtype=np.float32)

        # Process noise — uncertainty in our physics model.
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE

        # Measurement noise — uncertainty in the camera sensor.
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE

        # Initial error covariance — starting uncertainty about the state.
        # We multiply by 10 for vx, vy because we have no idea about initial
        # velocity yet. High initial uncertainty for velocity makes the filter
        # accept the first few measurements more readily.
        kf.errorCovPost = np.array([
            [1, 0, 0,  0 ],
            [0, 1, 0,  0 ],
            [0, 0, 10, 0 ],   # high initial uncertainty for vx
            [0, 0, 0,  10],   # high initial uncertainty for vy
        ], dtype=np.float32)

        # Set the initial state from the first detection.
        # bbox_center returns (cx, cy) — the center of the bounding box.
        cx, cy = bbox_center(*bbox)

        # statePost is the Kalman filter's current best estimate of the state.
        # We initialise it with the first measurement.
        # Shape must be (4, 1) — a column vector.
        # float32 is required by OpenCV.
        kf.statePost = np.array(
            [[float(cx)],   # cx — initial x position
             [float(cy)],   # cy — initial y position
             [0.0],          # vx — assume no initial velocity
             [0.0]],         # vy — assume no initial velocity
            dtype=np.float32
        )

        return kf


    def predict(self):
        """
        Advances this track's Kalman filter by one frame.

        Call this every frame for every track, before doing any matching.
        This gives the filter's best guess of where this object is NOW,
        based on where it was and how fast it was moving.

        After predict(), self.bbox is updated to the predicted position.
        """

        # Run the Kalman prediction step.
        # predicted shape: (4, 1) — [cx, cy, vx, vy]
        predicted = self._kalman.predict()

        # Extract the predicted center position.
        pred_cx = float(predicted[0][0])
        pred_cy = float(predicted[1][0])

        # Update the bounding box to the predicted position.
        # We keep the same width and height — only the center moves.
        x1, y1, x2, y2 = self.bbox
        w = x2 - x1   # bounding box width
        h = y2 - y1   # bounding box height

        # Compute new bounding box centered on the predicted position.
        # int() truncates the float to a whole pixel number.
        new_x1 = int(pred_cx - w / 2)
        new_y1 = int(pred_cy - h / 2)
        new_x2 = int(pred_cx + w / 2)
        new_y2 = int(pred_cy + h / 2)

        self.bbox = (new_x1, new_y1, new_x2, new_y2)

        # Increment the total age of this track.
        self.age += 1


    def update(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        distance_mm: float
    ):
        """
        Updates this track with a new matched detection.

        Call this when a YOLO detection has been matched to this track.
        Feeds the new measured position into the Kalman filter and
        updates all the track's metadata.

        Arguments:
            bbox:        new bounding box from YOLO (x1, y1, x2, y2)
            confidence:  new YOLO confidence score
            distance_mm: new depth reading in mm
        """

        # Get the center of the new detection.
        cx, cy = bbox_center(*bbox)

        # Feed the measurement into the Kalman filter.
        # correct() blends prediction + measurement → best estimate.
        measurement = np.array([[float(cx)], [float(cy)]], dtype=np.float32)
        self._kalman.correct(measurement)

        # Update the bounding box with the new actual detection.
        # We use the real YOLO box (not the Kalman-predicted one) because
        # YOLO gives us the actual size which the Kalman filter doesn't track.
        self.bbox        = bbox
        self.confidence  = confidence
        self.distance_mm = distance_mm

        # Reset the missed counter — we found this object again.
        self.missed = 0

        # Increment the consecutive hit counter.
        self.hits += 1

        # Record when we last saw this object.
        self.last_seen_ms = get_timestamp_ms()

        # Promote from DETECTED to CONFIRMED if we've seen it enough times.
        # Until confirmed, the rest of ECHORA ignores this track.
        if self.state == TrackState.DETECTED and self.hits >= MIN_HITS_TO_CONFIRM:
            self.state = TrackState.CONFIRMED
            logger.debug(f"Track confirmed: {self.track_id} ({self.label})")

        # If the track was LOST but the object reappeared, restore it.
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED
            logger.debug(f"Track recovered: {self.track_id} ({self.label})")


    def mark_lost(self):
        """
        Called when no detection was matched to this track this frame.

        Increments the missed counter.
        If missed too many frames → marks as DELETED.
        """

        # Increment the consecutive missed frames counter.
        self.missed += 1

        # If we just transitioned from confirmed/detected to lost, log it.
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST
            logger.debug(f"Track lost: {self.track_id} ({self.label}), missed={self.missed}")

        # If we've been lost for too many frames, mark for deletion.
        # The tracker will remove this track on the next cleanup pass.
        if self.missed > KALMAN_MAX_MISSED_FRAMES:
            self.state = TrackState.DELETED
            logger.debug(f"Track deleted: {self.track_id} ({self.label})")


    def is_confirmed(self) -> bool:
        """
        Returns True if this track is in CONFIRMED state.

        Only confirmed tracks are reported to the rest of ECHORA.
        DETECTED tracks might be false positives.
        LOST tracks are still being predicted but not acted upon.
        DELETED tracks are about to be removed.
        """
        return self.state == TrackState.CONFIRMED


    def get_predicted_center(self) -> Tuple[float, float]:
        """
        Returns the Kalman filter's current best estimate of center position.

        This is the smoothed position — more stable than the raw bbox center.
        Used by the IoU matching to predict where this track will be.
        """

        # statePost is the Kalman filter's current best state estimate.
        # [0][0] = cx (x position), [1][0] = cy (y position)
        cx = float(self._kalman.statePost[0][0])
        cy = float(self._kalman.statePost[1][0])
        return cx, cy


    def to_dict(self) -> Dict:
        """
        Exports this track's current state as a dictionary.

        This is the format consumed by:
          - obstacle_detection.py  (to check urgency)
          - control_unit.py        (to decide what feedback to give)
          - audio_feedback.py      (to speak the object name and distance)
          - draw_overlay() in utils.py (to draw on the debug frame)
        """

        # Get the current center of the bounding box.
        cx, cy = bbox_center(*self.bbox)

        # Convert pixel x position to real-world angle in degrees.
        # Negative = left of center, positive = right of center.
        angle = angle_from_x(cx, self.frame_width)

        # Classify the distance into DANGER / WARNING / SAFE / UNKNOWN.
        urgency = classify_urgency(self.distance_mm)

        return {
            "id":          self.track_id,
            "label":       self.label,
            "state":       self.state,
            "bbox":        self.bbox,
            "center":      (cx, cy),
            "distance_mm": self.distance_mm,
            "angle_deg":   angle,
            "urgency":     urgency,
            "confidence":  round(self.confidence, 3),
            "hits":        self.hits,
            "missed":      self.missed,
            "age":         self.age,
            "last_seen_ms": self.last_seen_ms,
        }


# =============================================================================
# KALMAN TRACKER CLASS
# =============================================================================

class KalmanTracker:
    """
    Manages a collection of Track objects — one per detected object.

    Every frame:
      1. Predict new positions for all existing tracks
      2. Match new YOLO detections to existing tracks (using IoU)
      3. Update matched tracks with new measurements
      4. Create new tracks for unmatched detections
      5. Mark unmatched tracks as lost
      6. Delete tracks that have been lost too long

    Usage:
        tracker = KalmanTracker(frame_width=1280)

        while True:
            bundle   = cam.get_synced_bundle()
            raw_dets = yolo.detect(bundle["rgb"], bundle["depth"])
            tracks   = tracker.update(raw_dets)

            for track in tracks:
                print(track["label"], track["distance_mm"], track["urgency"])
    """

    def __init__(self, frame_width: int):
        """
        Creates an empty tracker.

        Arguments:
            frame_width: width of the camera frame in pixels.
                         Needed to calculate angles for each track.
        """

        # List of all currently active Track objects.
        # Contains tracks in all states: DETECTED, CONFIRMED, LOST.
        # DELETED tracks are removed during the update() cleanup.
        self.tracks: List[Track] = []

        # Counter for generating unique track IDs.
        # Incremented each time a new track is created.
        # Never reused — even if a track is deleted, its ID is gone forever.
        # This prevents confusion if an old ID reappears.
        self._next_id: int = 0

        # Width of the camera frame — stored for angle calculations.
        self.frame_width = frame_width

        logger.info(f"KalmanTracker initialised (frame_width={frame_width}px).")


    def _generate_id(self, label: str) -> str:
        """
        Generates a unique, human-readable track ID.

        Format: "label_NNN" where NNN is a zero-padded counter.
        Examples: "person_001", "chair_002", "bottle_003"

        Arguments:
            label: the YOLO class name for this detection

        Returns:
            A unique ID string.
        """

        # Increment the counter first so IDs start at 001 not 000.
        self._next_id += 1

        # f-string with :03d formats the number as 3 digits with leading zeros.
        # 1 → "001", 23 → "023", 100 → "100"
        return f"{label}_{self._next_id:03d}"


    def _iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculates Intersection over Union (IoU) between two bounding boxes.

        IoU = area of overlap / area of union

        Returns a float between 0.0 (no overlap) and 1.0 (identical boxes).

        This is the core metric used to decide if a detection and a track
        represent the same physical object.

        Arguments:
            box1, box2: each is (x1, y1, x2, y2) in pixel coordinates
        """

        # Unpack both boxes into their corner coordinates.
        ax1, ay1, ax2, ay2 = box1
        bx1, by1, bx2, by2 = box2

        # ── Calculate Intersection ────────────────────────────────────────────
        # The intersection rectangle's top-left corner is the MAXIMUM of
        # both boxes' top-left corners.
        # The intersection rectangle's bottom-right is the MINIMUM of both
        # boxes' bottom-right corners.
        #
        # Visually:
        #   box1: starts at ax1, box2 starts at bx1
        #   The overlap can only start where BOTH boxes have started.
        #   So intersection starts at max(ax1, bx1).
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        # Calculate intersection width and height.
        # If inter_x2 < inter_x1, the boxes don't overlap horizontally.
        # max(0, ...) ensures we get 0 (not negative) in that case.
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)

        # Intersection area = width × height.
        # If either dimension is 0, area is 0 (no overlap).
        inter_area = inter_w * inter_h

        # ── Calculate Union ───────────────────────────────────────────────────
        # Union = area of box1 + area of box2 - intersection area.
        # We subtract intersection once because it was counted twice
        # (once in box1's area and once in box2's area).
        area1 = (ax2 - ax1) * (ay2 - ay1)
        area2 = (bx2 - bx1) * (by2 - by1)
        union_area = area1 + area2 - inter_area

        # ── Calculate IoU ─────────────────────────────────────────────────────
        # Avoid division by zero if union_area is somehow 0.
        if union_area <= 0:
            return 0.0

        return float(inter_area / union_area)


    def _center_distance(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculates the Euclidean distance between the centers of two boxes.

        Used as a pre-filter before IoU — if two boxes are very far apart,
        we skip the IoU calculation entirely. This is an optimisation.

        Euclidean distance = sqrt( (cx2-cx1)² + (cy2-cy1)² )

        Arguments:
            box1, box2: each is (x1, y1, x2, y2)

        Returns:
            Distance in pixels between the two centers.
        """

        # Get center of each box.
        cx1, cy1 = bbox_center(*box1)
        cx2, cy2 = bbox_center(*box2)

        # Euclidean distance formula.
        # ** 2 means "squared", ** 0.5 means "square root".
        # This is equivalent to math.sqrt((dx)² + (dy)²).
        dx = cx2 - cx1
        dy = cy2 - cy1
        return (dx ** 2 + dy ** 2) ** 0.5


    def _match_detections_to_tracks(
        self,
        detections: List[Dict]
    ) -> Tuple[List, List, List]:
        """
        Matches new YOLO detections to existing tracks using IoU.

        For each detection, finds the existing track it overlaps most with.
        If the best overlap is above IOU_THRESHOLD, they are matched.

        Returns three lists:
          matches:             list of (detection_index, track_index) pairs
          unmatched_dets:      detection indices with no matching track
          unmatched_tracks:    track indices with no matching detection

        These three lists tell update() what to do:
          matches          → update those tracks with the new measurements
          unmatched_dets   → create new tracks for these detections
          unmatched_tracks → mark those tracks as lost
        """

        # Edge case: if there are no existing tracks, all detections are new.
        if len(self.tracks) == 0:
            # All detections are unmatched — range(len(detections)) gives
            # [0, 1, 2, ...] — indices of all detections.
            return [], list(range(len(detections))), []

        # Edge case: if there are no detections, all tracks are unmatched.
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # ── Build the IoU matrix ──────────────────────────────────────────────
        # This is a 2D table: rows = detections, columns = tracks.
        # Each cell [i][j] = IoU between detection i and track j.
        #
        # Example with 2 detections and 3 tracks:
        #          track0  track1  track2
        # det0  [  0.85,   0.02,   0.00  ]
        # det1  [  0.01,   0.00,   0.76  ]
        #
        # det0 clearly matches track0 (0.85), det1 matches track2 (0.76).

        n_dets   = len(detections)
        n_tracks = len(self.tracks)

        # np.zeros creates a 2D array filled with 0.0.
        iou_matrix = np.zeros((n_dets, n_tracks), dtype=np.float32)

        for d_idx, det in enumerate(detections):
            det_box = det["bbox"]   # detection bounding box

            for t_idx, track in enumerate(self.tracks):
                track_box = track.bbox   # track's current (predicted) bbox

                # Pre-filter: skip IoU calculation if centers are too far apart.
                # This is an optimisation — IoU between far-apart boxes is 0.
                dist = self._center_distance(det_box, track_box)
                if dist > MAX_CENTER_DISTANCE_PX:
                    # Leave iou_matrix[d_idx][t_idx] as 0.0 — no match.
                    continue

                # Only try to match detections with the same label.
                # A "person" detection should never match a "chair" track.
                if det["label"] != track.label:
                    continue

                # Calculate and store the IoU score.
                iou_matrix[d_idx][t_idx] = self._iou(det_box, track_box)

        # ── Greedy matching ───────────────────────────────────────────────────
        # Find the best match for each detection using a greedy algorithm:
        #   1. Find the highest IoU in the entire matrix.
        #   2. If it's above threshold, record this as a match.
        #   3. Remove that detection and track from consideration.
        #   4. Repeat until no more valid matches exist.
        #
        # This is "greedy" because it always takes the best available match
        # first. It's not globally optimal but it's fast and works well
        # for real-time tracking with a small number of objects.

        matches          = []   # list of (det_idx, track_idx) tuples
        matched_dets     = set()  # set of detection indices already matched
        matched_tracks   = set()  # set of track indices already matched

        # Keep matching until no more valid pairs exist.
        # We loop at most min(n_dets, n_tracks) times — the maximum
        # number of possible matches.
        for _ in range(min(n_dets, n_tracks)):

            # Find the highest IoU score in the matrix.
            # np.unravel_index converts a flat index to (row, col).
            # np.argmax finds the index of the maximum value.
            max_iou = np.max(iou_matrix)

            # If the best remaining IoU is below our threshold, stop.
            # No more valid matches.
            if max_iou < IOU_THRESHOLD:
                break

            # Get the (row, col) = (detection_index, track_index) of the max.
            d_idx, t_idx = np.unravel_index(
                np.argmax(iou_matrix), iou_matrix.shape
            )

            # Convert numpy integers to regular Python integers.
            d_idx = int(d_idx)
            t_idx = int(t_idx)

            # Record this match.
            matches.append((d_idx, t_idx))

            # Mark both as used.
            matched_dets.add(d_idx)
            matched_tracks.add(t_idx)

            # Zero out this row and column so these won't be matched again.
            # Setting the entire row to 0 prevents this detection from
            # being matched to another track.
            # Setting the entire column to 0 prevents this track from
            # being matched to another detection.
            iou_matrix[d_idx, :] = 0.0
            iou_matrix[:, t_idx] = 0.0

        # ── Find unmatched detections and tracks ──────────────────────────────
        # Any detection index NOT in matched_dets → new object → new track.
        # set(range(n_dets)) creates {0, 1, 2, ..., n_dets-1}
        # - matched_dets removes the ones already matched.
        unmatched_dets   = list(set(range(n_dets))   - matched_dets)
        unmatched_tracks = list(set(range(n_tracks)) - matched_tracks)

        return matches, unmatched_dets, unmatched_tracks


    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Main function — call once per frame with YOLO's detections.

        Arguments:
            detections: list of detection dictionaries from obstacle_detection.py
                Each dict must have:
                  {
                    "label":       str,   e.g. "person"
                    "bbox":        (x1, y1, x2, y2),
                    "confidence":  float  0.0 to 1.0,
                    "distance_mm": float  depth in mm
                  }

        Returns:
            List of track dictionaries for all CONFIRMED tracks.
            Each dict has the format produced by Track.to_dict().
        """

        # ── Step 1: Predict ───────────────────────────────────────────────────
        # Advance every existing track's Kalman filter by one frame.
        # This updates each track's predicted position BEFORE we look
        # at the new detections. Prediction always happens first.
        for track in self.tracks:
            track.predict()

        # ── Step 2: Match ─────────────────────────────────────────────────────
        # Match new detections to existing tracks using IoU.
        # Returns three lists of indices.
        matches, unmatched_dets, unmatched_tracks = \
            self._match_detections_to_tracks(detections)

        # ── Step 3: Update matched tracks ─────────────────────────────────────
        # For each matched (detection, track) pair, update the track with
        # the new measurement from YOLO.
        for det_idx, track_idx in matches:
            det   = detections[det_idx]
            track = self.tracks[track_idx]
            track.update(
                bbox        = det["bbox"],
                confidence  = det["confidence"],
                distance_mm = det["distance_mm"]
            )

        # ── Step 4: Create new tracks for unmatched detections ─────────────────
        # Any detection that didn't match an existing track is a new object.
        # Create a new Track for it.
        for det_idx in unmatched_dets:
            det      = detections[det_idx]
            new_id   = self._generate_id(det["label"])
            new_track = Track(
                track_id    = new_id,
                bbox        = det["bbox"],
                label       = det["label"],
                confidence  = det["confidence"],
                distance_mm = det["distance_mm"],
                frame_width = self.frame_width
            )
            self.tracks.append(new_track)

        # ── Step 5: Mark unmatched tracks as lost ─────────────────────────────
        # Any existing track that got no matching detection this frame
        # is marked as lost. Its Kalman filter keeps predicting its position.
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_lost()

        # ── Step 6: Remove deleted tracks ─────────────────────────────────────
        # Filter out tracks that have been marked DELETED.
        # List comprehension: keep only tracks that are NOT deleted.
        before = len(self.tracks)
        self.tracks = [
            t for t in self.tracks
            if t.state != TrackState.DELETED
        ]
        after = len(self.tracks)

        # Log how many tracks were removed.
        if before != after:
            logger.debug(f"Removed {before - after} deleted tracks. "
                         f"Active tracks: {after}")

        # ── Step 7: Return confirmed tracks ───────────────────────────────────
        # Only return CONFIRMED tracks to the rest of ECHORA.
        # DETECTED tracks (new, unconfirmed) are silently excluded.
        # LOST tracks are excluded — their predictions aren't reliable enough.
        confirmed = [
            t.to_dict()
            for t in self.tracks
            if t.is_confirmed()
        ]

        return confirmed


    def get_confirmed_tracks(self) -> List[Dict]:
        """
        Returns all currently CONFIRMED tracks as a list of dictionaries.

        Use this when you need the current track list WITHOUT running
        an update (e.g. for drawing the debug overlay between YOLO runs).
        """
        return [
            t.to_dict()
            for t in self.tracks
            if t.is_confirmed()
        ]


    def get_track_by_id(self, track_id: str) -> Optional[Dict]:
        """
        Returns a specific track by its ID string.

        Returns None if no track with that ID exists.

        Used by control_unit.py when it needs to check the latest state
        of a specific object (e.g. "what is person_001's distance now?").
        """

        # Loop through all tracks and check for a matching ID.
        for track in self.tracks:
            if track.track_id == track_id:
                return track.to_dict()

        # No track found with that ID.
        return None


    def get_most_urgent(self) -> Optional[Dict]:
        """
        Returns the single most urgent confirmed track — the one that
        is closest to the user and in the highest urgency state.

        Used by control_unit.py when it can only announce one thing
        at a time and needs to pick the most important one.

        Priority order: DANGER > WARNING > SAFE
        Within same urgency: closest object wins.
        """

        confirmed = self.get_confirmed_tracks()

        if not confirmed:
            return None

        # Sort by two criteria:
        # 1. Urgency priority: DANGER first, then WARNING, then SAFE.
        # 2. Distance: closer objects first within same urgency level.
        #
        # We assign a numeric priority to each urgency string.
        urgency_priority = {"DANGER": 0, "WARNING": 1, "SAFE": 2, "UNKNOWN": 3}

        # sorted() returns a new sorted list.
        # key= defines what to sort by.
        # lambda creates a small inline function.
        # We sort by (urgency_number, distance_mm) as a tuple.
        # Python sorts tuples element by element:
        #   first by urgency_number (DANGER=0 comes first),
        #   then by distance_mm (smaller = closer = more urgent).
        sorted_tracks = sorted(
            confirmed,
            key=lambda t: (
                urgency_priority.get(t["urgency"], 3),
                t["distance_mm"]
            )
        )

        # The first item after sorting is the most urgent.
        return sorted_tracks[0]


    def reset(self):
        """
        Clears all tracks completely.

        Call this when the system switches modes — for example when
        going from NAVIGATION to OCR mode, navigation tracks become
        irrelevant and should not persist.

        The ID counter is NOT reset — we never reuse IDs.
        """

        count = len(self.tracks)
        self.tracks.clear()
        logger.info(f"KalmanTracker reset. Cleared {count} tracks.")


    def get_stats(self) -> Dict:
        """
        Returns diagnostic statistics about the current tracker state.
        Useful for logging and debugging performance.
        """

        # Count tracks in each state using list comprehensions.
        n_detected  = sum(1 for t in self.tracks if t.state == TrackState.DETECTED)
        n_confirmed = sum(1 for t in self.tracks if t.state == TrackState.CONFIRMED)
        n_lost      = sum(1 for t in self.tracks if t.state == TrackState.LOST)

        return {
            "total_tracks":     len(self.tracks),
            "detected":         n_detected,
            "confirmed":        n_confirmed,
            "lost":             n_lost,
            "total_ids_issued": self._next_id,
        }


# =============================================================================
# SELF-TEST
# =============================================================================
# Run directly to verify the tracker works WITHOUT needing a camera:
#   python kalman_tracker.py
#
# We simulate YOLO detections manually so we can test tracking logic
# without needing any hardware.

if __name__ == "__main__":

    print("=== ECHORA kalman_tracker.py self-test ===\n")

    # Create a tracker for a 1280px wide frame.
    tracker = KalmanTracker(frame_width=1280)

    # ── Simulate a sequence of YOLO detections ────────────────────────────────
    # Each frame is a list of detection dictionaries.
    # We simulate a person moving from left to right,
    # and a chair that stays still.
    # Frame 3 has no person detection (occlusion) to test LOST state.

    simulated_frames = [
        # Frame 1: person on the left, chair in the middle
        [
            {"label": "person", "bbox": (100, 100, 200, 300), "confidence": 0.91, "distance_mm": 1500},
            {"label": "chair",  "bbox": (500, 200, 650, 400), "confidence": 0.85, "distance_mm": 2200},
        ],
        # Frame 2: person moved slightly right, chair same
        [
            {"label": "person", "bbox": (115, 100, 215, 300), "confidence": 0.89, "distance_mm": 1480},
            {"label": "chair",  "bbox": (502, 200, 652, 400), "confidence": 0.86, "distance_mm": 2200},
        ],
        # Frame 3: person temporarily not detected (e.g. occluded)
        [
            {"label": "chair",  "bbox": (501, 200, 651, 400), "confidence": 0.84, "distance_mm": 2200},
        ],
        # Frame 4: person reappears, now closer and more to the right
        [
            {"label": "person", "bbox": (140, 100, 240, 300), "confidence": 0.92, "distance_mm": 750},
            {"label": "chair",  "bbox": (500, 200, 650, 400), "confidence": 0.87, "distance_mm": 2200},
        ],
        # Frame 5: same
        [
            {"label": "person", "bbox": (155, 100, 255, 300), "confidence": 0.90, "distance_mm": 700},
            {"label": "chair",  "bbox": (500, 200, 650, 400), "confidence": 0.85, "distance_mm": 2200},
        ],
    ]

    # ── Run the tracker through each simulated frame ──────────────────────────
    for frame_num, detections in enumerate(simulated_frames, start=1):

        print(f"--- Frame {frame_num} ---")
        print(f"  YOLO detections: {len(detections)}")

        # Update the tracker with this frame's detections.
        confirmed_tracks = tracker.update(detections)

        # Print confirmed tracks.
        if confirmed_tracks:
            for t in confirmed_tracks:
                print(f"  CONFIRMED: [{t['id']}] {t['label']:8s} | "
                      f"dist={t['distance_mm']:5.0f}mm | "
                      f"angle={t['angle_deg']:+.1f}deg | "
                      f"urgency={t['urgency']:7s} | "
                      f"hits={t['hits']} missed={t['missed']}")
        else:
            print("  No confirmed tracks yet.")

        # Print tracker stats.
        stats = tracker.get_stats()
        print(f"  Stats: {stats}\n")

    # ── Test get_most_urgent ──────────────────────────────────────────────────
    print("--- Most urgent track ---")
    most_urgent = tracker.get_most_urgent()
    if most_urgent:
        print(f"  → {most_urgent['label']} at {most_urgent['distance_mm']}mm "
              f"[{most_urgent['urgency']}]")
    else:
        print("  → No confirmed tracks.")

    # ── Test reset ────────────────────────────────────────────────────────────
    print("\n--- Testing reset ---")
    tracker.reset()
    stats = tracker.get_stats()
    print(f"  After reset: {stats}")

    print("\n=== All tests complete ===")