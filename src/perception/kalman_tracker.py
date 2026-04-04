import numpy as np
import cv2
import time
from typing import List, Dict, Optional, Tuple

from src.core.config import (
    KALMAN_PROCESS_NOISE,
    KALMAN_MEASUREMENT_NOISE,
    KALMAN_MAX_MISSED_FRAMES,
)
from src.core.utils import logger, bbox_center, angle_from_x, classify_urgency, get_timestamp_ms

class TrackState:
    DETECTED  = "DETECTED"
    CONFIRMED = "CONFIRMED"
    LOST      = "LOST"
    DELETED   = "DELETED"

MIN_HITS_TO_CONFIRM = 2
IOU_THRESHOLD = 0.3
MAX_CENTER_DISTANCE_PX = 200

class Track:
    """Represents one tracked object with its own Kalman filter."""

    def __init__(self, track_id: str, bbox: Tuple[int, int, int, int], label: str,
                 confidence: float, distance_mm: float, frame_width: int):
        self.track_id    = track_id
        self.label       = label
        self.confidence  = confidence
        self.state       = TrackState.DETECTED
        self.bbox        = bbox
        self.distance_mm = distance_mm
        self.frame_width = frame_width

        self.hits        = 1
        self.missed      = 0
        self.age         = 1
        self.created_at  = time.time()
        self.last_seen_ms = get_timestamp_ms()

        self._kalman     = self._init_kalman(bbox)
        logger.debug(f"Track created: {track_id} ({label}) at bbox={bbox}")

    def _init_kalman(self, bbox: Tuple[int, int, int, int]) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0], [0, 1, 0, 1],
            [0, 0, 1, 0], [0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0],
        ], dtype=np.float32)

        kf.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE
        kf.errorCovPost = np.array([
            [1, 0, 0,  0], [0, 1, 0,  0],
            [0, 0, 10, 0], [0, 0, 0, 10],
        ], dtype=np.float32)

        cx, cy = bbox_center(*bbox)
        kf.statePost = np.array(
            [[float(cx)], [float(cy)], [0.0], [0.0]],
            dtype=np.float32
        )
        return kf

    def predict(self):
        predicted = self._kalman.predict()
        pred_cx, pred_cy = float(predicted[0][0]), float(predicted[1][0])

        x1, y1, x2, y2 = self.bbox
        w, h = x2 - x1, y2 - y1

        self.bbox = (
            int(pred_cx - w / 2), int(pred_cy - h / 2),
            int(pred_cx + w / 2), int(pred_cy + h / 2)
        )
        self.age += 1

    def update(self, bbox: Tuple[int, int, int, int], confidence: float, distance_mm: float):
        cx, cy = bbox_center(*bbox)
        measurement = np.array([[float(cx)], [float(cy)]], dtype=np.float32)
        self._kalman.correct(measurement)

        self.bbox        = bbox
        self.confidence  = confidence
        self.distance_mm = distance_mm
        self.missed      = 0
        self.hits       += 1
        self.last_seen_ms = get_timestamp_ms()

        if self.state == TrackState.DETECTED and self.hits >= MIN_HITS_TO_CONFIRM:
            self.state = TrackState.CONFIRMED
            logger.debug(f"Track confirmed: {self.track_id} ({self.label})")
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED
            logger.debug(f"Track recovered: {self.track_id} ({self.label})")

    def mark_lost(self):
        self.missed += 1
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST
            logger.debug(f"Track lost: {self.track_id} ({self.label}), missed={self.missed}")

        if self.missed > KALMAN_MAX_MISSED_FRAMES:
            self.state = TrackState.DELETED
            logger.debug(f"Track deleted: {self.track_id} ({self.label})")

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def get_predicted_center(self) -> Tuple[float, float]:
        cx = float(self._kalman.statePost[0][0])
        cy = float(self._kalman.statePost[1][0])
        return cx, cy

    def to_dict(self) -> Dict:
        cx, cy = bbox_center(*self.bbox)
        angle = angle_from_x(cx, self.frame_width)
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

class KalmanTracker:
    def __init__(self, frame_width: int):
        self.tracks: List[Track] = []
        self._next_id: int = 0
        self.frame_width = frame_width
        logger.info(f"KalmanTracker initialised (frame_width={frame_width}px).")

    def _generate_id(self, label: str) -> str:
        self._next_id += 1
        return f"{label}_{self._next_id:03d}"

    def _iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box1
        bx1, by1, bx2, by2 = box2

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area1 = (ax2 - ax1) * (ay2 - ay1)
        area2 = (bx2 - bx1) * (by2 - by1)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)

    def _center_distance(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        cx1, cy1 = bbox_center(*box1)
        cx2, cy2 = bbox_center(*box2)
        return ((cx2 - cx1)**2 + (cy2 - cy1)**2) ** 0.5

    def _match_detections_to_tracks(self, detections: List[Dict]) -> Tuple[List, List, List]:
        n_dets = len(detections)
        n_tracks = len(self.tracks)

        if n_tracks == 0:
            return [], list(range(n_dets)), []
        if n_dets == 0:
            return [], [], list(range(n_tracks))

        iou_matrix = np.zeros((n_dets, n_tracks), dtype=np.float32)

        for d_idx, det in enumerate(detections):
            for t_idx, track in enumerate(self.tracks):
                if self._center_distance(det["bbox"], track.bbox) > MAX_CENTER_DISTANCE_PX:
                    continue
                if det["label"] != track.label:
                    continue
                iou_matrix[d_idx][t_idx] = self._iou(det["bbox"], track.bbox)

        matches = []
        matched_dets = set()
        matched_tracks = set()

        for _ in range(min(n_dets, n_tracks)):
            max_iou = np.max(iou_matrix)
            if max_iou < IOU_THRESHOLD:
                break

            d_idx, t_idx = map(int, np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape))
            matches.append((d_idx, t_idx))
            matched_dets.add(d_idx)
            matched_tracks.add(t_idx)

            iou_matrix[d_idx, :] = 0.0
            iou_matrix[:, t_idx] = 0.0

        unmatched_dets = list(set(range(n_dets)) - matched_dets)
        unmatched_tracks = list(set(range(n_tracks)) - matched_tracks)
        return matches, unmatched_dets, unmatched_tracks

    def update(self, detections: List[Dict]) -> List[Dict]:
        for track in self.tracks:
            track.predict()

        matches, unmatched_dets, unmatched_tracks = self._match_detections_to_tracks(detections)

        for det_idx, track_idx in matches:
            det = detections[det_idx]
            self.tracks[track_idx].update(det["bbox"], det["confidence"], det["distance_mm"])

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_track = Track(
                track_id=self._generate_id(det["label"]),
                bbox=det["bbox"],
                label=det["label"],
                confidence=det["confidence"],
                distance_mm=det["distance_mm"],
                frame_width=self.frame_width
            )
            self.tracks.append(new_track)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_lost()

        before = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]
        if before != len(self.tracks):
            logger.debug(f"Removed {before - len(self.tracks)} deleted tracks. Active: {len(self.tracks)}")

        return [t.to_dict() for t in self.tracks if t.is_confirmed()]

    def get_confirmed_tracks(self) -> List[Dict]:
        return [t.to_dict() for t in self.tracks if t.is_confirmed()]

    def get_track_by_id(self, track_id: str) -> Optional[Dict]:
        for track in self.tracks:
            if track.track_id == track_id:
                return track.to_dict()
        return None

    def get_most_urgent(self) -> Optional[Dict]:
        confirmed = self.get_confirmed_tracks()
        if not confirmed:
            return None

        urgency_priority = {"DANGER": 0, "WARNING": 1, "SAFE": 2, "UNKNOWN": 3}
        sorted_tracks = sorted(confirmed, key=lambda t: (urgency_priority.get(t["urgency"], 3), t["distance_mm"]))
        return sorted_tracks[0]

    def reset(self):
        count = len(self.tracks)
        self.tracks.clear()
        logger.info(f"KalmanTracker reset. Cleared {count} tracks.")

    def get_stats(self) -> Dict:
        return {
            "total_tracks":     len(self.tracks),
            "detected":         sum(1 for t in self.tracks if t.state == TrackState.DETECTED),
            "confirmed":        sum(1 for t in self.tracks if t.state == TrackState.CONFIRMED),
            "lost":             sum(1 for t in self.tracks if t.state == TrackState.LOST),
            "total_ids_issued": self._next_id,
        }