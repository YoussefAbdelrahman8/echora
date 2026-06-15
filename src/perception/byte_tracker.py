import time
from typing import Dict, List, Optional, Tuple

from src.core.config import settings
from src.core.utils import logger, get_timestamp_ms

VELOCITY_WINDOW     = 5   # frames kept for approach-speed estimation
MIN_HITS_TO_CONFIRM = 2   # consecutive hits before CONFIRMED


class TrackState:
    DETECTED  = "DETECTED"
    CONFIRMED = "CONFIRMED"
    LOST      = "LOST"


class ByteTracker:
    """
    Lifecycle manager that sits on top of YOLO's built-in ByteTrack.

    ByteTrack (inside the YOLO model) handles detection association and
    short-occlusion recovery using its two-stage matching strategy.
    This class adds:
      • DETECTED → CONFIRMED promotion (MIN_HITS_TO_CONFIRM frames)
      • Auto-deletion after KALMAN_MAX_MISSED_FRAMES consecutive misses
      • Per-track approach-speed estimation (mm/s, negative = closing in)
      • Last-known-state cache so callers can query between detection frames
    """

    def __init__(self, frame_width: int):
        self.frame_width = frame_width
        self._states: Dict[int, dict] = {}
        self._last_dets: Dict[int, Dict] = {}  # track_id → last full det dict

    # ── Public API ────────────────────────────────────────────────────

    def update(self, yolo_tracks: List[Dict]) -> List[Dict]:
        """
        Args:
            yolo_tracks: detection dicts from ObstacleDetector, each containing
                         track_id, label, bbox, confidence, distance_mm,
                         angle_deg, urgency.
        Returns:
            CONFIRMED tracks only, each dict extended with:
                state, hits, missed, age, approach_mm_s
        """
        seen_ids: set = set()
        now_ms: float = get_timestamp_ms()
        output: List[Dict] = []

        for det in yolo_tracks:
            tid = det["track_id"]
            seen_ids.add(tid)

            if tid not in self._states:
                self._states[tid] = {
                    "hits":       1,
                    "missed":     0,
                    "state":      TrackState.DETECTED,
                    "created_at": time.time(),
                    "dist_hist":  [(now_ms, det["distance_mm"])],
                }
            else:
                s = self._states[tid]
                s["hits"]   += 1
                s["missed"]  = 0
                s["dist_hist"].append((now_ms, det["distance_mm"]))
                if len(s["dist_hist"]) > VELOCITY_WINDOW:
                    s["dist_hist"].pop(0)
                if s["hits"] >= MIN_HITS_TO_CONFIRM:
                    s["state"] = TrackState.CONFIRMED

            s = self._states[tid]
            det["state"]         = s["state"]
            det["hits"]          = s["hits"]
            det["missed"]        = s["missed"]
            det["age"]           = s["hits"] + s["missed"]
            det["approach_mm_s"] = self._velocity(s["dist_hist"])

            self._last_dets[tid] = det

            if s["state"] == TrackState.CONFIRMED:
                output.append(det)

        # Age out unseen tracks
        for tid in list(self._states.keys()):
            if tid not in seen_ids:
                s = self._states[tid]
                s["missed"] += 1
                if s["missed"] > settings.KALMAN_MAX_MISSED_FRAMES:
                    self._states.pop(tid, None)
                    self._last_dets.pop(tid, None)
                    logger.debug(f"ByteTracker: track {tid} expired after {s['missed']} misses.")

        return output

    def get_confirmed_tracks(self) -> List[Dict]:
        """Returns the last cached full dict for each confirmed track."""
        return [
            self._last_dets[tid]
            for tid, s in self._states.items()
            if s["state"] == TrackState.CONFIRMED and tid in self._last_dets
        ]

    def reset(self):
        count = len(self._states)
        self._states.clear()
        self._last_dets.clear()
        logger.info(f"ByteTracker reset. Cleared {count} tracks.")

    def get_stats(self) -> Dict:
        states = [s["state"] for s in self._states.values()]
        return {
            "total_tracks":    len(self._states),
            "confirmed":       states.count(TrackState.CONFIRMED),
            "detected":        states.count(TrackState.DETECTED),
            "lost_recovering": sum(1 for s in self._states.values() if s["missed"] > 0),
        }

    # ── Private ───────────────────────────────────────────────────────

    @staticmethod
    def _velocity(dist_hist: List[Tuple[float, float]]) -> float:
        """
        Approach speed in mm/s computed over the stored distance history.
        Negative = object is getting closer.
        Positive = object is moving away.
        Returns 0.0 when there is not enough history.
        """
        if len(dist_hist) < 2:
            return 0.0
        t0, d0 = dist_hist[0]
        t1, d1 = dist_hist[-1]
        dt_sec = (t1 - t0) / 1000.0
        if dt_sec < 0.001:
            return 0.0
        return (d1 - d0) / dt_sec
