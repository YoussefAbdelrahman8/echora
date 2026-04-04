import depthai as dai
import numpy as np
import cv2
import time
import threading
import datetime
from typing import Optional, Dict, Any, List

from src.core.config import settings
from src.core.utils import logger, get_timestamp_ms

class EchoraCamera:
    """Manages the OAK-D hardware using depthai v3 API."""

    def __init__(self):
        self.pipeline: Optional[dai.Pipeline] = None
        self.sync_queue = None
        self.imu_queue  = None
        self.kalman: Optional[cv2.KalmanFilter] = None
        self.missed_frames: Dict[str, int] = {}
        self._latest_imu: Dict[str, Any] = {
            "accel": {"x": 0.0, "y": 0.0, "z": 0.0},
            "gyro":  {"x": 0.0, "y": 0.0, "z": 0.0},
            "timestamp_ms": 0.0
        }
        self._imu_lock = threading.Lock()
        self._running = False
        logger.info("EchoraCamera created. Call init_pipeline() to start.")

    def init_pipeline(self):
        logger.info("Initialising depthai v3 pipeline...")
        self.pipeline = dai.Pipeline()

        cam_rgb   = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam_left  = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        cam_right = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        rgb_out = cam_rgb.requestOutput((settings.CAMERA_RGB_WIDTH, settings.CAMERA_RGB_HEIGHT), dai.ImgFrame.Type.BGR888p)
        left_out = cam_left.requestOutput((settings.CAMERA_DEPTH_WIDTH, settings.CAMERA_DEPTH_HEIGHT), dai.ImgFrame.Type.GRAY8)
        right_out = cam_right.requestOutput((settings.CAMERA_DEPTH_WIDTH, settings.CAMERA_DEPTH_HEIGHT), dai.ImgFrame.Type.GRAY8)

        stereo = self.pipeline.create(dai.node.StereoDepth)
        stereo.setRectification(True)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(settings.CAMERA_DEPTH_WIDTH, settings.CAMERA_DEPTH_HEIGHT)
        left_out.link(stereo.left)
        right_out.link(stereo.right)

        sync = self.pipeline.create(dai.node.Sync)
        sync.setRunOnHost(True)
        sync.setSyncThreshold(datetime.timedelta(milliseconds=200))

        rgb_out.link(sync.inputs["rgb"])
        stereo.depth.link(sync.inputs["depth"])

        self.sync_queue = sync.out.createOutputQueue(maxSize=2, blocking=False)

        imu_node = self.pipeline.create(dai.node.IMU)
        imu_node.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 480)
        imu_node.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)
        imu_node.setBatchReportThreshold(1)
        imu_node.setMaxBatchReports(10)

        self.imu_queue = imu_node.out.createOutputQueue(maxSize=50, blocking=False)

        logger.info("Starting OAK-D device...")
        self.pipeline.start()
        logger.info("Device started successfully.")

        self.init_kalman()

        self._running = True
        self._imu_thread = threading.Thread(target=self._imu_reader_thread, daemon=True)
        self._imu_thread.start()
        logger.info("Pipeline fully initialised. Ready to stream.")

    def init_kalman(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0], [0, 1, 0, 1],
            [0, 0, 1, 0], [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0],
        ], dtype=np.float32)

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * settings.KALMAN_PROCESS_NOISE
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * settings.KALMAN_MEASUREMENT_NOISE
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        logger.info("Kalman filter initialised.")

    def kalman_update(self, x: float, y: float) -> tuple:
        self.kalman.predict()
        measurement = np.array([[x], [y]], dtype=np.float32)
        corrected = self.kalman.correct(measurement)
        return float(corrected[0][0]), float(corrected[1][0])

    def kalman_predict(self) -> tuple:
        predicted = self.kalman.predict()
        return float(predicted[0][0]), float(predicted[1][0])

    def _imu_reader_thread(self):
        logger.info("IMU reader thread started.")
        while self._running:
            try:
                imu_data = self.imu_queue.tryGet()
                if imu_data is None:
                    time.sleep(0.002)
                    continue

                for packet in imu_data.packets:
                    accel = packet.acceleroMeter
                    gyro  = packet.gyroscope
                    ts_ms = accel.getTimestamp().total_seconds() * 1000

                    new_imu = {
                        "accel": {"x": float(accel.x), "y": float(accel.y), "z": float(accel.z)},
                        "gyro":  {"x": float(gyro.x), "y": float(gyro.y), "z": float(gyro.z)},
                        "timestamp_ms": ts_ms
                    }

                    with self._imu_lock:
                        self._latest_imu = new_imu

            except Exception as e:
                logger.warning(f"IMU read error: {e}")
                time.sleep(0.01)

        logger.info("IMU reader thread stopped.")

    def get_synced_bundle(self) -> Optional[Dict]:
        if not self.pipeline.isRunning():
            logger.error("Pipeline is not running.")
            return None

        try:
            msg_group = self.sync_queue.tryGet()
            if msg_group is None:
                return None

            rgb = msg_group["rgb"].getCvFrame()
            depth = msg_group["depth"].getFrame()
            depth = np.clip(depth, 0, settings.DEPTH_MAX_MM).astype(np.uint16)

            with self._imu_lock:
                imu = self._latest_imu.copy()

            return {
                "rgb": rgb,
                "depth": depth,
                "imu": imu,
                "timestamp_ms": get_timestamp_ms()
            }

        except Exception as e:
            logger.error(f"Error reading bundle: {e}")
            return None

    def get_imu_data(self) -> Dict:
        with self._imu_lock:
            return self._latest_imu.copy()

    def update_missed_frames(self, active_ids: list) -> list:
        for obj_id in list(self.missed_frames.keys()):
            self.missed_frames[obj_id] += 1

        for obj_id in active_ids:
            self.missed_frames[obj_id] = 0

        to_drop = [
            obj_id for obj_id, count in self.missed_frames.items()
            if count > settings.KALMAN_MAX_MISSED_FRAMES
        ]

        for obj_id in to_drop:
            del self.missed_frames[obj_id]
            logger.debug(f"Dropped {obj_id} — missed {settings.KALMAN_MAX_MISSED_FRAMES} frames.")

        return to_drop

    def release(self):
        logger.info("Releasing camera resources...")
        self._running = False
        if hasattr(self, '_imu_thread') and self._imu_thread.is_alive():
            self._imu_thread.join(timeout=2.0)

        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as e:
                logger.warning(f"Pipeline stop warning: {e}")
            self.pipeline = None
        logger.info("Camera released cleanly.")