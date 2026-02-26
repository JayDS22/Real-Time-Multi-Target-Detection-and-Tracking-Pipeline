"""
Prototype detection pipeline for multi-target tracking in image sequences.

Workflow:
  1. Background subtraction (running average)
  2. Morphological cleanup
  3. Connected-component labeling
  4. Centroid-based tracking (greedy nearest-neighbor)

Written for clarity, not speed — the C++ version (src/) is the
optimized translation of this.
"""

import numpy as np
import cv2
import time
from typing import List, Tuple, Dict, Optional


class Detection:
    def __init__(self, bbox: Tuple[int, int, int, int], centroid: Tuple[float, float],
                 area: float, confidence: float):
        self.bbox = bbox
        self.centroid = centroid
        self.area = area
        self.confidence = confidence


class Track:
    def __init__(self, track_id: int, centroid: Tuple[float, float]):
        self.track_id = track_id
        self.centroid = centroid
        self.age = 0
        self.frames_since_seen = 0
        self.history: List[Tuple[float, float]] = [centroid]


class BackgroundModel:
    """Exponentially weighted running average."""

    def __init__(self, alpha: float = 0.02):
        self.alpha = alpha
        self.bg_model: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        if self.bg_model is None:
            self.bg_model = gray.copy()
            return np.zeros_like(gray, dtype=np.uint8)

        self.bg_model = self.alpha * gray + (1.0 - self.alpha) * self.bg_model
        diff = np.abs(gray - self.bg_model)
        _, mask = cv2.threshold(diff.astype(np.uint8), 25, 255, cv2.THRESH_BINARY)
        return mask


class MorphologyFilter:
    """Opening + closing to clean up the fg mask."""

    def __init__(self, kernel_size: int = 5, open_iter: int = 2, close_iter: int = 2):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (kernel_size, kernel_size))
        self.open_iter = open_iter
        self.close_iter = close_iter

    def apply(self, mask: np.ndarray) -> np.ndarray:
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel,
                                    iterations=self.open_iter)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel,
                                    iterations=self.close_iter)
        return cleaned


class DetectionExtractor:
    """Pull candidate detections out of a binary mask."""

    def __init__(self, min_area: int = 100, max_area: int = 50000):
        self.min_area = min_area
        self.max_area = max_area

    def extract(self, mask: np.ndarray) -> List[Detection]:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        detections = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area or area > self.max_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]

            aspect = w / max(h, 1)
            conf = min(1.0, area / 5000.0) * min(1.0, 1.0 / (abs(aspect - 1.0) + 0.5))

            detections.append(Detection(
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                area=float(area),
                confidence=conf
            ))
        return detections


class CentroidTracker:
    """
    Greedy nearest-neighbor tracker. Should be swapped for Hungarian
    assignment if we need to handle heavy occlusion.
    """

    def __init__(self, max_disappeared: int = 15, max_distance: float = 80.0):
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections: List[Detection]) -> Dict[int, Track]:
        if len(detections) == 0:
            to_remove = []
            for tid, track in self.tracks.items():
                track.frames_since_seen += 1
                if track.frames_since_seen > self.max_disappeared:
                    to_remove.append(tid)
            for tid in to_remove:
                del self.tracks[tid]
            return self.tracks

        det_centroids = np.array([d.centroid for d in detections])

        if len(self.tracks) == 0:
            for det in detections:
                self._register(det.centroid)
            return self.tracks

        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])

        # pairwise distances: (num_tracks, num_detections)
        dists = np.linalg.norm(
            track_centroids[:, np.newaxis, :] - det_centroids[np.newaxis, :, :],
            axis=2
        )

        assigned_tracks = set()
        assigned_dets = set()

        flat_indices = np.argsort(dists, axis=None)
        for idx in flat_indices:
            t_idx = idx // dists.shape[1]
            d_idx = idx % dists.shape[1]

            if t_idx in assigned_tracks or d_idx in assigned_dets:
                continue
            if dists[t_idx, d_idx] > self.max_distance:
                break

            tid = track_ids[t_idx]
            self.tracks[tid].centroid = tuple(det_centroids[d_idx])
            self.tracks[tid].frames_since_seen = 0
            self.tracks[tid].age += 1
            self.tracks[tid].history.append(tuple(det_centroids[d_idx]))

            assigned_tracks.add(t_idx)
            assigned_dets.add(d_idx)

        for t_idx, tid in enumerate(track_ids):
            if t_idx not in assigned_tracks:
                self.tracks[tid].frames_since_seen += 1
                if self.tracks[tid].frames_since_seen > self.max_disappeared:
                    del self.tracks[tid]

        for d_idx in range(len(detections)):
            if d_idx not in assigned_dets:
                self._register(detections[d_idx].centroid)

        return self.tracks

    def _register(self, centroid: Tuple[float, float]):
        self.tracks[self.next_id] = Track(self.next_id, centroid)
        self.next_id += 1


class DetectionPipeline:
    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.bg_model = BackgroundModel(alpha=cfg.get("bg_alpha", 0.02))
        self.morph_filter = MorphologyFilter(
            kernel_size=cfg.get("morph_kernel", 5),
            open_iter=cfg.get("morph_open_iter", 2),
            close_iter=cfg.get("morph_close_iter", 2)
        )
        self.extractor = DetectionExtractor(
            min_area=cfg.get("min_area", 100),
            max_area=cfg.get("max_area", 50000)
        )
        self.tracker = CentroidTracker(
            max_disappeared=cfg.get("max_disappeared", 15),
            max_distance=cfg.get("max_distance", 80.0)
        )

    def process_frame(self, frame):
        fg_mask = self.bg_model.update(frame)
        cleaned = self.morph_filter.apply(fg_mask)
        detections = self.extractor.extract(cleaned)
        tracks = self.tracker.update(detections)
        return detections, tracks, cleaned


def draw_results(frame, detections, tracks):
    vis = frame.copy()
    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, f"conf: {det.confidence:.2f}", (x, y - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    for tid, track in tracks.items():
        cx, cy = int(track.centroid[0]), int(track.centroid[1])
        cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(vis, f"ID:{tid}", (cx + 8, cy - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        pts = track.history[-30:]
        for i in range(1, len(pts)):
            p1 = (int(pts[i-1][0]), int(pts[i-1][1]))
            p2 = (int(pts[i][0]), int(pts[i][1]))
            cv2.line(vis, p1, p2, (255, 128, 0), 1)

    return vis


def benchmark_pipeline(width=640, height=480, num_frames=200):
    """Run on synthetic data to get a throughput number."""
    pipeline = DetectionPipeline()
    rng = np.random.default_rng(42)

    n_targets = 5
    positions = rng.uniform(50, min(width, height) - 50, size=(n_targets, 2))
    velocities = rng.uniform(-3, 3, size=(n_targets, 2))
    radii = rng.integers(10, 30, size=n_targets)
    base_bg = rng.integers(40, 80, size=(height, width, 3), dtype=np.uint8)

    elapsed_times = []
    for frame_idx in range(num_frames):
        frame = base_bg.copy()
        for i in range(n_targets):
            positions[i] += velocities[i]
            for axis in range(2):
                limit = width if axis == 0 else height
                if positions[i][axis] < radii[i] or positions[i][axis] > limit - radii[i]:
                    velocities[i][axis] *= -1
                    positions[i][axis] = np.clip(positions[i][axis], radii[i], limit - radii[i])
            cx, cy = int(positions[i][0]), int(positions[i][1])
            cv2.circle(frame, (cx, cy), int(radii[i]), (200, 200, 200), -1)

        noise = rng.normal(0, 10, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        t0 = time.perf_counter()
        dets, tracks, mask = pipeline.process_frame(frame)
        elapsed_times.append(time.perf_counter() - t0)

    elapsed_times = elapsed_times[10:]  # drop warmup
    avg_ms = np.mean(elapsed_times) * 1000
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    print(f"Python prototype benchmark ({width}x{height}, {num_frames} frames):")
    print(f"  Average frame time: {avg_ms:.2f} ms")
    print(f"  Throughput: {fps:.1f} FPS")
    print(f"  Detections in last frame: {len(dets)}")
    print(f"  Active tracks in last frame: {len(tracks)}")


if __name__ == "__main__":
    benchmark_pipeline()
