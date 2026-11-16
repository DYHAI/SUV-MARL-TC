"""Nearest-active-TC diagnostics for surface fleet-center distances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRACK_ROOT = ROOT / "data" / "selected_typhoons"
STEP_MIN = 10
D2KM = 111.0


@dataclass(frozen=True)
class Track:
    sid: str
    tag: str
    epoch_s: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


@dataclass(frozen=True)
class CandidateOnTargetTimeline:
    sid: str
    tag: str
    lat: np.ndarray
    lon: np.ndarray
    active: np.ndarray


class NearestTCCatalog:
    """Resolve the nearest active IBTrACS TC at a given simulation step."""

    def __init__(self, tracks_dir: Path = TRACK_ROOT, active_tolerance_min: float = 90.0):
        self.active_tolerance_s = float(active_tolerance_min) * 60.0
        by_sid: dict[str, Track] = {}
        for path in tracks_dir.glob("*/track.csv"):
            try:
                frame = pd.read_csv(
                    path,
                    usecols=["SID", "ISO_TIME", "LAT", "LON"],
                    parse_dates=["ISO_TIME"],
                ).dropna(subset=["LAT", "LON"])
            except (FileNotFoundError, ValueError, pd.errors.ParserError):
                continue
            if frame.empty:
                continue
            sid = str(frame["SID"].iloc[0])
            if sid in by_sid and len(frame) <= len(by_sid[sid].epoch_s):
                continue
            frame = frame.sort_values("ISO_TIME")
            by_sid[sid] = Track(
                sid=sid,
                tag=path.parent.name,
                epoch_s=frame["ISO_TIME"].astype("int64").to_numpy() / 1e9,
                lat=frame["LAT"].to_numpy(dtype=np.float64),
                lon=frame["LON"].to_numpy(dtype=np.float64),
            )
        self.tracks = tuple(by_sid.values())
        self.tag_to_sid = {track.tag: track.sid for track in self.tracks}
        self._timeline_cache: dict[str, list[CandidateOnTargetTimeline]] = {}

    def _target_timeline(self, data: dict) -> np.ndarray:
        start = pd.Timestamp(np.asarray(data["times"])[0])
        return pd.date_range(
            start,
            periods=int(data["T"]),
            freq=f"{STEP_MIN}min",
        ).asi8.astype(np.float64) / 1e9

    def candidates_for(self, data: dict) -> list[CandidateOnTargetTimeline]:
        storm = str(data["storm"])
        if storm in self._timeline_cache:
            return self._timeline_cache[storm]

        target_sid = self.tag_to_sid.get(storm)
        target_epoch = self._target_timeline(data)
        candidates: list[CandidateOnTargetTimeline] = []
        for track in self.tracks:
            if track.sid == target_sid:
                continue
            if (
                target_epoch[-1] < track.epoch_s[0] - self.active_tolerance_s
                or target_epoch[0] > track.epoch_s[-1] + self.active_tolerance_s
            ):
                continue

            pos = np.searchsorted(track.epoch_s, target_epoch)
            left = np.clip(pos - 1, 0, len(track.epoch_s) - 1)
            right = np.clip(pos, 0, len(track.epoch_s) - 1)
            nearest = np.where(
                np.abs(track.epoch_s[left] - target_epoch)
                <= np.abs(track.epoch_s[right] - target_epoch),
                left,
                right,
            )
            active = (
                np.abs(track.epoch_s[nearest] - target_epoch)
                <= self.active_tolerance_s
            )
            if not active.any():
                continue
            candidates.append(
                CandidateOnTargetTimeline(
                    sid=track.sid,
                    tag=track.tag,
                    lat=np.interp(target_epoch, track.epoch_s, track.lat),
                    lon=np.interp(target_epoch, track.epoch_s, track.lon),
                    active=active,
                )
            )
        self._timeline_cache[storm] = candidates
        return candidates

    def nearest(
        self,
        data: dict,
        step: int,
        fleet_lon: float,
        fleet_lat: float,
    ) -> tuple[float, str, int]:
        """Return distance, nearest TC tag, and number of active TCs."""
        idx = int(step)
        best_distance = float("inf")
        best_tag = str(data["storm"])
        active_count = 1

        def distance(lat: float, lon: float) -> float:
            dx = (fleet_lon - lon) * D2KM * np.cos(
                np.deg2rad((fleet_lat + lat) / 2.0)
            )
            dy = (fleet_lat - lat) * D2KM
            return float(np.hypot(dx, dy))

        target_distance = distance(
            float(data["track_lat"][idx]),
            float(data["track_lon"][idx]),
        )
        best_distance = target_distance
        for candidate in self.candidates_for(data):
            if not bool(candidate.active[idx]):
                continue
            active_count += 1
            candidate_distance = distance(
                float(candidate.lat[idx]),
                float(candidate.lon[idx]),
            )
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_tag = candidate.tag
        return best_distance, best_tag, active_count
