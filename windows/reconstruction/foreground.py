"""Deterministic green-background calibration and foreground masks for Object Scan Gate A."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class ForegroundConfig:
    border_fraction: float = 0.08
    calibration_samples_per_frame: int = 8192
    green_hue_min_degrees: float = 70.0
    green_hue_max_degrees: float = 170.0
    default_hue_tolerance_degrees: float = 35.0
    saturation_minimum: float = 0.20
    value_minimum: float = 0.10
    morphology_kernel: int = 3
    minimum_component_pixels: int = 64
    fill_small_holes: bool = True

    def validate(self) -> None:
        if not 0 < self.border_fraction <= 0.25:
            raise ValueError("border_fraction must be in (0, 0.25]")
        if self.calibration_samples_per_frame < 1:
            raise ValueError("calibration_samples_per_frame must be positive")
        if not 0 < self.default_hue_tolerance_degrees <= 90:
            raise ValueError("default_hue_tolerance_degrees must be in (0, 90]")
        if not 0 <= self.saturation_minimum <= 1 or not 0 <= self.value_minimum <= 1:
            raise ValueError("saturation_minimum and value_minimum must be in [0, 1]")
        if self.morphology_kernel < 1 or self.morphology_kernel % 2 == 0:
            raise ValueError("morphology_kernel must be an odd positive integer")
        if self.minimum_component_pixels < 1:
            raise ValueError("minimum_component_pixels must be positive")


@dataclass(frozen=True)
class GreenBackgroundModel:
    hue_center_degrees: float
    hue_tolerance_degrees: float
    saturation_floor: float
    value_floor: float
    sample_count: int
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _hsv(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB input must be uint8 HxWx3")
    return cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2HSV)


def _border_samples(rgb: np.ndarray, fraction: float, maximum: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    band = max(1, int(round(min(height, width) * fraction)))
    border = np.concatenate((rgb[:band], rgb[-band:], rgb[band:-band, :band], rgb[band:-band, -band:]), axis=None)
    samples = border.reshape(-1, 3)
    if len(samples) <= maximum:
        return samples
    indexes = np.linspace(0, len(samples) - 1, maximum, dtype=np.int64)
    return samples[indexes]


def calibrate_green_background(images: Iterable[np.ndarray], config: ForegroundConfig = ForegroundConfig()) -> GreenBackgroundModel:
    """Estimate the green distribution from central-object-safe border samples."""

    config.validate()
    candidates: list[np.ndarray] = []
    minimum_hue = config.green_hue_min_degrees / 2.0
    maximum_hue = config.green_hue_max_degrees / 2.0
    for image in images:
        samples = _border_samples(image, config.border_fraction, config.calibration_samples_per_frame)
        hsv = cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        likely_green = (
            (hsv[:, 0] >= minimum_hue)
            & (hsv[:, 0] <= maximum_hue)
            & (hsv[:, 1] >= config.saturation_minimum * 255.0)
            & (hsv[:, 2] >= config.value_minimum * 255.0)
        )
        if np.any(likely_green):
            candidates.append(hsv[likely_green])
    if not candidates:
        return GreenBackgroundModel(
            hue_center_degrees=120.0,
            hue_tolerance_degrees=config.default_hue_tolerance_degrees,
            saturation_floor=config.saturation_minimum,
            value_floor=config.value_minimum,
            sample_count=0,
            source="default_no_green_border_samples",
        )
    values = np.concatenate(candidates, axis=0)
    hue_center = float(np.median(values[:, 0]))
    hue_distance = np.abs(values[:, 0] - hue_center)
    tolerance = float(np.clip(np.percentile(hue_distance, 90) * 2.0 + 6.0, 15.0, config.default_hue_tolerance_degrees))
    return GreenBackgroundModel(
        hue_center_degrees=hue_center * 2.0,
        hue_tolerance_degrees=tolerance * 2.0,
        saturation_floor=float(max(config.saturation_minimum, np.median(values[:, 1]) / 255.0 * 0.45)),
        value_floor=float(max(config.value_minimum, np.median(values[:, 2]) / 255.0 * 0.25)),
        sample_count=int(len(values)),
        source="border_green_samples",
    )


def foreground_mask(rgb: np.ndarray, model: GreenBackgroundModel, config: ForegroundConfig = ForegroundConfig()) -> np.ndarray:
    """Return a cleaned foreground mask; background is calibrated green, not simply high-G RGB."""

    config.validate()
    hsv = _hsv(rgb)
    hue_distance = np.abs(hsv[:, :, 0].astype(np.float32) * 2.0 - model.hue_center_degrees)
    background = (
        (hue_distance <= model.hue_tolerance_degrees)
        & (hsv[:, :, 1] >= model.saturation_floor * 255.0)
        & (hsv[:, :, 2] >= model.value_floor * 255.0)
    )
    mask = (~background).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morphology_kernel, config.morphology_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, component_count):
        if stats[label, cv2.CC_STAT_AREA] >= config.minimum_component_pixels:
            cleaned[labels == label] = 1
    if config.fill_small_holes:
        inverse = (1 - cleaned).astype(np.uint8)
        flooded = inverse.copy()
        cv2.floodFill(flooded, None, (0, 0), 2)
        holes = flooded == 1
        cleaned[holes] = 1
    return cleaned.astype(bool)


def project_rgb_mask_to_depth(rgb_mask: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    """Sample RGB mask at the same depth-pixel centers used for RGB color sampling."""

    if rgb_mask.ndim != 2 or not rgb_mask.size:
        raise ValueError("rgb_mask must be a non-empty 2D mask")
    depth_height, depth_width = depth_shape
    if depth_height < 1 or depth_width < 1:
        raise ValueError("depth_shape must be positive")
    rgb_height, rgb_width = rgb_mask.shape
    v, u = np.indices((depth_height, depth_width), dtype=np.float64)
    u_rgb = np.rint(((u + 0.5) * rgb_width / depth_width) - 0.5).astype(np.int64)
    v_rgb = np.rint(((v + 0.5) * rgb_height / depth_height) - 0.5).astype(np.int64)
    return rgb_mask[np.clip(v_rgb, 0, rgb_height - 1), np.clip(u_rgb, 0, rgb_width - 1)].astype(bool)
