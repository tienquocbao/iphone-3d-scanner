from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from foreground import ForegroundConfig, calibrate_green_background, foreground_mask, project_rgb_mask_to_depth


class ForegroundTests(unittest.TestCase):
    def test_green_background_is_removed_and_non_green_object_is_preserved(self):
        rgb = np.full((32, 32, 3), [20, 170, 35], dtype=np.uint8)
        rgb[10:22, 10:22] = [210, 35, 30]
        model = calibrate_green_background([rgb])
        mask = foreground_mask(rgb, model, ForegroundConfig(minimum_component_pixels=8))
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[16, 16])

    def test_moderate_green_illumination_variation_calibrates(self):
        first = np.full((24, 24, 3), [20, 150, 30], dtype=np.uint8)
        second = np.full((24, 24, 3), [30, 205, 45], dtype=np.uint8)
        second[8:16, 8:16] = [30, 40, 210]
        model = calibrate_green_background([first, second])
        mask = foreground_mask(second, model, ForegroundConfig(minimum_component_pixels=8))
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[12, 12])

    def test_morphology_removes_noise_and_fills_small_hole(self):
        rgb = np.full((40, 40, 3), [20, 180, 30], dtype=np.uint8)
        rgb[10:30, 10:30] = [220, 20, 20]
        rgb[20, 20] = [20, 180, 30]
        rgb[3, 3] = [220, 20, 20]
        model = calibrate_green_background([rgb])
        mask = foreground_mask(rgb, model, ForegroundConfig(morphology_kernel=3, minimum_component_pixels=20))
        self.assertFalse(mask[3, 3])
        self.assertTrue(mask[20, 20])

    def test_depth_mask_projection_uses_pixel_centers_at_edges_and_center(self):
        rgb_mask = np.zeros((8, 8), dtype=bool)
        # For 8 RGB pixels projected to 2 depth pixels, the specified
        # pixel-center mapping samples RGB indexes 2 and 6.
        rgb_mask[2, 2] = True
        rgb_mask[6, 6] = True
        projected = project_rgb_mask_to_depth(rgb_mask, (2, 2))
        np.testing.assert_array_equal(projected, [[True, False], [False, True]])
        center_mask = np.zeros((8, 8), dtype=bool)
        center_mask[4, 4] = True
        center = project_rgb_mask_to_depth(center_mask, (1, 1))
        self.assertTrue(center[0, 0])


if __name__ == "__main__":
    unittest.main()
