import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from dataclasses import dataclass
from typing import Tuple
import os


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KhmerZones:
    """Pixel row boundaries for each Khmer zone within a character image."""
    zone_a_rows: Tuple[int, int]   # (top, bottom) — above-base vowels
    zone_b_rows: Tuple[int, int]   # (top, bottom) — base consonant body
    zone_c_rows: Tuple[int, int]   # (top, bottom) — subscript / coeng

@dataclass
class PreprocessingResult:
    """Output of Stage 1."""
    binary:        np.ndarray   # binarized image (0=background, 1=foreground)
    zones:         KhmerZones   # detected zone boundaries
    zone_a_mask:   np.ndarray   # boolean mask for Zone A pixels
    zone_b_mask:   np.ndarray   # boolean mask for Zone B pixels
    zone_c_mask:   np.ndarray   # boolean mask for Zone C pixels
    debug_overlay: np.ndarray   # BGR image with zone boundaries drawn (for visualization)

# ---------------------------------------------------------------------------
# Step 1 — Grayscale conversion
# ---------------------------------------------------------------------------

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR or RGBA image to grayscale. Pass-through if already gray."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        # RGBA — blend alpha onto white background first
        alpha = image[:, :, 3] / 255.0
        rgb   = image[:, :, :3].astype(np.float32)
        white = np.ones_like(rgb) * 255
        blended = (alpha[..., None] * rgb + (1 - alpha[..., None]) * white).astype(np.uint8)
        return cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Step 2 — Sauvola adaptive binarization
# ---------------------------------------------------------------------------

def sauvola_binarize(gray: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
    """
    Sauvola thresholding — adapts to local mean and variance.
    Much better than Otsu for handwritten Khmer where ink pressure varies.

    Returns binary image: 1 = foreground (ink), 0 = background.
    """
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1

    thresh = threshold_sauvola(gray, window_size=window_size, k=k)
    binary = (gray < thresh).astype(np.uint8)  # dark pixels = ink = foreground
    return binary


# ---------------------------------------------------------------------------
# Step 3 — Vertical Zone Detection
# ---------------------------------------------------------------------------

def detect_khmer_zones(binary: np.ndarray) -> KhmerZones:
    """
    Detect the three Khmer vertical zones from a binarized character image.

    Strategy:
      - Compute the horizontal ink density profile (sum of foreground pixels per row)
      - Find the ink centroid row (center of mass of ink)
      - Detect valleys (low-ink rows) that separate zones
      - Assign zones: A = top portion, B = main body, C = bottom portion

    This is a visual/pixel approach — no Unicode metadata needed.
    Works for both printed fonts and handwritten Khmer.
    """
    
    h, w = binary.shape
    
    # Row-wise ink density (number of foreground pixels per row)
    row_density = binary.sum(axis=1).astype(np.float32)
    
    # Smooth the density profile to reduce noise
    kernel = np.ones(5) / 5
    smoothed = np.convolve(row_density, kernel, mode='same')

    # Find ink center of mass
    total_ink = smoothed.sum()
    if total_ink == 0:
        # Empty image — return equal thirds
        third = h // 3
        return KhmerZones(
            zone_a_rows=(0, third),
            zone_b_rows=(third, 2 * third),
            zone_c_rows=(2 * third, h)
        )

    center_of_mass = int(np.average(np.arange(h), weights=smoothed))

    # Find first and last rows with ink
    ink_rows = np.where(smoothed > smoothed.max() * 0.05)[0]
    first_ink = int(ink_rows[0])
    last_ink  = int(ink_rows[-1])

    # --- Detect zone boundary between A and B ---
    # Look for a valley in the top 40% of the ink region
    top_region_end = first_ink + int((center_of_mass - first_ink) * 0.6)
    top_slice = smoothed[first_ink:top_region_end]

    a_b_boundary = _find_valley(smoothed, first_ink, top_region_end)

    # --- Detect zone boundary between B and C ---
    # Look for a valley in the bottom 40% of the ink region
    bottom_region_start = center_of_mass + int((last_ink - center_of_mass) * 0.4)
    b_c_boundary = _find_valley(smoothed, bottom_region_start, last_ink)

    # Safety: ensure zones don't overlap and are sensible
    a_b_boundary = max(a_b_boundary, first_ink + 2)
    b_c_boundary = min(b_c_boundary, last_ink - 2)
    if a_b_boundary >= b_c_boundary:
        # No clear separation — use proportional split
        a_b_boundary = first_ink + (last_ink - first_ink) // 4
        b_c_boundary = last_ink - (last_ink - first_ink) // 4

    return KhmerZones(
        zone_a_rows=(0,             a_b_boundary),
        zone_b_rows=(a_b_boundary,  b_c_boundary),
        zone_c_rows=(b_c_boundary,  h)
    )


def _find_valley(profile: np.ndarray, start: int, end: int) -> int:
    """Return the row index of the minimum ink density in profile[start:end]."""
    if start >= end:
        return (start + end) // 2
    segment = profile[start:end]
    local_min = int(np.argmin(segment))
    return start + local_min


# ---------------------------------------------------------------------------
# Step 4 — Per-zone contrast normalization (on GRAYSCALE, before binarization)
# ---------------------------------------------------------------------------

def normalize_zones_on_gray(gray: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE independently to each horizontal third of the grayscale image.
    This is done BEFORE binarization so that thin diacritics (Zone A) and
    subscripts (Zone C) get their own local contrast boost without being
    crushed by the thick base-consonant strokes in Zone B.

    Returns a contrast-normalized grayscale image (same dtype as input).
    """
    h, w = gray.shape
    result = gray.copy()

    # Use a rough proportional split for pre-binarization normalization
    # (exact zone detection happens after binarization)
    boundaries = [0, h // 5, h * 4 // 5, h]   # A=top 20%, B=middle 60%, C=bottom 20%

    clahe_configs = [
        cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2)),  # Zone A — small tiles, high clip (diacritics)
        cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)),  # Zone B — standard
        cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2)),  # Zone C — small tiles, high clip (subscripts)
    ]

    for i, clahe in enumerate(clahe_configs):
        r0, r1 = boundaries[i], boundaries[i + 1]
        if r1 <= r0:
            continue
        zone_gray = gray[r0:r1, :]
        if zone_gray.size == 0:
            continue
        result[r0:r1, :] = clahe.apply(zone_gray)

    return result


def normalize_zones(binary: np.ndarray, zones: KhmerZones) -> np.ndarray:
    """
    Legacy function kept for compatibility.
    The correct approach is normalize_zones_on_gray() called before binarization.
    This version simply returns the binary unchanged.
    """
    return binary


# ---------------------------------------------------------------------------
# Step 5 — Morphological noise cleanup (conservative)
# ---------------------------------------------------------------------------

def clean_noise(binary: np.ndarray, zones: KhmerZones) -> np.ndarray:
    """
    Remove only truly isolated single-pixel specks (salt noise).
    Uses a minimal 1-connectivity check — only removes a pixel if ALL
    8 neighbors are also background. This is much safer than morphological
    opening which can destroy thin coeng connectors and hairline strokes.
    """
    result = binary.copy().astype(np.uint8)

    # Build neighbor count map using a 3x3 sum (excluding center)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0  # exclude center pixel
    neighbor_sum = cv2.filter2D(result, -1, kernel.astype(np.float32))

    # A foreground pixel with 0 neighbors is isolated noise — remove it
    isolated = (result == 1) & (neighbor_sum == 0)
    result[isolated] = 0

    return result


# ---------------------------------------------------------------------------
# Build zone masks
# ---------------------------------------------------------------------------

def build_zone_masks(binary: np.ndarray, zones: KhmerZones):
    """Return boolean masks for each zone."""
    h, w = binary.shape
    a_mask = np.zeros((h, w), dtype=bool)
    b_mask = np.zeros((h, w), dtype=bool)
    c_mask = np.zeros((h, w), dtype=bool)

    a_mask[zones.zone_a_rows[0]:zones.zone_a_rows[1], :] = True
    b_mask[zones.zone_b_rows[0]:zones.zone_b_rows[1], :] = True
    c_mask[zones.zone_c_rows[0]:zones.zone_c_rows[1], :] = True

    # Only mark pixels that are actually foreground
    a_mask &= binary.astype(bool)
    b_mask &= binary.astype(bool)
    c_mask &= binary.astype(bool)

    return a_mask, b_mask, c_mask


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------

def build_debug_overlay(binary: np.ndarray, zones: KhmerZones) -> np.ndarray:
    """Draw zone boundary lines on a BGR image for visualization."""
    # Convert binary to BGR
    overlay = cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    h, w = binary.shape
    colors = {
        'A': (0, 200, 255),   # yellow — Zone A (above vowels)
        'B': (0, 255, 100),   # green  — Zone B (base consonant)
        'C': (255, 100, 0),   # blue   — Zone C (subscripts)
    }

    def draw_zone(rows, color, label):
        r0, r1 = rows
        cv2.line(overlay, (0, r0), (w, r0), color, 1)
        cv2.line(overlay, (0, r1-1), (w, r1-1), color, 1)
        mid = (r0 + r1) // 2
        cv2.putText(overlay, f'Zone {label}', (3, max(mid, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    draw_zone(zones.zone_a_rows, colors['A'], 'A')
    draw_zone(zones.zone_b_rows, colors['B'], 'B')
    draw_zone(zones.zone_c_rows, colors['C'], 'C')

    return overlay


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess(image: np.ndarray,
               sauvola_window: int = 25,
               sauvola_k: float = 0.2) -> PreprocessingResult:
    """
    Full Stage 1 pipeline.

    Args:
        image:          Input image (BGR, RGBA, or grayscale numpy array)
        sauvola_window: Window size for Sauvola thresholding (default 25)
        sauvola_k:      Sensitivity parameter for Sauvola (default 0.2)

    Returns:
        PreprocessingResult with binary image, zone masks, and debug overlay
    """
    # 1. Grayscale
    gray = to_grayscale(image)

    # 2. Per-zone contrast normalization on the GRAY image BEFORE binarizing
    #    This is the key fix — normalize on grayscale, not on binary
    gray = normalize_zones_on_gray(gray)

    # 3. Sauvola binarization on the normalized grayscale
    binary = sauvola_binarize(gray, window_size=sauvola_window, k=sauvola_k)

    # 4. Zone detection from the clean binary
    zones = detect_khmer_zones(binary)

    # 5. Morphological noise cleanup (conservative — only remove isolated specks)
    binary = clean_noise(binary, zones)

    # 6. Build masks and debug overlay
    a_mask, b_mask, c_mask = build_zone_masks(binary, zones)
    overlay = build_debug_overlay(binary, zones)

    return PreprocessingResult(
        binary=binary,
        zones=zones,
        zone_a_mask=a_mask,
        zone_b_mask=b_mask,
        zone_c_mask=c_mask,
        debug_overlay=overlay,
    )


# ---------------------------------------------------------------------------
# CLI usage: python stage1_preprocessing.py <image_path>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stage1_preprocessing.py <image_path>")
        print("       Output saved to ./output/stage1_*.png")
        sys.exit(1)

    path = sys.argv[1]
    img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: cannot read image at '{path}'")
        sys.exit(1)

    result = preprocess(img)

    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/stage1_binary.png",  result.binary * 255)
    cv2.imwrite("output/stage1_zones.png",   result.debug_overlay)
    cv2.imwrite("output/stage1_zone_a.png",  result.zone_a_mask.astype(np.uint8) * 255)
    cv2.imwrite("output/stage1_zone_b.png",  result.zone_b_mask.astype(np.uint8) * 255)
    cv2.imwrite("output/stage1_zone_c.png",  result.zone_c_mask.astype(np.uint8) * 255)

    print("Stage 1 complete.")
    print(f"  Zone A rows: {result.zones.zone_a_rows}")
    print(f"  Zone B rows: {result.zones.zone_b_rows}")
    print(f"  Zone C rows: {result.zones.zone_c_rows}")
    print("  Saved to ./output/stage1_*.png")