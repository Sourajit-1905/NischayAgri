# Pseudo-Severity Method

**Status:** documented heuristic — NOT ground truth. See "Limitations and
honest assessment" before citing these severity labels in quantitative
claims.

## Why pseudo-severity?

PlantVillage has no real severity annotations. The 38 class labels
encode crop species and disease identity, but not how severe the
disease expression is in any given image. Generating severity labels
was therefore necessary for the severity head (Phase 2.4) to have
anything to train on.

## Method

### Input

The `segmented/` image variant from PlantVillage (leaf isolated on a
near-pure-black background by the dataset's own segmentation method).
The `color/` variant (variable real-world backgrounds) was deliberately
NOT used for label generation because:
- Background pixels mixed into the lesion-ratio computation would
  produce systematically incorrect estimates.
- The `segmented/` variant gives reliable leaf-vs-background separation
  without requiring a separate segmentation model.

**Important:** although labels were derived from `segmented/`, the
severity head's training INPUTS are the `color/` variant — matching
what real inference will receive (a user's uploaded, unsegmented photo).

### Algorithm

1. **Background separation:** any pixel with all three RGB channels
   ≤ 10 is treated as background (black) and excluded. Healthy leaf
   tissue, veins, and genuine disease all have at least some color
   component above this floor.

2. **Healthy-green pixel classification:** a non-background pixel is
   classified as "healthy green" if:
   - Hue (HSV, 0–360°) is in [70°, 170°] (yellow-green through
     green to cyan-green), AND
   - Saturation (HSV, 0–1) ≥ 0.15 (not washed-out or grayish)
   Any non-background pixel NOT meeting these criteria is classified
   as "lesion-colored."

3. **Lesion-area ratio:** `lesion_pixels / leaf_pixels`

4. **Severity bucket (for TRAINING LABELS on known-diseased images):**
   - Healthy images (`*_healthy` class label): always `none` (see
     "Critical reframing" below — no pixel analysis performed)
   - Diseased images: `low` / `moderate` / `high` by ratio thresholds
     calibrated from this project's real data (see "Threshold
     calibration")

### Critical reframing discovered during calibration

**An early version of this method tried to use the color heuristic to
distinguish healthy from diseased images**, deriving a 4-bucket
none/low/moderate/high assignment purely from pixel content.

This failed empirically:
- Healthy leaves' median lesion_ratio (0.067) overlapped heavily with
  diseased leaves' 25th percentile (0.091).
- Brightness-based "dark lesion" detection caught ~28% of genuinely
  healthy leaf pixels as "dark lesion" — natural shading from leaf
  curvature, veins, and lighting gradients — regardless of any
  threshold tried.
- Morphological opening (3×3, 1 iteration) did not meaningfully
  change results (healthy avg ratio: 0.1115 vs 0.1066 before),
  ruling out small-pixel noise as the cause.

**Root cause:** the heuristic cannot reliably separate "disease-colored
pixels" from "normally-shaded-or-shadowed leaf pixels" using color
alone on PlantVillage images, because the two distributions overlap
too heavily in HSV space.

**Fix:** healthy/diseased status comes from the class label (always
reliable, since PlantVillage class names encode disease identity
directly). The pixel heuristic's scope was narrowed to ranking
severity **within already-diseased images only** — a task where
color is more discriminative, since it's comparing different amounts
of disease expression within the same disease class rather than
distinguishing disease from health.

This finding — that simple color features can rank intra-disease
severity but cannot reliably separate healthy from diseased leaves —
is consistent with the domain-adaptation literature, where feature
distributions of healthy and lightly-diseased tissue overlap
substantially.

### Threshold calibration

Bucket thresholds for diseased images were calibrated empirically from
374 stratified real PlantVillage images (up to 15 per diseased class,
across all 26 available diseased classes, using the fixed heuristic):

| Percentile | lesion_ratio |
|---|---|
| p10 | 0.001 |
| p33 | 0.014 |
| p50 | 0.046 |
| p66 | 0.092 |
| p90 | 0.313 |

Thresholds set at approximately p33/p66 for a balanced 3-way split:
- `low`: ratio ≤ 0.015
- `moderate`: 0.015 < ratio ≤ 0.09
- `high`: ratio > 0.09

**Final real-data bucket distribution (54,305 images):**
- `none`: 15,084 (healthy classes, assigned directly from label)
- `low`: 13,204
- `moderate`: 11,658
- `high`: 13,163

The near-balance across low/moderate/high is not accidental — it was
the goal of the empirical calibration.

## Limitations and honest assessment

1. **This is not ground truth.** PlantVillage has no real severity
   annotations. These labels are a methodological convenience, not an
   independent measurement. Any claim about the severity head's
   accuracy should acknowledge this.

2. **The color heuristic has known false-positive modes** even for
   diseased images: yellow/brown veins on a healthy leaf background,
   JPEG compression artifacts at high lesion ratios, and powdery
   mildew (white/gray, low-saturation) may be systematically
   misclassified at the low end (powdery mildew could appear as "low"
   or "none" severity despite significant visual coverage, since gray
   is classified as "healthy-ish" by the low-saturation rule).

3. **The 1,196 images with no `segmented/` counterpart** were skipped
   from severity label generation and therefore also from severity head
   training. These are likely images that failed the original
   PlantVillage segmentation step. Their absence from training is
   systematic, not random.

4. **The model is trained on `color/` images but labels came from
   `segmented/`** — there is an inherent domain mismatch between the
   label-generation context and the training context. This is considered
   acceptable because the alternative (using the color variant for
   label generation) would have been worse due to background noise, but
   it should be acknowledged.

5. **Generalization to field images (PlantDoc, FieldPlant) is
   untested.** Real-world severity (non-uniform backgrounds, variable
   lighting, partial leaf occlusion) is not represented in the training
   distribution at all.

## Config file

`ml/configs/severity/default.yaml` — contains all thresholds used by
the heuristic and the CORAL ordinal head config. Any change to
threshold values should trigger re-running
`scripts/compute_pseudo_severity.py` to regenerate the cache, since
the cache is tied to the specific threshold values used.