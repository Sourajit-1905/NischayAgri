# Data Sheet: Crop Disease Datasets

Status: partial — class counts marked TBD require running
`ml/scripts/build_splits.py` against the real downloaded data.
Last updated: step 12 of Phase 2.2 (ml/ data layer).

## Overview

| Dataset | Domain | Role | Images (reported) | Classes (reported) |
|---|---|---|---|---|
| PlantVillage | Lab | Training source | ~54,300 | 38 |
| PlantDoc | Field | Target-domain eval 1 | ~2,482 | 27 (paper) / 29 (detection variant) |
| FieldPlant | Field | Target-domain eval 2 | 5,170 | 30 (Roboflow listing) |

"Reported" = the figures from each dataset's paper or distribution
listing, used as the `expected.n_images_approx` / `expected.n_classes`
sanity-check targets in `ml/configs/data/*.yaml`. Actual counts from
your local copy may differ; `build_splits.py` prints the real counts
and warns if they drift past the configured tolerance.

## PlantVillage

- **Source**: Hughes & Salathe (2015), arXiv:1511.08060
- **Role in this project**: the only training source (`training_sources`
  in `label_space.yaml`). All `in_domain` and `shared` classes derive
  from what PlantVillage contains.
- **Images**: 54,305 (actual, confirmed via `build_splits.py` — within
  0.01% of the ~54,300 reported figure)
- **Classes**: 38 (confirmed — all real folder names matched
  `label_space.yaml`'s mapping with zero unmapped classes)
- **Class balance**: confirmed imbalanced. Smallest class
  `potato__healthy` (152 images) vs. largest `orange__citrus_greening`
  (5,507 images) — a ~36x spread. Two other large classes:
  `soybean__healthy` (5,090), `tomato__yellow_leaf_curl_virus` (5,357).
  Worth accounting for via class-weighted loss in training
  (`ml/src/cropml/training/train_loop.py`, Phase 2.3).
- **Long-tail classes** (under 400 images): `potato__healthy` (152),
  `apple__cedar_apple_rust` (275), `peach__healthy` (360),
  `raspberry__healthy` (371), `tomato__mosaic_virus` (373),
  `grape__healthy` (423)
- **Near-duplicate rate**: 8,091 of 54,305 images (~14.9%) were
  clustered as near-duplicates by `dedup.py`'s perceptual hashing —
  confirms the dedup step was necessary, not precautionary. One class,
  `soybean__healthy`, has a large enough duplicate cluster that its
  achieved split ratio deviated from the 70/15/15 target
  (74.4/12.8/12.8) — not a bug (the group can't be subdivided further),
  but its test-set metrics may be noisier than other classes due to
  having proportionally fewer held-out images.
- **License**: **UNRESOLVED.** The original paper (arXiv:1511.08060)
  states the dataset is released under CC BY-SA 3.0, "with an important
  extension: algorithms trained on the data also fall under the same
  license" — meaning a strict reading would require any model trained
  on it to be released under CC BY-SA 3.0 too. Several popular mirrors
  (including a common Kaggle upload) instead list it as CC0. **This
  needs to be resolved before publication** — either track down which
  exact copy you downloaded and its stated terms, or default to the
  more restrictive CC BY-SA 3.0 interpretation from the original paper
  to be safe.
- **Known caveats**: lab-captured images on uniform/plain backgrounds —
  this is precisely the domain-gap problem the project's lab-to-field
  evaluation (PlantDoc, FieldPlant) is designed to measure. Do not
  download an offline-augmented variant (e.g. "New Plant Diseases
  Dataset (Augmented)" on Kaggle) — `plantvillage.yaml`'s
  `allow_offline_augmented: false` guard blocks this, since augmented
  copies bake near-duplicates into the data before our own dedup step
  ever runs.

## PlantDoc

- **Source**: Singh et al. (2020), "PlantDoc: A Dataset for Visual
  Plant Disease Detection," CoDS-COMAD 2020
- **Role in this project**: field-domain target-eval set 1. Per
  `plantdoc.yaml`'s `official_test_role: target_eval_only`, PlantDoc's
  own `test` split (231 images, per the paper) is used exclusively for
  evaluation and is never part of any train/val pool — enforced in
  `splitting.py` via `forced_test_sample_ids`.
- **Images**: 2,598 total (paper) / ~2,482 in the released train+test
  folders (2,251 train + 231 test) — the discrepancy is unexplained in
  the source material I found; `expected.tolerance: 0.10` in
  `plantdoc.yaml` accounts for this
- **Classes**: 27 (paper, classification) / 29 (a separately-described
  detection/annotation variant) — our mapping in `label_space.yaml`
  targets the 27-class classification version
- **Collection method**: scraped from Google Images and Ecosia, not
  collected directly by the authors
- **Class balance**: TBD, pending real counts
- **Long-tail classes**: TBD
- **License**: **UNVERIFIED.** No license was confirmed from the
  sources checked. More importantly: since images were scraped from
  Google Images and Ecosia rather than originated by the dataset
  authors, any dataset-level license does not necessarily clear
  per-image copyright. Check the dataset's repository/page directly
  before any public release of derived work.
- **Known caveats**: two of our mapped classes (`pepper_bell__leaf_spot`,
  `corn_maize__leaf_blight`) are marked `ambiguous` in
  `label_space.yaml` because their equivalence to a specific PlantVillage
  class is uncertain — they're excluded from both the shared-class and
  OOD evaluation sets to avoid contaminating either with a wrong guess.
  Our full PlantDoc class-name mapping in `label_space.yaml` was built
  from best-effort recollection, not verified against the actual
  downloaded folder names — confirm this once you have the real data
  (the loader will raise `UnmappedClassError` listing any folder name
  it can't resolve).

## FieldPlant

- **Source**: Moupojou et al. (2023), "FieldPlant: A Dataset of Field
  Plant Images for Plant Disease Detection and Classification With
  Deep Learning," IEEE Access 11:35398-35410
- **Role in this project**: field-domain target-eval set 2, and the
  primary source of field-only classes (e.g. the Cassava classes,
  `corn_maize__smut`) that populate the `unseen_in_training` OOD
  evaluation bucket.
- **Images**: 5,170 (paper)
- **Classes**: 30 (per the Roboflow Universe listing found during
  research) — only 7 of these class names were confirmed from available
  sources (5 Cassava classes, 2 Corn classes); the remaining ~23 are
  unknown and currently **not yet mapped** in `label_space.yaml`
- **Class balance / long-tail classes**: TBD — entirely unknown until
  the real dataset is downloaded and run through the loader
- **License**: **UNVERIFIED.** Check the source you download from
  (Roboflow Universe listing, or a data link from the paper itself)
  before use.
- **Known caveats**: the Plant Methods survey table (found during
  research) describes FieldPlant as including "augmented" images as
  part of its 5,170-image count — meaning near-duplicate contamination
  is a real risk for this dataset specifically, more so than for
  PlantVillage or PlantDoc. `fieldplant.yaml`'s
  `check_for_augmented_copies: true` flag exists for this reason, and
  the dedup step (`dedup.py`) is not optional for this dataset. The
  exact folder layout of a FieldPlant download is also unverified — the
  loader (`fieldplant.py`) currently assumes classification-style class
  folders and explicitly rejects any other `annotation_format` until
  this is confirmed.

## Action items before this data sheet is complete

1. Download all three datasets and run `python scripts/build_splits.py`
   for each, to replace every `TBD` above with real class counts and
   identify actual long-tail classes.
2. Resolve the PlantVillage license question (CC BY-SA 3.0 vs. CC0)
   against whichever specific copy you download.
3. Get the real FieldPlant folder structure and complete its class-name
   mapping in `label_space.yaml` (currently only 7 of ~30 classes are
   mapped) — the loader will list every unmapped folder name the first
   time you run it against real data.
4. Verify the PlantDoc class-name mapping in `label_space.yaml` against
   real folder names, since it was built from memory, not confirmed
   source data.
5. Source a small non-leaf / non-plant image set for the `non_leaf`
   OOD bucket (`ood_sets.py`) — none of the three datasets contain
   these, so this needs an external source (even a few dozen varied
   photos is enough to start).