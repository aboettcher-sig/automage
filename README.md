# AutoMage

AutoMage is a Python library for automated image analysis. Supply a georeferenced image and a classification dictionary, or use the default dictionary, to produce detected features and optional superpixel objects.

SAM3 detects features that may overlap. SLIC divides valid, processed imagery into non-overlapping superpixel objects, assigns proposed classes, and provides fields for review. A relationship table connects each object to the features that intersect it.

## Installation

Use Python 3.12 or later with GPU PyTorch installed for your accelerator. Install [Git LFS](https://git-lfs.com/) before cloning so Git retrieves the bundled weights:

```bash
git lfs install
git clone https://github.com/aboettcher-sig/automage.git
cd automage
```

Then install from the repository:

```bash
python -m pip install .
```

For the notebook, install `'.[notebook]'`; for tests, install `'.[test]'`. Installation declares PyTorch as a dependency but does not choose an accelerator-specific build. Neural inference requires a GPU. Geometry processing, SLIC, and export use the CPU. AMD ROCm uses PyTorch's CUDA API.

SAM3 weights and processor files are included with AutoMage (approximately 3.44 GB). The weights are stored as two standard model shards through Git LFS. Classification uses this bundled copy automatically and makes no model downloads. The weights are distributed under the included [SAM license](automage/models/sam3/LICENSE).

## Python

```python
from automage import Config, classify

config = Config(objects=True)
summary = classify("/path/to/image.tif", "/path/to/new_result", config)
```

To use the included demo image:

```python
from importlib.resources import files

image = files("automage").joinpath("data/boulder_features_objects.tif")
summary = classify(str(image), "/path/to/new_demo_result", config)
```

Set `Config(dictionary="/path/to/dictionary.json")` for a custom dictionary. The [default dictionary](automage/dictionary.json) has 17 classes, 44 detailed concepts, and 87 prompt entries. It groups `types` under `classes`; each type has a unique `id`, a `name`, and a nonempty `prompts` list. Prompt text conditions SAM3. Descriptions and irrigation metadata are not inferred attributes.

## Command line

```bash
automage plan --input /path/to/image.tif
automage classify \
  --input /path/to/image.tif \
  --out /path/to/new_result \
  --objects
```

`python -m automage` provides the same commands. `plan` describes the image and processing windows without loading a model. `classify` also accepts a directory of GeoTIFFs; keep its output directory outside the input tree.

Omit `--objects` for features only. Use `--dictionary` for a custom dictionary and `--target-gsd` for ground metres per model-input pixel. Native sampling is the default. Resampling changes the presented scale, not the detail acquired in the image.

The model window is 1008×1008 pixels with 20% overlap by default. Superpixels default to a target width of 28 source pixels, compactness 12, and 2048-pixel processing blocks. Block boundaries can introduce seams. `--object-region-px` and `--object-compactness` control the first two settings.

Inference defaults to float32 and prompt batches of eight. `--autocast-dtype float16` enables mixed precision on a compatible GPU. The default limits are 90 minutes, including five minutes reserved for exports, 32 GiB GPU allocation, and 10 GiB retained output. Additional settings, including RGB band selection, are available through `Config`.

## Outputs

| Output | Contents |
| --- | --- |
| `classification.gpkg` | `features` layer; with objects enabled, `objects` layer and `object_features` relationship table |
| `features/<class_name>.tif` | Band 1: detection score; band 2: feature ID, for each detected class |
| `objects.tif` | Optional superpixel IDs matching the GeoPackage |
| `overview.png` | Source imagery, features, and optional proposed object classes |
| `summary.json` | Status, counts, settings, runtime, and provenance |
| `<result>.work/` | Companion directory containing diagnostic rasters and observations |

Same-class detections accumulate when their intersection covers at least half the smaller mask. Touching alone does not merge features, and cross-class overlaps remain. A feature is not guaranteed to represent one physical entity.

Feature rasters display the highest-scoring feature where same-class features overlap, with the lowest feature ID breaking ties. Both bands use float64 to preserve IDs. All overlapping feature geometry remains in the GeoPackage. Rasters retain the source grid and CRS; invalid or unprocessed pixels are masked.

Superpixels cover valid, processed imagery. Each object's modal pixel class becomes `class_proposed`; `class_final` initially copies it. Unknown, mixed, or conflicting objects are flagged by `needs_review`. `reviewed` and `notes` support editing in GIS software. The relationship table retains intersecting feature IDs, metric intersection areas, and fractions of both geometries. The overview is a snapshot and does not refresh after manual edits.

Use a fresh output directory. `--resume` verifies and reuses an unchanged, complete result; it does not continue partial inference. Keep the `.work` directory for this verification. Editing a result changes its hashes and prevents immutable reuse. `--max-windows` produces an explicitly incomplete run.

Detection scores, coverage, purity, and review flags do not establish classification accuracy.

## Demo notebook

Open [AutoMage_Demo.ipynb](notebooks/AutoMage_Demo.ipynb) locally or in Colab. It installs AutoMage in Colab, classifies the included image or one you select, and displays the outputs. Select a GPU runtime. No separate model setup or Hugging Face credentials are needed.

Colab execution remains unverified. The GitHub installation cell requires this repository's contents to be published.

The included `boulder_features_objects.tif` is a 1600×1200 georeferenced crop, approximately 3 MB, from `2023-07-18T14_34_25_us-co-boulder-2023_4002923_-10525474_4003297_-10524817.tif`. Its source window starts at column 1250, row 950; ground sampling is approximately 11.4 cm per pixel.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests -q
```

The tests check tiling, dictionary prompts, feature merging, non-overlapping object coverage, raster/vector IDs, relationships, result reuse, and notebook syntax. They use controlled detections and do not require GPU inference or measure classification accuracy.
