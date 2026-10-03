# AutoMage

Automated image analysis for georeferenced imagery.

Give AutoMage a GeoTIFF and a classification dictionary to detect and map features with SAM3. Use the included dictionary or supply your own. Optionally generate non-overlapping superpixels with proposed class labels for review.

The SAM3 model is included with the package.

## Install

Requires Python 3.12+, a GPU with compatible PyTorch, and [Git LFS](https://git-lfs.com/) to retrieve the bundled model.

```bash
git lfs install
git clone https://github.com/aboettcher-sig/automage.git
cd automage
python -m pip install .
```

## Python

```python
from automage import Config, classify

classify(
    "image.tif",
    "results",
    Config(objects=True),
)
```

Use `Config(dictionary="classes.json")` to supply a custom dictionary.

## Command line

```bash
automage classify --input image.tif --out results --objects
```

Omit `--objects` to generate features only. Add `--dictionary classes.json` to use a custom dictionary.

## Outputs

- **GeoPackage** — detected features, optional superpixel objects, and their relationships.
- **GeoTIFFs** — feature IDs and scores by class, plus optional superpixel IDs.
- **Preview** — an overview of the results.
- **Summary** — processing settings and result counts.

Features can overlap. Superpixels divide the image into non-overlapping objects, with proposed labels and fields for review in GIS software.

## Demo

Try the [demo notebook](notebooks/AutoMage_Demo.ipynb) locally or in Google Colab with a GPU runtime. It includes a sample GeoTIFF.
