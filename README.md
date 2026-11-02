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

## Autotune

`autotune/` tunes AutoMage's dictionary and classify settings for one image, without a human reviewing each round. It
doesn't change the `automage` package: it works only through dictionaries and command-line flags.

It needs the package installed in `.venv` at the repository root and the [Claude Code](https://claude.com/claude-code)
CLI, which runs the labellers and the reviewer as separate headless sessions.

### How it works

1. **Goal.** You say what the classification is for (for example, irrigation status). The agent drafts a goal file:
   which classes matter, which confusions are costly, any new classes, and the caveats. It starts only after you
   approve it. See [`autotune/goals/README.md`](autotune/goals/README.md).
2. **Reference points.** Stratified points are labelled blind by two independent Claude labellers (Opus and Sonnet by
   default). Points they agree on become the reference, split 70% tune / 30% holdout. Points they disagree on are kept
   as lenient "soft" points.
3. **Rounds.** Each round makes one small change (1–3 prompt edits or one setting). A numeric gate compares it with the
   best round so far, weighted by the goal, and a blind reviewer looks for what got worse. The change is kept only if
   both agree.
4. **Final report.** Holdout accuracy, a goal map (`goal_map.py`), and an independent final accuracy assessment of the
   finished map (`final_assess.py`): area-weighted overall accuracy with 95% intervals, and user's and producer's
   accuracy per class.

### Running it

Open a fresh Claude Code session at the repository root and paste the block from
[`autotune/PROMPT.md`](autotune/PROMPT.md), with the image, a run name and the goal filled in. The goal step is the
only interactive one; after that the session runs on its own and writes `autotune/runs/<name>/REPORT.md`.
Each script's docstring gives its usage.

Runs (`runs/`) and image crops (`data/`) are kept out of git.

### `autotune_samepoints/`

The same tools with a cheaper final assessment. `autotune/final_assess.py` labels a whole fresh set of points drawn
from the finished map. `autotune_samepoints/final_assess.py` reuses the tuning points and their labels, weighted by
area, and labels new points only for map classes that have too few. Use it the same way, with
[`autotune_samepoints/PROMPT.md`](autotune_samepoints/PROMPT.md); runs go in `autotune_samepoints/runs/`.
