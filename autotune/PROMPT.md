# Autonomous AutoMage tuning

Paste the block below into a fresh Claude Code session started at the root of this repository (with `.venv` set up),
with `<IMAGE>` and `<NAME>` filled in.

---

Produce the most accurate land-cover classification of `<IMAGE>` using AutoMage. Work fully autonomously:
no human review, and don't stop to ask me anything. Optimise for correctness on the image. Coverage, overlap
and prompt counts are diagnostics, not goals.

Use only the tools in `autotune/` (each script's docstring has its usage). Workspace: `autotune/runs/<NAME>/`.
Python is `.venv/bin/python`. Do not modify the `automage/` package; tune only through dictionaries and CLI flags.

## 1. Check the input and build the measuring stick before any tuning

1. Check the image: AutoMage classifies bands 1–3 as RGB, so confirm they are red, green, blue (inspect the file and
   look at a rendering). If they aren't, stop and report it; it can't be fixed from the CLI.
2. `autotune/sample_points.py <IMAGE> autotune/runs/<NAME> --n <N>` samples stratified points and renders labelling
   sheets. Use N = 300 for an image up to about 4 megapixels, and 500 for a larger or more varied scene. It prints
   the native ground GSD and stores it in `workspace.json`.
3. `autotune/label_points.py <ws> A` and `... <ws> B` (run both in parallel) label the points with two independent
   Claude sessions that see only the source image. Do not label points yourself, and do not edit the labels.
   Labels are saved after every sheet. If a labeller exits with failed sheets (e.g. a usage limit), wait and rerun
   the same command: it labels only the sheets still missing. Don't merge until both labellers have every sheet.
4. `autotune/merge_labels.py <ws>` keeps the points where both labellers agree and splits them 70% tune / 30% holdout.
   Never look at holdout results until the final report.
5. Look at the source image and the `reference.json` stats. In the journal, note the scene type, the classes
   actually present, and which visible surfaces have few or no reference points: those can only be judged by the
   reviewer.

## 2. Baseline

`autotune/run.sh <ws> 00 automage/dictionary.json -` runs the bundled dictionary with default settings. Start it as
soon as sampling is done, in parallel with labelling: if the reference isn't merged yet, `run.sh` finishes the run,
skips scoring with a warning, and prints the `score.py` command to run once `merge_labels.py` is done.
Note its run time and window count, and use them to size later runs. The number of windows grows roughly with
(native GSD / target GSD)², and with overlap. Set `--wall-seconds` with a margin, and treat any round whose
diagnostics don't show `"state": "complete"` as invalid.

Read `rounds/00/score.json`, `run/eval/prompts.csv` and the images in `run/eval/`. `score.json` measures two kinds
of error at the reference points:
- `errors`: the winning class is wrong (truth → predicted, with point ids). These show on the final map.
- `false_detections`: a wrong class is also detected at a point where the right class wins. These cause
  cross-class overlap, and they can start winning after any other change. They're counted only at points whose
  winner is right: at a point that is already an error, a second wrong class changes nothing and isn't counted again.

`prompts` attributes both to individual prompts: for every prompt, how many points it fires on rightly and wrongly.

`soft` covers the points the labellers didn't agree on, which are dropped from the strict reference and tend to sit
on boundaries, mixed ground and shadow. A soft point is acceptable when it is classified with a class either labeller
gave (any class if both were uncertain). Soft points never count in accuracy. They measure gap filling where the strict
points are thin, and they are lenient: a broad label that covers everything also scores well on them.

## 3. Rounds

Keep `<ws>/journal.md` and read it before every round. It holds:
- the current best round
- a table of every round: number, base, change, tune accuracy for features and objects, false detections, gate,
  reviewer verdict, accepted or rejected
- a Lessons section, including a list of prompts that never fire on this image

Never repeat a change the journal records as failed unless the base has changed in a way that could matter.

Each round:
1. **Pick the largest problem** in the best round's `score.json`. That is either an `errors` cluster, or a prompt that
   fires wrongly at many points. Look at the points on the source image and the overlays.
2. **Find the cause.** Run `autotune/point_prompts.py <ws> <ws>/rounds/<best> <point ids>` on the error's points (or
   a selector: `errors`, `truth=<class>`, `wins=<class>`; all use tune points only): it lists every class and prompt
   that fires there, with scores and windows. Do wrong prompts fire there, and by how
   much do they beat the right one? Is the feature right but the object wrong? Or does nothing fire (a detection
   gap)? `prompts` in `score.json` gives the same picture across all points.
3. **Make the smallest change that targets that cause:** 1–3 related prompt edits (`autotune/edit_dictionary.py`) or a
   single settings change. Write the hypothesis and expected effect in the journal before running.
4. **Run** `autotune/run.sh <ws> NN <dictionary> <best_round> [classify flags]`. It prints the numeric gate against the
   base, which passes when all of these hold:
   - no winning-class point is lost overall
   - objects net is at least −1
   - no class loses more than 1 point in features, nor in objects unless the features net is at least +5 (a broad
     phrase can swallow narrow surfaces in objects while fixing many feature points; such tolerated losses are
     listed in `objects_class_losses_allowed`, so record them and say in the report which layer to use for them)
   - gain is above 0, where gain = features net + 0.5 × false-detection net, plus 0.5 × soft net when no strict
     feature point is broken. For a round where only objects change, gain is the objects net (plus 0.5 × soft net in
     objects when no strict object point is broken).
5. **Review:** if the gate passes, run `autotune/review.py <ws> <ws>/rounds/<best> <ws>/rounds/NN`. This is a blind,
   independent reviewer that looks for what got worse in both layers (features and objects).
6. **Decide:**
   - Reject if the gate fails.
   - Reject if the reviewer prefers the base with high confidence.
   - If the reviewer prefers the base with low confidence, reject if the soft points also got worse
     (`soft_says_worse` in the gate output); otherwise accept only if gain ≥ 3. Record the disagreement, because it
     usually means the reviewer saw something in areas without reference points.
   - If the reviewer calls it a tie, accept: the gate says it's better and the reviewer sees no harm.
   - If the reviewer prefers the candidate, accept.
   Either way, record what was learned, including any useful part of a rejected change that is worth retesting alone.

Read `compare.json` every round, especially `broken` and `false_detections.added_pairs`. A change that fixes many
points but breaks a few is a lead for a narrower version, not something to accept as is.

## 4. Levers, in the order that has worked best

1. **Remove misleading prompts.** Start with prompts that are wrong at many points and right at almost none. Generic
   material or land-use words (a bare material name, "lot", "coarse" or "synthetic" grass) tend to grab every region
   of similar texture. Before removing one, check its `right` count, and ask what it finds legitimately where there
   are no reference points. If the reviewer rejects a removal because a real surface lost its label, replace the
   prompt with narrower phrases for that surface instead (e.g. "pedestrian sidewalk" instead of a bare material word).
   Drop types or classes with no support in the scene.
2. **Viewing:**
   - Try `--overlap 0.5` first (more views of each spot). Going higher rarely adds more. Like a finer scale, more
     overlap fills gaps but can make some prompts fire wrongly. If the overlap round fails only because of false
     detections or broken points that trace to specific prompts, run **one paired round: overlap 0.5 plus removing
     or narrowing those prompts**, judged against the same base.
   - Scale (`--target-gsd`) depends on the native ground GSD. Use `native_ground_gsd_m` from `workspace.json` or
     the diagnostics, never the raster's pixel size: in projected CRSs such as Web Mercator it is not ground metres.
     Check `presented_ground_gsd_m` in the diagnostics afterwards.
     - **Native coarser than about 8 cm:** try a finer scale of about 6 cm, or half of native if that is coarser.
       Do this early, right after overlap and before gap-filling prompts: once other fixes have filled the gaps,
       finer scale has little left to gain and mostly adds wrong detections.
     - **Native about 8 cm or finer:** don't go finer. The image is already at the scale that has worked best, and
       finer would multiply run time (windows grow with the square of the scale ratio) for little gain. Record
       "finer scale not applicable" in the journal; that counts as having tried the scale lever. Only if large
       objects (tree crowns, roofs) come out split into unclassified fragments, try one coarser round at about 8 cm.
   - A finer scale usually fills gaps but makes some prompts fire wrongly. If a scale round fails only because of
     false detections or broken points that trace to specific prompts, run **one paired round: the same scale plus
     removing or narrowing those prompts**, judged against the same base as the scale round.
   - If a finer scale starts splitting large objects (crowns, roofs) into unclassified fragments, the previous
     scale was the optimum.
3. **Detection gaps:** add specific, object-like phrases for surfaces where nothing fires. A new phrase often never
   fires at all, so put 2–3 alternative phrasings for the same gap in one round. Afterwards, check
   `feature_observations.jsonl` and record any phrases with zero detections as silent, so they aren't retried.
4. **Objects (last):** `--object-region-px` (superpixel size, in source pixels, so its ground size depends on the
   GSD: 28 px is about 3 m at 11 cm but about 1.6 m at 6 cm) and `--object-compactness`. Change
   these only once features are stable. Smaller superpixels follow boundaries better but leave more unclassified
   slivers; larger ones fill slivers but blur boundaries. Let the gate and the reviewer weigh that trade-off.
   An object takes the class of the strongest feature it overlaps, and a large merged feature (a whole field, a
   forest mass) often scores highest. So broad area phrases can swallow narrow surfaces (drives, paths, road edges)
   in the objects layer even when features get them right. Watch per-class objects results after adding a broad
   phrase, and prefer the features layer for narrow linear surfaces in the final report.

5. **Dark-gap fill (final step, optional):** if many errors are shadow or dark ground left unclassified, run
   `autotune/dark_fill.py <ws> <ws>/rounds/<best> <ws>/rounds/NN` on the best round after tuning ends. It labels
   still-unclassified dark pixels as shadow, then scores against the base; review it like any round and accept by
   the same rule. Later run.sh rounds re-run the model and would lose the fill, so apply it last.

## 5. Stopping

Stop after 30 rounds, or after 6 rejected rounds in a row once every lever (prompt removal, overlap, scale with its
paired cleanup or recorded as not applicable, gap phrases, objects) has been tried at least once. A round where nothing fired or nothing changed
counts as rejected. Don't stop because a single lever looks exhausted.

## 6. Final report

Run `autotune/score.py <ws> <ws>/rounds/00 --final` and the same for the best round. These write
`final_score.json` (tune and holdout) next to each round's `score.json`; read the holdout figures only from there. Write `<ws>/REPORT.md` with:
- holdout and tune accuracy (features and objects, overall and per class) for baseline vs. best
- false detections (tune and holdout), coverage, overlap and share of unclassified objects for baseline vs. best
- which layer to use: features or objects, based on holdout accuracy and gaps
- the final dictionary path and the exact `automage classify` command
- the table of all rounds
- the remaining error clusters, and the problems the reviewer saw in areas without reference points
- an honest note on the limits: the reference labels come from Claude reading the image and may be wrong, and
  surfaces the labellers couldn't agree on (often boundaries and mixed ground) have no reference points
