#!/bin/sh
# Run one round: run.sh <workspace> <round> <dictionary.json> <base_round|-> [extra automage classify flags...]
# Creates <workspace>/rounds/<round>/ with the run, diagnostics (run/eval/), score.json and, with a base, compare.json.
# If the reference isn't merged yet (labelling still running), it runs and evaluates but skips scoring with a warning.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE"); PY="$ROOT/.venv/bin/python"
WS=$(cd "$1" && pwd); DIR="$WS/rounds/$2"; DICT=$3; BASE=$4; shift 4
[ -e "$DIR/run" ] && { echo "$DIR/run already exists" >&2; exit 1; }
mkdir -p "$DIR"; [ "$DICT" -ef "$DIR/dictionary.json" ] || cp "$DICT" "$DIR/dictionary.json"
echo "$*" > "$DIR/flags.txt"
IMG=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['image'])" "$WS/workspace.json")
"$ROOT/.venv/bin/automage" classify --input "$IMG" --out "$DIR/run" --objects --dictionary "$DIR/dictionary.json" "$@" \
    > "$DIR/classify.log" 2>&1 || { tail -20 "$DIR/classify.log"; exit 1; }
"$PY" "$HERE/evaluate.py" "$DIR/run" > /dev/null
if [ ! -f "$WS/reference/reference.json" ]; then
    echo "WARNING: no reference/reference.json yet (labelling not merged); run finished but not scored." >&2
    echo "Score it later: score.py $WS $DIR${BASE:+ --base $WS/rounds/$BASE}" | sed 's# --base [^ ]*/rounds/-$##' >&2
elif [ "$BASE" = - ]; then "$PY" "$HERE/score.py" "$WS" "$DIR"
else "$PY" "$HERE/score.py" "$WS" "$DIR" --base "$WS/rounds/$BASE"; fi
