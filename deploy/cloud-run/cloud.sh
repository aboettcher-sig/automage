#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
project=${AUTOMAGE_PROJECT:-}
region=${AUTOMAGE_REGION:-us-east4}
job=automage-test
repository=automage-test
bucket="${project}-automage-test"
runtime_account="automage-test@${project}.iam.gserviceaccount.com"
build_account="automage-build@${project}.iam.gserviceaccount.com"
gc=(gcloud --project="$project" --quiet)
scratch=''
trap 'if [[ -n "$scratch" ]]; then rm -rf -- "$scratch"; fi' EXIT

usage() {
    cat <<'EOF'
Usage: bash deploy/cloud-run/cloud.sh COMMAND [ARGUMENTS]

  check                         Check login, project access, and billing (read-only)
  deploy                        Create test resources, build, and deploy the GPU job
  submit IMAGE.tif [DICTIONARY]  Upload inputs and start one analysis; print its run ID
  status RUN_ID                 Show Cloud Run execution and analysis status
  download RUN_ID DIRECTORY     Download results and working files into a new directory

Set AUTOMAGE_PROJECT to your Google Cloud project ID before running commands.
Default region: us-east4. Override with AUTOMAGE_REGION if needed.
EOF
}

check() {
    "${gc[@]}" auth list --filter=status:ACTIVE --format='value(account)'
    "${gc[@]}" projects describe "$project" --format='value(projectId,lifecycleState)'
    local billing
    billing=$("${gc[@]}" billing projects describe "$project" --format='value(billingEnabled)')
    if [[ "${billing,,}" != true ]]; then
        echo "Billing is not enabled for $project. No deployment can proceed." >&2
        echo "Link an active account: gcloud billing projects link $project --billing-account=ACCOUNT_ID" >&2
        return 1
    fi
    printf 'Ready: project=%s region=%s job=%s bucket=%s\n' "$project" "$region" "$job" "$bucket"
}

ensure_account() {
    local id=$1 found
    found=$("${gc[@]}" iam service-accounts list --filter="email=${id}@${project}.iam.gserviceaccount.com" --format='value(email)')
    if [[ -z "$found" ]]; then
        "${gc[@]}" iam service-accounts create "$id" --display-name="$id"
    fi
}

deploy() {
    check
    # Fail before cloud changes if a clone contains LFS pointers instead of weights.
    python3 - "$root" <<'PY'
import json, sys
from pathlib import Path
model = Path(sys.argv[1]) / 'automage/models/sam3'
index = json.loads((model / 'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    path = model / name
    if not path.is_file() or path.stat().st_size < 1024:
        raise SystemExit('Bundled weights missing. Run git lfs pull before deploying.')
PY
    "${gc[@]}" services enable run.googleapis.com artifactregistry.googleapis.com \
        cloudbuild.googleapis.com iam.googleapis.com storage.googleapis.com logging.googleapis.com
    ensure_account automage-test
    ensure_account automage-build
    if ! "${gc[@]}" storage buckets describe "gs://$bucket" >/dev/null 2>&1; then
        "${gc[@]}" storage buckets create "gs://$bucket" --location="$region" \
            --uniform-bucket-level-access --public-access-prevention
    fi
    if ! "${gc[@]}" artifacts repositories describe "$repository" --location="$region" >/dev/null 2>&1; then
        "${gc[@]}" artifacts repositories create "$repository" --location="$region" --repository-format=docker
    fi
    "${gc[@]}" storage buckets add-iam-policy-binding "gs://$bucket" \
        --member="serviceAccount:$runtime_account" --role=roles/storage.objectUser >/dev/null
    "${gc[@]}" storage buckets add-iam-policy-binding "gs://$bucket" \
        --member="serviceAccount:$build_account" --role=roles/storage.objectViewer >/dev/null
    "${gc[@]}" artifacts repositories add-iam-policy-binding "$repository" --location="$region" \
        --member="serviceAccount:$build_account" --role=roles/artifactregistry.writer >/dev/null
    "${gc[@]}" projects add-iam-policy-binding "$project" \
        --member="serviceAccount:$build_account" --role=roles/logging.logWriter --condition=None >/dev/null

    scratch=$(mktemp -d)
    # Send only package, deployment, and test files; never the local Git history.
    python3 - "$root" "$scratch" <<'PY'
import shutil, sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
for name in ('pyproject.toml', 'README.md', 'MANIFEST.in'):
    shutil.copy2(source / name, target / name)
for name in ('automage', 'tests', 'notebooks', 'deploy/cloud-run'):
    shutil.copytree(source / name, target / name,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
PY
    local image
    image="${region}-docker.pkg.dev/${project}/${repository}/automage:$(date -u +%Y%m%d-%H%M%S)"
    "${gc[@]}" builds submit "$scratch" --region="$region" \
        --config="$root/deploy/cloud-run/cloudbuild.yaml" --substitutions="_IMAGE=$image" \
        --service-account="projects/$project/serviceAccounts/$build_account" \
        --gcs-source-staging-dir="gs://$bucket/build-source"
    "${gc[@]}" run jobs deploy "$job" --region="$region" --image="$image" \
        --service-account="$runtime_account" --gpu=1 --gpu-type=nvidia-l4 \
        --no-gpu-zonal-redundancy --cpu=8 --memory=32Gi \
        --tasks=1 --parallelism=1 --max-retries=0 --task-timeout=3600s
    echo 'Deployment ready. No analysis has been started.'
}

submit() {
    [[ $# -ge 1 && $# -le 2 && -f "$1" ]] || { usage >&2; return 1; }
    if [[ $# == 2 && ! -f "$2" ]]; then echo 'Dictionary file not found.' >&2; return 1; fi
    local run_id prefix arguments
    run_id="$(date -u +%Y%m%d-%H%M%S)-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
    prefix="gs://$bucket/runs/$run_id"
    "${gc[@]}" storage cp "$1" "$prefix/input.tif"
    arguments="--input,$prefix/input.tif,--output,$prefix"
    if [[ $# == 2 ]]; then
        "${gc[@]}" storage cp "$2" "$prefix/dictionary.json"
        arguments+=",--dictionary,$prefix/dictionary.json"
    fi
    scratch=$(mktemp -d)
    "${gc[@]}" run jobs execute "$job" --region="$region" --args="$arguments" \
        --async --format=json > "$scratch/execution.json"
    "${gc[@]}" storage cp "$scratch/execution.json" "$prefix/execution.json"
    printf '\nRun ID: %s\nStatus: bash deploy/cloud-run/cloud.sh status %s\n' "$run_id" "$run_id"
}

run_prefix() {
    [[ "$1" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid run ID.' >&2; return 1; }
    printf 'gs://%s/runs/%s' "$bucket" "$1"
}

case "${1:-help}" in
    check|deploy|submit|status|download)
        : "${AUTOMAGE_PROJECT:?Set AUTOMAGE_PROJECT to your Google Cloud project ID}"
        ;;
esac

case "${1:-help}" in
    check) check ;;
    deploy) deploy ;;
    submit) shift; submit "$@" ;;
    status)
        [[ $# == 2 ]] || { usage >&2; exit 1; }
        prefix=$(run_prefix "$2")
        execution=$("${gc[@]}" storage cat "$prefix/execution.json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["metadata"]["name"])')
        "${gc[@]}" run jobs executions describe "$execution" --region="$region"
        if "${gc[@]}" storage objects describe "$prefix/status.json" >/dev/null 2>&1; then
            "${gc[@]}" storage cat "$prefix/status.json"
            printf '\n'
        fi
        ;;
    download)
        [[ $# == 3 ]] || { usage >&2; exit 1; }
        prefix=$(run_prefix "$2")
        [[ ! -e "$3" ]] || { echo 'Choose a new download directory.' >&2; exit 1; }
        mkdir -p -- "$3"
        "${gc[@]}" storage cp --recursive "$prefix/result" "$prefix/result.work" "$3/"
        ;;
    help|--help|-h) usage ;;
    *) usage >&2; exit 1 ;;
esac
