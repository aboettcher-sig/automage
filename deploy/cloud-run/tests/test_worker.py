"""Exercise transfer, configuration, and failure reporting without cloud calls."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location('cloud_worker', Path(__file__).parents[1] / 'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Storage:
    def __init__(self):
        self.data = {'input/image.tif': b'image', 'input/dictionary.json': b'{"classes": []}'}
        self.states = []

    def bucket(self, name):
        return self

    def blob(self, name):
        client = self

        class Blob:
            def download_to_filename(self, path):
                Path(path).write_bytes(client.data[name])

            def upload_from_filename(self, path):
                client.data[name] = Path(path).read_bytes()

            def upload_from_string(self, value, **options):
                if options.get('if_generation_match') == 0 and name in client.data:
                    raise RuntimeError('Output already exists')
                client.data[name] = value.encode()
                client.states.append(json.loads(value)['state'])

        return Blob()


@pytest.mark.parametrize('custom_dictionary', [False, True])
def test_run_preserves_native_scale_and_publishes_results(custom_dictionary):
    client = Storage()

    def classify(source, out, config):
        assert source.read_bytes() == b'image'
        assert config.target_gsd is None and config.scale == 1
        assert config.autocast_dtype == 'float16' and config.objects
        assert config.wall_seconds == 3000
        if custom_dictionary:
            assert Path(config.dictionary).read_bytes() == client.data['input/dictionary.json']
        out.mkdir()
        (out / 'summary.json').write_text('{"state": "complete"}')
        work = out.with_name(out.name + '.work')
        work.mkdir()
        (work / 'provenance.json').write_text('{}')
        return {'state': 'complete', 'feature_count': 2, 'seconds': 1}

    worker.run('gs://test/input/image.tif', 'gs://test/runs/example',
               'gs://test/input/dictionary.json' if custom_dictionary else None,
               client=client, classify=classify)
    assert client.states == ['running', 'complete']
    assert 'runs/example/result/summary.json' in client.data
    assert 'runs/example/result.work/provenance.json' in client.data


def test_failure_is_reported_and_propagated():
    client = Storage()

    def classify(*args):
        raise RuntimeError('GPU failure')

    with pytest.raises(RuntimeError, match='GPU failure'):
        worker.run('gs://test/input/image.tif', 'gs://test/runs/example',
                   client=client, classify=classify)
    assert client.states == ['running', 'failed']
    assert 'GPU failure' in json.loads(client.data['runs/example/status.json'])['error']


def test_existing_output_is_not_overwritten():
    client = Storage()
    client.data['runs/example/status.json'] = b'{"state": "complete"}'
    with pytest.raises(RuntimeError, match='Output already exists'):
        worker.run('gs://test/input/image.tif', 'gs://test/runs/example', client=client)
    assert json.loads(client.data['runs/example/status.json'])['state'] == 'complete'


def test_incomplete_analysis_is_not_reported_as_success():
    client = Storage()
    with pytest.raises(RuntimeError, match='incomplete_window_limit'):
        worker.run('gs://test/input/image.tif', 'gs://test/runs/example', client=client,
                   classify=lambda *args: {'state': 'incomplete_window_limit'})
    assert client.states == ['running', 'failed']


def test_deploy_stops_before_cloud_changes_when_billing_is_disabled(tmp_path):
    command = tmp_path / 'gcloud'
    command.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$CALL_LOG"
case "$*" in
    *"auth list"*) echo test@example.com ;;
    *"projects describe"*"billingEnabled"*) echo False ;;
    *"projects describe"*) echo test-project ;;
    *) echo 'Unexpected cloud command' >&2; exit 99 ;;
esac
''')
    command.chmod(0o755)
    log = tmp_path / 'calls'
    environment = {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
                   'CALL_LOG': str(log), 'AUTOMAGE_PROJECT': 'test-project'}
    script = Path(__file__).parents[1] / 'cloud.sh'
    result = subprocess.run(['bash', str(script), 'deploy'], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert 'Billing is not enabled' in result.stderr
    assert len(log.read_text().splitlines()) == 3
