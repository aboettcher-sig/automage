"""Run one AutoMage analysis using private Cloud Storage inputs and outputs."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from google.cloud import storage


def gs_path(uri):
    if not uri.startswith('gs://'):
        raise ValueError('Expected a gs://bucket/object path')
    bucket, separator, name = uri[5:].partition('/')
    if not bucket or not separator or not name.strip('/'):
        raise ValueError('A bucket and object path are required')
    return bucket, name.rstrip('/')


def run(input_uri, output_uri, dictionary_uri=None, client=None, classify=None):
    from automage import Config, classify as automage_classify

    client = client or storage.Client()
    classify = classify or automage_classify
    bucket_name, prefix = gs_path(output_uri)
    bucket = client.bucket(bucket_name)
    status_blob = bucket.blob(prefix + '/status.json')

    def status(state, **details):
        record = {'state': state, 'input': input_uri, 'output': output_uri,
                  'execution': os.environ.get('CLOUD_RUN_EXECUTION'), **details}
        options = {'if_generation_match': 0} if state == 'running' else {}
        status_blob.upload_from_string(json.dumps(record), content_type='application/json', **options)
        print(json.dumps(record), flush=True)

    def download(uri, destination):
        source_bucket, name = gs_path(uri)
        client.bucket(source_bucket).blob(name).download_to_filename(destination)

    status('running')
    with tempfile.TemporaryDirectory(prefix='automage-') as directory:
        root = Path(directory)
        try:
            download(input_uri, root / 'image.tif')
            config = Config(objects=True, autocast_dtype='float16', wall_seconds=3000)
            if dictionary_uri:
                download(dictionary_uri, root / 'dictionary.json')
                config.dictionary = str(root / 'dictionary.json')
            result = classify(root / 'image.tif', root / 'result', config)
            # Preserve both the standard deliverables and their working provenance.
            for folder in (root / 'result', root / 'result.work'):
                for path in sorted(folder.rglob('*')):
                    if path.is_file():
                        name = prefix + '/' + path.relative_to(root).as_posix()
                        bucket.blob(name).upload_from_filename(path)
            if result['state'] != 'complete':
                raise RuntimeError('Analysis did not complete: ' + result['state'])
            status('complete', feature_count=result['feature_count'],
                   seconds=result['seconds'])
            return result
        except Exception as error:
            status('failed', error=f'{type(error).__name__}: {error}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Input GeoTIFF in Cloud Storage')
    parser.add_argument('--output', required=True, help='New Cloud Storage run prefix')
    parser.add_argument('--dictionary', help='Optional dictionary JSON in Cloud Storage')
    args = parser.parse_args()
    run(args.input, args.output, args.dictionary)


if __name__ == '__main__':
    main()
