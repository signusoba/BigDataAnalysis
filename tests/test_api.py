"""API contracts: label provenance, safe files, reviewed annotation split boundaries."""
from __future__ import annotations

import io
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from persian_ocr import api
from persian_ocr.arshasb import ArshasbDataset


def png_image(*, blank=False, offset=0):
    image = Image.new('RGB', (80, 80), 'white')
    if not blank:
        ImageDraw.Draw(image).rectangle((30 + offset, 20, 37 + offset, 60), fill='black')
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    root = tmp_path / 'dataset'
    page = root / '00001'
    page.mkdir(parents=True)
    (page / 'page_00001.png').write_bytes(png_image())
    (page / 'fulltext_00001.txt').write_text('ا', encoding='utf-8')
    pd.DataFrame([{'word': 'ا', 'line': 1, 'point1': '(30, 20)', 'point2': '(30, 61)',
                   'point3': '(38, 20)', 'point4': '(38, 61)'}]).to_excel(page / 'label_00001.xlsx', index=False)
    dataset = ArshasbDataset(root)
    monkeypatch.setattr(api, 'get_dataset', lambda: dataset)
    monkeypatch.setenv('PERSIAN_ANNOTATIONS_DIR', str(tmp_path / 'annotations'))
    monkeypatch.setenv('PERSIAN_ARTIFACTS_DIR', str(tmp_path / 'artifacts'))
    monkeypatch.setattr(api, 'get_recognizer', lambda: SimpleNamespace(loaded=False, model_name=None))
    with TestClient(api.app) as test_client:
        yield test_client


def test_ground_truth_is_not_reported_as_model_accuracy(client):
    overview = client.get('/api/overview').json()
    assert overview['dataset']['pages'] == 1
    assert overview['dataset']['words'] == 1
    assert overview['dataset']['line_count'] == 1
    assert overview['dataset']['unique_words'] == 1
    assert len(overview['letters']) == 32
    assert overview['model']['trained'] is False
    page = client.get('/api/dataset/pages/00001').json()
    assert page['transcription_source'] == 'dataset_ground_truth'
    assert page['words'][0]['bbox'] == [30, 20, 38, 61]
    assert page['full_text'] == 'ا'
    assert client.get('/api/dataset/pages').json()['total'] == 1
    assert client.get('/api/dataset/pages/00001/image?thumbnail=true').headers['content-type'] == 'image/png'
    assert client.get('/api/dataset/pages/00002').status_code == 404
    assert client.get('/api/dataset/pages/not-a-page').status_code == 404
    assert client.get('/api/evaluation').json() == {'status': 'not_trained'}


def test_predict_requires_model_and_readable_upload(client, monkeypatch):
    assert client.post('/api/predict', files={'file': ('x.png', png_image(), 'image/png')}).status_code == 503
    assert client.post('/api/predict', files={'file': ('x.png', b'not an image', 'image/png')}).status_code == 422
    assert client.post('/api/predict', files={'file': ('x.svg', b'<svg/>', 'image/svg+xml')}).status_code == 422
    monkeypatch.setattr(api, 'MAX_UPLOAD_BYTES', 20)
    assert client.post('/api/predict', files={'file': ('x.png', png_image(), 'image/png')}).status_code == 413


def test_predict_preserves_model_provenance_and_blank_error(client, monkeypatch):
    def predict(image):
        if image.convert('L').getextrema() == (255, 255):
            raise ValueError('Image is blank.')
        return {'prediction': {'index': 0, 'character': 'ا', 'name': 'alef', 'score': 0.8},
                'top_predictions': [], 'needs_review': True, 'warnings': [], 'latency_ms': 2.0,
                'preprocessed_image': 'data:image/png;base64,example'}
    monkeypatch.setattr(api, 'get_recognizer', lambda: SimpleNamespace(loaded=True, model_name='test', predict=predict))
    response = client.post('/api/predict', files={'file': ('x.png', png_image(), 'image/png')})
    assert response.status_code == 200
    assert response.json()['source'] == 'model_prediction'
    assert response.json()['prediction']['character'] == 'ا'
    assert response.json()['needs_review'] is True
    assert client.post('/api/predict', files={'file': ('x.png', png_image(blank=True), 'image/png')}).status_code == 422


def test_manual_annotations_validate_labels_bounds_and_split_groups(client):
    record = {'page_id': '00001', 'bbox': [25, 15, 45, 65], 'character': 'ا',
              'reviewer': 'reviewer one', 'group_id': 'page-1', 'split': 'train'}
    assert client.post('/api/annotations', json=record | {'character': 'word'}).status_code == 422
    assert client.post('/api/annotations', json=record | {'bbox': [-1, 0, 40, 60]}).status_code == 422
    assert client.post('/api/annotations', json=record | {'bbox': [25.5, 15, 45, 65]}).status_code == 422
    response = client.post('/api/annotations', json=record)
    assert response.status_code == 201
    result = response.json()
    assert result['verified'] is True
    assert result['source'] == 'manual_arshasb'
    assert result['path'] == result['image']
    assert result['label'] == result['character'] == 'ا'
    assert client.post('/api/annotations', json=record | {'split': 'test'}).status_code == 409
    assert client.post('/api/annotations', json=record).status_code == 409
    assert client.get('/api/annotations').json()['total'] == 1
    export = client.get('/api/annotations/export')
    assert export.status_code == 200
    assert 'ا' in export.content.decode('utf-8-sig')
    assert 'reviewer one' in export.text


def test_uploaded_samples_require_human_labels_and_keep_domains_separate(client):
    data = {'character': 'ك', 'reviewer': 'A', 'group_id': 'specimen-1', 'split': 'validation', 'source': 'upload'}
    response = client.post('/api/samples', data=data, files={'file': ('letter.png', png_image(), 'image/png')})
    assert response.status_code == 201
    assert response.json()['character'] == 'ک'
    assert response.json()['source'] == 'upload'
    assert client.post('/api/samples', data=data | {'split': 'test'},
                       files={'file': ('different.png', png_image(offset=2), 'image/png')}).status_code == 409
    assert client.post('/api/samples', data=data | {'source': 'synthetic'},
                       files={'file': ('letter.png', png_image(), 'image/png')}).status_code == 422
    assert client.post('/api/samples', data=data | {'group_id': 'blank'},
                       files={'file': ('blank.png', png_image(blank=True), 'image/png')}).status_code == 422


def test_recognizer_cache_refreshes_when_training_finishes_and_artifact_changes(tmp_path, monkeypatch):
    import hashlib
    import json
    import os
    import sys

    directory = tmp_path / 'artifacts'
    directory.mkdir()
    monkeypatch.setenv('PERSIAN_ARTIFACTS_DIR', str(directory))
    constructions = []

    class FakeRecognizer:
        def __init__(self, artifacts_dir):
            constructions.append(artifacts_dir)
            metadata_path = artifacts_dir / 'model_metadata.json'
            self.loaded = metadata_path.exists()
            self.model_name = 'hog' if self.loaded else 'untrained'
            if self.loaded:
                metadata = json.loads(metadata_path.read_text())
                actual = hashlib.sha256((artifacts_dir / 'hog.joblib').read_bytes()).hexdigest()
                if metadata['artifact_sha256'] != actual:
                    raise RuntimeError('Model artifact checksum mismatch.')

    # No heavyweight ML imports or training are needed to exercise cache invalidation.
    monkeypatch.setitem(sys.modules, 'persian_ocr.inference', SimpleNamespace(Recognizer=FakeRecognizer))
    api._load_recognizer.cache_clear()
    try:
        assert api.get_recognizer().loaded is False
        artifact = directory / 'hog.joblib'
        artifact.write_bytes(b'valid model')
        (directory / 'model_metadata.json').write_text(json.dumps({
            'selected_model': 'hog', 'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }))
        trained = api.get_recognizer()
        assert trained.loaded is True
        assert api.get_recognizer() is trained
        assert len(constructions) == 2
        original_stat = artifact.stat()
        artifact.write_bytes(b'bad!! model')  # same size, and restore mtime to verify byte-level detection
        os.utime(artifact, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        with pytest.raises(RuntimeError, match='checksum'):
            api.get_recognizer()
    finally:
        api._load_recognizer.cache_clear()


@pytest.mark.parametrize('failure_point', ['load', 'predict'])
def test_runtime_model_failure_returns_service_unavailable(client, monkeypatch, failure_point):
    def fail(*args):
        raise RuntimeError('Model artifact checksum mismatch.')

    if failure_point == 'load':
        monkeypatch.setattr(api, 'get_recognizer', fail)
    else:
        monkeypatch.setattr(api, 'get_recognizer', lambda: SimpleNamespace(loaded=True, predict=fail))
    response = client.post('/api/predict', files={'file': ('x.png', png_image(), 'image/png')})
    assert response.status_code == 503
    assert 'matching model artifact' in response.json()['detail']


def test_dataset_empty_cache_recovers_after_import(client, tmp_path):
    import shutil

    imported = tmp_path / 'imported'
    dataset = ArshasbDataset(imported)
    assert dataset.pages() == []
    assert dataset.summary()['pages'] == 0
    shutil.copytree(api.get_dataset().root, imported)
    assert len(dataset.pages()) == 1
    assert dataset.summary()['pages'] == 1
