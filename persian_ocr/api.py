"""Local prototype API: isolated-letter predictions and reviewed dataset annotations."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import threading
import uuid
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, StrictInt, field_validator

from .arshasb import ArshasbDataset, PROJECT_ROOT
from .letters import LETTERS, normalize_label

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
ANNOTATION_LOCK = threading.Lock()
WEB_DIR = PROJECT_ROOT / 'web'
app = FastAPI(title='Persian Letter Lab', version='0.1.0', description='Local research prototype; bind to localhost.')
app.mount('/static', StaticFiles(directory=str(WEB_DIR), check_dir=False), name='static')


@lru_cache(maxsize=1)
def get_dataset() -> ArshasbDataset:
    return ArshasbDataset()


def artifacts_dir() -> Path:
    return Path(os.getenv('PERSIAN_ARTIFACTS_DIR', str(PROJECT_ROOT / 'artifacts')))


def annotations_dir() -> Path:
    return Path(os.getenv('PERSIAN_ANNOTATIONS_DIR', str(PROJECT_ROOT / 'data' / 'annotations')))


@lru_cache(maxsize=4)
def _load_recognizer(directory: str, version: tuple[str, str]):
    from .inference import Recognizer
    return Recognizer(Path(directory))


def get_recognizer():
    directory = artifacts_dir().resolve()
    metadata_path = directory / 'model_metadata.json'
    try:
        metadata_bytes = metadata_path.read_bytes()
    except FileNotFoundError:
        return _load_recognizer(str(directory), ('missing', 'missing'))
    try:
        metadata = json.loads(metadata_bytes)
        if not isinstance(metadata, dict):
            raise ValueError('Expected a metadata object.')
    except (ValueError, TypeError) as error:
        raise RuntimeError('The trained model metadata is unreadable. Retrain or restore its matching report.') from error
    model_name = metadata.get('selected_model')
    filenames = {'hog': 'hog.joblib', 'cnn': 'cnn.pt', 'mobilenet': 'mobilenet.pt'}
    if model_name not in filenames:
        raise RuntimeError('The trained model report does not name a supported model.')
    artifact_digest = 'missing'
    if model_name in filenames:
        artifact_path = directory / filenames[model_name]
        try:
            # Hash the actual bytes on each lookup, including cached models. A changed or
            # corrupted artifact must pass the recognizer's checksum check before reuse.
            digest = hashlib.sha256()
            with artifact_path.open('rb') as artifact:
                while chunk := artifact.read(1024 * 1024):
                    digest.update(chunk)
            artifact_digest = digest.hexdigest()
        except FileNotFoundError:
            pass
    version = (hashlib.sha256(metadata_bytes).hexdigest(), artifact_digest)
    return _load_recognizer(str(directory), version)


def _json_file(path: Path, fallback):
    if not path.is_file():
        return fallback
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return fallback


def _model_status() -> dict:
    try:
        model = get_recognizer()
        trained = bool(model.loaded)
        name = model.model_name if trained else None
    except (FileNotFoundError, ImportError, OSError, ValueError, RuntimeError):
        trained, name = False, None
    metadata = _json_file(artifacts_dir() / 'model_metadata.json', {})
    return {'trained': trained, 'selected_model': name,
            'summary': metadata if trained else {'status': 'not_trained'}}


def _get_page(page_id: str) -> dict:
    try:
        return get_dataset().page(page_id)
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except (ValueError, OSError) as error:
        raise HTTPException(422, 'Dataset page annotations could not be read.') from error


async def _upload_image(file: UploadFile) -> Image.Image:
    content = bytearray()
    while chunk := await file.read(64 * 1024):
        content.extend(chunk)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, 'Image file exceeds the 10 MB limit.')
    if not content:
        raise HTTPException(422, 'Choose a nonempty image file.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in {'PNG', 'JPEG', 'WEBP'}:
                    raise HTTPException(422, 'Use a PNG, JPEG, or WebP image.')
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, 'Image exceeds the 16 megapixel limit.')
                image.load()
                oriented = ImageOps.exif_transpose(image)
                # Composite transparent images onto white rather than creating a black glyph background.
                rgba = oriented.convert('RGBA')
                background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
                background.alpha_composite(rgba)
                return background.convert('RGB')
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise HTTPException(413, 'Image dimensions exceed the safe decoding limit.') from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(422, 'The uploaded file is not a readable image.') from error


class AnnotationRequest(BaseModel):
    page_id: str = Field(pattern=r'^\d{5}$')
    bbox: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    character: str = Field(min_length=1, max_length=8)
    reviewer: str = Field(min_length=1, max_length=120)
    group_id: str = Field(min_length=1, max_length=120)
    split: Literal['train', 'validation', 'test']

    @field_validator('reviewer', 'group_id')
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('A nonempty reviewer and specimen/capture group are required.')
        return value


def _read_annotations() -> list[dict]:
    manifest = annotations_dir() / 'manifest.jsonl'
    if not manifest.is_file():
        return []
    try:
        return [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        raise HTTPException(500, 'The annotation manifest needs review; no records were overwritten.') from error


def _save_annotation(image: Image.Image, *, character: str, reviewer: str, group_id: str,
                     split: str, source: str, page_id: str | None = None,
                     bbox: list[int] | None = None) -> dict:
    try:
        label = normalize_label(character)
    except (ValueError, KeyError) as error:
        raise HTTPException(422, 'Label must be one of the 32 supported Persian letters.') from error
    letter = next((item for item in LETTERS if item['character'] == label), None)
    if letter is None:
        raise HTTPException(422, 'Label must be one of the 32 supported Persian letters.')
    reviewer, group_id = reviewer.strip(), group_id.strip()
    if not reviewer or not group_id or len(reviewer) > 120 or len(group_id) > 120:
        raise HTTPException(422, 'Reviewer and specimen/capture group must contain 1–120 characters.')
    # Only reject truly blank crops here; correctness/isolation remains an explicit human review task.
    extrema = image.convert('L').getextrema()
    if extrema[1] - extrema[0] < 5:
        raise HTTPException(422, 'The selected image is blank. Select one complete isolated letter, including its dots.')
    buffer = io.BytesIO()
    image.convert('RGB').save(buffer, format='PNG')
    content = buffer.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    record = {'id': uuid.uuid4().hex, 'image': f'images/{digest}.png', 'path': f'images/{digest}.png',
              'label': label, 'character': label, 'index': letter['index'], 'name': letter['name'],
              'page_id': page_id, 'bbox': bbox, 'group_id': group_id, 'split': split,
              'source': source, 'reviewer': reviewer, 'sha256': digest,
              'created_at': datetime.now(timezone.utc).isoformat(), 'verified': True}
    with ANNOTATION_LOCK:
        annotations = _read_annotations()
        for existing in annotations:
            same_image = existing.get('sha256') == digest
            same_group = existing.get('group_id') == group_id
            same_page = page_id is not None and existing.get('page_id') == page_id
            if (same_image or same_group or same_page) and existing['split'] != split:
                raise HTTPException(409, 'This image, source page, or specimen/capture group already belongs to another split.')
            if same_image:
                raise HTTPException(409, 'This exact image is already annotated. Duplicate images are not counted twice.')
        directory = annotations_dir()
        (directory / 'images').mkdir(parents=True, exist_ok=True)
        image_path = directory / record['path']
        image_path.write_bytes(content)
        with (directory / 'manifest.jsonl').open('a', encoding='utf-8') as manifest:
            manifest.write(json.dumps(record, ensure_ascii=False) + '\n')
    return record


@app.get('/')
def index():
    if not (WEB_DIR / 'index.html').is_file():
        raise HTTPException(503, 'The prototype interface has not been built yet.')
    return FileResponse(WEB_DIR / 'index.html')


@app.get('/api/health')
def health():
    return {'status': 'ok', 'dataset_available': get_dataset().available,
            'model_loaded': _model_status()['trained']}


@app.get('/api/overview')
def overview():
    return {'letters': LETTERS, 'dataset': get_dataset().summary(), 'model': _model_status(),
            'objectives': [
                'Recognize one complete isolated printed Persian letter from the 32-letter alphabet.',
                'Inspect supplied Arshasb word annotations and collect separately reviewed isolated-letter examples.',
                'Compare trained models using grouped held-out splits and report actual accuracy, macro F1, per-class recall, and latency.',
                'Collect independent camera samples before making claims about phone or real-world recognition.',
            ]}


@app.get('/api/dataset/pages')
def dataset_pages():
    pages = get_dataset().pages()
    return {'pages': pages, 'total': len(pages)}


@app.get('/api/dataset/pages/{page_id}')
def dataset_page(page_id: str):
    return _get_page(page_id)


@app.get('/api/dataset/pages/{page_id}/image')
def dataset_image(page_id: str, thumbnail: bool = False):
    _get_page(page_id)
    path = get_dataset().image_path(page_id)
    if not thumbnail:
        return FileResponse(path, media_type='image/png')
    with Image.open(path) as original:
        image = original.convert('RGB')
        image.thumbnail((160, 200))
        output = io.BytesIO()
        image.save(output, format='PNG')
    return Response(output.getvalue(), media_type='image/png')


@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    image = await _upload_image(file)
    try:
        model = get_recognizer()
        if not model.loaded:
            raise HTTPException(503, 'Train the isolated-letter model before requesting predictions.')
        result = model.predict(image)
    except HTTPException:
        raise
    except (FileNotFoundError, ImportError, OSError, RuntimeError) as error:
        raise HTTPException(503, 'The trained isolated-letter model is unavailable. Retrain or restore the matching model artifact and report.') from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return dict(result) | {'source': 'model_prediction'}


@app.get('/api/evaluation')
def evaluation():
    return _json_file(artifacts_dir() / 'metrics.json', {'status': 'not_trained'})


@app.post('/api/annotations', status_code=201)
def annotate(request: AnnotationRequest):
    page = _get_page(request.page_id)
    left, top, right, bottom = request.bbox
    if not (0 <= left < right <= page['width'] and 0 <= top < bottom <= page['height']):
        raise HTTPException(422, 'Crop bounds must fit inside the page with positive width and height.')
    with Image.open(get_dataset().image_path(request.page_id)) as image:
        crop = image.convert('RGB').crop(request.bbox)
    return _save_annotation(crop, character=request.character, reviewer=request.reviewer,
                            group_id=request.group_id, split=request.split, source='manual_arshasb',
                            page_id=request.page_id, bbox=list(request.bbox))


@app.post('/api/samples', status_code=201)
async def sample(file: UploadFile = File(...), character: str = Form(...), reviewer: str = Form(...),
                 group_id: str = Form(...), split: Literal['train', 'validation', 'test'] = Form(...),
                 source: Literal['camera', 'upload'] = Form('upload')):
    image = await _upload_image(file)
    return _save_annotation(image, character=character, reviewer=reviewer, group_id=group_id,
                            split=split, source=source)


@app.get('/api/annotations')
def annotations():
    records = _read_annotations()
    return {'annotations': records, 'total': len(records)}


@app.get('/api/annotations/export')
def export_annotations():
    output = io.StringIO(newline='')
    fields = ['image', 'character', 'page_id', 'bbox', 'group_id', 'split', 'source', 'reviewer', 'sha256']
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for record in _read_annotations():
        writer.writerow(record | {'bbox': json.dumps(record.get('bbox'), ensure_ascii=False)})
    return Response(output.getvalue().encode('utf-8-sig'), media_type='text/csv; charset=utf-8',
                    headers={'Content-Disposition': 'attachment; filename="persian-annotations.csv"'})
