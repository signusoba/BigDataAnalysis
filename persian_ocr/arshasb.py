"""Read the supplied Arshasb pages without treating word labels as model output."""
from __future__ import annotations

import ast
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data' / 'raw' / 'arshasb'
PAGE_ID = re.compile(r'^\d{5}$')


class DatasetUnavailable(FileNotFoundError):
    pass


class ArshasbDataset:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or os.getenv('ARSHASB_DATA_DIR', str(DEFAULT_DATA_DIR))).resolve()

    @property
    def available(self) -> bool:
        return self.root.is_dir() and any(self.root.glob('*/page_*.png'))

    def _directory(self, page_id: str) -> Path:
        if not PAGE_ID.fullmatch(page_id):
            raise FileNotFoundError('Page ID must contain exactly five digits.')
        page = self.root / page_id
        if not page.is_dir() or not (page / f'page_{page_id}.png').is_file():
            raise FileNotFoundError(f'Unknown dataset page: {page_id}')
        return page

    def image_path(self, page_id: str) -> Path:
        return self._directory(page_id) / f'page_{page_id}.png'

    @lru_cache(maxsize=256)
    def page(self, page_id: str) -> dict[str, Any]:
        folder = self._directory(page_id)
        labels = pd.read_excel(folder / f'label_{page_id}.xlsx', dtype={'word': str}, keep_default_na=False)
        required = {'word', 'line', 'point1', 'point2', 'point3', 'point4'}
        if not required.issubset(labels.columns):
            raise ValueError(f'Unsupported label columns for page {page_id}.')
        with Image.open(self.image_path(page_id)) as image:
            width, height = image.size
        words = []
        for index, row in labels.iterrows():
            points = [ast.literal_eval(row[f'point{i}']) for i in range(1, 5)]
            if any(not isinstance(p, (tuple, list)) or len(p) != 2
                   or any(type(coordinate) is not int for coordinate in p) for p in points):
                raise ValueError(f'Invalid word coordinates on page {page_id}.')
            bbox = [min(p[0] for p in points), min(p[1] for p in points),
                    max(p[0] for p in points), max(p[1] for p in points)]
            if not (0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height):
                raise ValueError(f'Word coordinates outside page {page_id}.')
            words.append({'index': int(index), 'text': str(row['word']), 'line': int(row['line']), 'bbox': bbox})
        return {'id': page_id, 'width': width, 'height': height,
                'image_url': f'/api/dataset/pages/{page_id}/image', 'words': words,
                'full_text': (folder / f'fulltext_{page_id}.txt').read_text(encoding='utf-8'),
                'transcription_source': 'dataset_ground_truth'}

    def pages(self) -> list[dict[str, Any]]:
        if not self.available:
            return []
        pages = []
        for folder in sorted(self.root.iterdir()):
            if (folder.is_dir() and PAGE_ID.fullmatch(folder.name)
                    and all((folder / filename).is_file() for filename in
                            (f'page_{folder.name}.png', f'label_{folder.name}.xlsx', f'fulltext_{folder.name}.txt'))):
                page = self.page(folder.name)
                pages.append({key: page[key] for key in ('id', 'width', 'height', 'image_url')}
                             | {'word_count': len(page['words']),
                                'thumbnail_url': page['image_url'] + '?thumbnail=true'})
        return pages

    def summary(self) -> dict[str, Any]:
        pages = self.pages()
        unique_words = {word['text'] for page in pages for word in self.page(page['id'])['words']}
        line_count = sum(len({word['line'] for word in self.page(page['id'])['words']}) for page in pages)
        return {
            'available': self.available, 'pages': len(pages),
            'words': sum(page['word_count'] for page in pages),
            'line_count': line_count, 'unique_words': len(unique_words),
            'description': 'Arshasb printed Persian text in the Far_ketab font; supplied word and line annotations.',
            'attribution': 'Arshasb by persiandataset; original LICENSE accompanies the imported dataset.',
            'limitations': [
                'Boxes and transcriptions are supplied ground truth, not OCR predictions or measured accuracy.',
                'The supplied pages label whole words, not isolated letters. Manual isolated-letter review is required for the 32-class recognizer.',
                'One printed font does not validate handwriting, unseen fonts, or real camera photographs.',
                'Repeated word images and invisible boundary zero-width non-joiners require care when splitting and scoring word OCR.',
            ],
        }
