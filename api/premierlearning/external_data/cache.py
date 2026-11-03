"""Polite HTTP fetching and a gzip JSON cache on disk."""
from __future__ import annotations

import gzip
import json
import os
import time

import requests

API_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTERNAL_DIR = os.path.join(API_ROOT, 'data', 'external')
RAW_DIR = os.path.join(EXTERNAL_DIR, 'raw')

USER_AGENT = 'Mozilla/5.0 (compatible; premier-learning research)'


def read_json_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def write_json_gz(path, data):
    """Write atomically, so a killed crawl never leaves half a file behind."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + '.tmp'
    with gzip.open(temp, 'wt', encoding='utf-8') as handle:
        json.dump(data, handle, separators=(',', ':'), ensure_ascii=False)
    os.replace(temp, path)


class Fetcher:
    """GET JSON with a minimum gap between requests and retries with backoff."""

    def __init__(self, min_interval: float = 0.25, retries: int = 5):
        self.min_interval = min_interval
        self.retries = retries
        self.session = requests.Session()
        self.session.headers['User-Agent'] = USER_AGENT
        self._last = 0.0
        self.count = 0

    def _wait(self):
        gap = time.time() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.time()

    def get(self, url, params=None):
        delay = 2.0
        for attempt in range(self.retries):
            self._wait()
            try:
                response = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as error:
                print(f'  network error {error!r}, retry in {delay:.0f}s', flush=True)
                time.sleep(delay)
                delay *= 2
                continue
            self.count += 1
            if response.status_code == 200:
                return response
            if response.status_code in (429, 500, 502, 503, 504):
                print(f'  HTTP {response.status_code} for {url}, retry in {delay:.0f}s', flush=True)
                time.sleep(delay)
                delay *= 2
                continue
            response.raise_for_status()
        raise RuntimeError(f'gave up on {url} {params}')

    def get_json(self, url, params=None):
        return self.get(url, params).json()
