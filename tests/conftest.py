"""Shared test isolation for optional local embedding downloads."""

import os


os.environ.setdefault("PAPER_READER_DISABLE_EMBEDDINGS", "1")
