"""Parquet-backed dataset reader for token extraction.

Reads audio (stored as encoded WAV/FLAC bytes in an ``audio`` struct column,
HF-Audio style) directly from parquet shards, so we never have to explode a
multi-million-clip corpus into individual files on disk.

Yields the SAME structure as ``JsonlDatasetReader`` —
``{"audio": Tensor(1, T), "label": meta}`` — so it is a drop-in input for
``extract_audio_tokens.py``; all downstream sharding/output is unchanged.

Language tag (``language_id``) per clip is either a constant or derived from
the parent directory name's last ``_``-separated segment (the ghana-speech
configs embed the ISO-639-3 code there, e.g. ``Ewe_ewe`` -> ``ewe``).
"""
import logging
import os

import torch
import torch.distributed as dist
import pyarrow.parquet as pq

from omnivoice.utils.audio import load_audio_bytes


class ParquetDatasetReader:
    def __init__(
        self,
        parquet_files: list[str],
        sample_rate: int = 24_000,
        text_column: str = "text",
        audio_column: str = "audio",
        lang_const: str | None = None,
        lang_from_dir: bool = False,
        normalize_audio: bool = True,
        batch_rows: int = 256,
    ):
        self.parquet_files = list(parquet_files)
        self.sample_rate = sample_rate
        self.text_column = text_column
        self.audio_column = audio_column
        self.lang_const = lang_const
        self.lang_from_dir = lang_from_dir
        self.normalize_audio = normalize_audio
        self.batch_rows = batch_rows

    def set_epoch(self, epoch: int):
        pass

    def _lang_for(self, path: str) -> str | None:
        if self.lang_from_dir:
            return os.path.basename(os.path.dirname(path)).split("_")[-1]
        return self.lang_const

    def _shard_files(self) -> list[str]:
        files = self.parquet_files
        # split across distributed ranks
        if dist.is_initialized():
            r, w = dist.get_rank(), dist.get_world_size()
            files = [f for i, f in enumerate(files) if i % w == r]
        # split across DataLoader workers
        wi = torch.utils.data.get_worker_info()
        if wi is not None:
            files = [f for i, f in enumerate(files) if i % wi.num_workers == wi.id]
        return files

    def __iter__(self):
        for path in self._shard_files():
            lang = self._lang_for(path)
            stem = os.path.splitext(os.path.basename(path))[0]
            parent = os.path.basename(os.path.dirname(path))
            try:
                pf = pq.ParquetFile(path)
            except Exception as e:
                logging.warning(f"Cannot open {path}: {e}")
                continue
            row_idx = 0
            for rb in pf.iter_batches(batch_size=self.batch_rows):
                cols = rb.to_pydict()
                audios = cols[self.audio_column]
                texts = cols.get(self.text_column, [None] * len(audios))
                for au, txt in zip(audios, texts):
                    rid = f"{parent}_{stem}_{row_idx:08d}"
                    row_idx += 1
                    ab = au.get("bytes") if isinstance(au, dict) else None
                    if not ab or not txt or not str(txt).strip():
                        continue
                    try:
                        waveform = torch.from_numpy(load_audio_bytes(ab, self.sample_rate))
                        if waveform.dim() == 1:
                            waveform = waveform.unsqueeze(0)
                        if self.normalize_audio:
                            waveform = (waveform / (waveform.abs().max() + 1e-7)) * 0.9
                        meta = {
                            "id": rid,
                            "text": str(txt),
                            "language_id": lang,
                            "audio_duration": waveform.shape[1] / self.sample_rate,
                        }
                        yield {"audio": waveform, "label": meta}
                    except Exception as e:
                        logging.warning(f"Skipping {rid}: {e}")

    def __len__(self) -> int:
        # best-effort; only used for progress estimation
        total = 0
        for f in self.parquet_files:
            try:
                total += pq.ParquetFile(f).metadata.num_rows
            except Exception:
                pass
        return total
