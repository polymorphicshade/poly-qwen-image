"""A single-worker job queue so one GPU serves many browser tabs and API clients."""

from __future__ import annotations

import io
import json
import logging
import queue
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .engine import Cancelled, Engine, Hooks, Result

log = logging.getLogger(__name__)

TERMINAL = ("done", "error", "cancelled")


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    run: Callable[[Hooks], Result | None] = field(repr=False)
    status: str = "queued"
    step: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    images: list[str] = field(default_factory=list)
    seed: int | None = None
    prompt: str | None = None
    variant: str | None = None
    info: dict = field(default_factory=dict)
    cancel_requested: bool = False
    # Latest live preview as PNG bytes, and the sampling step it shows.
    preview_png: bytes | None = field(default=None, repr=False)
    preview_step: int = 0
    done_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self, base_url: str = "/api") -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": {"step": self.step, "total": self.total},
            "message": self.message,
            "error": self.error,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "seed": self.seed,
            "prompt": self.prompt,
            "variant": self.variant,
            "info": self.info,
            "params": self.params,
            "images": [f"{base_url}/jobs/{self.id}/images/{i}" for i in range(len(self.images))],
            # The query string changes with every new preview so clients can cache-bust on it.
            "preview": f"{base_url}/jobs/{self.id}/preview?step={self.preview_step}" if self.preview_png else None,
            "preview_step": self.preview_step,
        }


class JobManager:
    def __init__(self, engine: Engine, output_dir: Path | str = "outputs", max_history: int = 200):
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.max_history = max_history
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, name="qwen-image-worker", daemon=True)
        self._worker.start()

    def submit(self, kind: str, params: dict, run: Callable[[Hooks], Result | None]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params, run=run)
        with self._lock:
            self.jobs[job.id] = job
            self._prune()
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job and job.status not in TERMINAL:
            job.cancel_requested = True
            if job.status == "queued":
                self._finish(job, "cancelled")
        return job

    def queue_position(self, job: Job) -> int:
        queued = [j for j in self.jobs.values() if j.status == "queued"]
        queued.sort(key=lambda j: j.created)
        return queued.index(job) + 1 if job in queued else 0

    def _prune(self):
        done = [j for j in self.jobs.values() if j.status in TERMINAL]
        done.sort(key=lambda j: j.created)
        for j in done[: max(0, len(self.jobs) - self.max_history)]:
            self.jobs.pop(j.id, None)

    def _finish(self, job: Job, status: str, error: str | None = None):
        job.status = status
        job.error = error
        job.finished = time.time()
        job.done_event.set()

    def _loop(self):
        while True:
            job = self._queue.get()
            if job.status != "queued":
                continue
            job.status = "running"
            job.started = time.time()

            def progress(step, total, job=job):
                job.step, job.total = step, total

            def status(msg, job=job):
                job.message = msg
                log.info("[%s] %s", job.id, msg)

            def preview(image, step, job=job):
                buf = io.BytesIO()
                image.save(buf, "PNG")
                job.preview_png, job.preview_step = buf.getvalue(), step

            hooks = Hooks(
                progress=progress,
                status=status,
                cancelled=lambda job=job: job.cancel_requested,
                preview=preview,
            )
            try:
                result = job.run(hooks)
                if result is not None:
                    self._store(job, result)
                job.message = "Done"
                self._finish(job, "done")
            except Cancelled:
                job.message = "Cancelled"
                self._finish(job, "cancelled")
            except Exception as e:  # surfaced to the client
                log.error("Job %s failed:\n%s", job.id, traceback.format_exc())
                job.message = "Failed"
                self._finish(job, "error", f"{type(e).__name__}: {e}")

    def _store(self, job: Job, result: Result):
        folder = self.output_dir / job.id
        folder.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(result.images):
            path = folder / f"{i}.png"
            img.save(path)
            job.images.append(str(path))
        job.seed, job.prompt, job.variant, job.info = result.seed, result.prompt, result.variant, result.info
        meta = {k: v for k, v in job.to_dict().items() if k not in ("images", "preview")} | {"files": [Path(p).name for p in job.images]}
        (folder / "job.json").write_text(json.dumps(meta, indent=2, default=str))
