"""In-process job queue (thread pool) for long-running analyses, with progress reporting."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .util import log


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    def to_dict(self, include_result: bool = True) -> dict:
        d = {"job_id": self.id, "kind": self.kind, "status": self.status, "progress": round(self.progress, 3),
             "message": self.message, "params": self.params, "elapsed_s": round((self.finished or time.time()) - (self.started or self.created), 2)}
        if self.error:
            d["error"] = self.error
        if include_result and self.status == "done":
            d["result"] = self.result
        return d


class JobManager:
    def __init__(self, max_workers: int = 1, keep: int = 200):
        self._pool = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="audioref-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._keep = keep

    def submit(self, kind: str, fn: Callable[..., Any], params: dict | None = None, **kwargs: Any) -> str:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params or {})
        with self._lock:
            self._jobs[job.id] = job
            self._gc()

        def progress(p: float, msg: str = "") -> None:
            job.progress = max(job.progress, float(p))
            if msg:
                job.message = msg

        def run() -> None:
            job.status = "running"
            job.started = time.time()
            try:
                job.result = fn(progress=progress, **kwargs)
                job.status = "done"
                job.progress = 1.0
            except Exception as e:  # reported through get_job
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                log.error("job %s failed: %s\n%s", job.id, e, traceback.format_exc())
            finally:
                job.finished = time.time()

        self._pool.submit(run)
        return job.id

    def get(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"unknown job '{job_id}'")
            return self._jobs[job_id]

    def list(self) -> list[dict]:
        with self._lock:
            return [j.to_dict(include_result=False) for j in sorted(self._jobs.values(), key=lambda j: -j.created)]

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        t0 = time.time()
        while True:
            j = self.get(job_id)
            if j.status in ("done", "error"):
                return j
            if timeout is not None and time.time() - t0 >= timeout:
                return j
            time.sleep(0.1)

    def _gc(self) -> None:
        if len(self._jobs) <= self._keep:
            return
        done = sorted((j for j in self._jobs.values() if j.status in ("done", "error")), key=lambda j: j.created)
        for j in done[: len(self._jobs) - self._keep]:
            self._jobs.pop(j.id, None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
