"""Application logging: requests, background jobs and model runs, on the console and in the current profile's file.

The file lives next to that profile's data (`<data dir>/jobmagnet.log`) and rotates, so it never grows without end
and can be quoted in a bug report. Every line says which profile it belongs to, and background work also says
which job. Long jobs write their progress at DEBUG; set `JOBMAGNET_LOG_LEVEL=DEBUG` to see it in the console.
"""
import contextvars
import logging
import os
import sys
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

NAME = "jobmagnet"
LOG_NAME = "jobmagnet.log"
MAX_BYTES = 1_000_000
BACKUPS = 3

_job: contextvars.ContextVar[str | None] = contextvars.ContextVar("job_id", default=None)
_ready = False


def get_logger(name: str = "") -> logging.Logger:
    """A logger under the app's namespace. Configures logging the first time anything asks for one."""
    if not _ready:
        setup()
    return logging.getLogger(f"{NAME}.{name}" if name else NAME)


def bind_job(job_id: str) -> contextvars.Token:
    """Tag every record written on this thread with `job_id` (a background job's own context)."""
    return _job.set(job_id)


def current_job() -> str | None:
    """The background job this thread belongs to, if any (the AI trace uses it as its session)."""
    return _job.get()


def reset_job(token: contextvars.Token) -> None:
    _job.reset(token)


def log_path() -> Path:
    """The current profile's log file."""
    from .profiles import data_dir
    return data_dir() / LOG_NAME


def _level(value) -> int:
    if isinstance(value, int):
        return value
    return getattr(logging, str(value or os.environ.get("JOBMAGNET_LOG_LEVEL") or "INFO").upper(), logging.INFO)


class _Context(logging.Filter):
    """Add the profile and job to each record, so one formatter serves every handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        from .profiles import active
        record.profile = active() or "-"
        record.job = _job.get() or "-"
        return True


class _ProfileFileHandler(logging.Handler):
    """One rotating file per profile, opened when its first record arrives.

    The profile is resolved when a record is written, not when logging starts: a single process serves every profile
    (and background jobs keep the profile that started them), so the file has to follow the active one.
    """

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)
        self._files: dict[str, RotatingFileHandler] = {}

    def _file(self) -> RotatingFileHandler:
        path = log_path()
        key = str(path)
        handler = self._files.get(key)
        if handler is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8", delay=True)
            handler.setFormatter(self.formatter)
            self._files[key] = handler
        return handler

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._file().emit(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        for handler in self._files.values():
            handler.close()
        self._files.clear()
        super().close()

    def close_profile(self, pid: str) -> None:
        """Release a profile's file so its folder can be removed (Windows cannot delete an open file)."""
        from .profiles import data_dir
        key = str(data_dir(pid) / LOG_NAME)
        with self.lock:
            handler = self._files.pop(key, None)
        if handler is not None:
            handler.close()


def setup(level=None, stream=None) -> logging.Logger:
    """Configure logging once. File records are DEBUG; the console starts at `level` (or JOBMAGNET_LOG_LEVEL)."""
    global _ready
    logger = logging.getLogger(NAME)
    if _ready:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False   # handled here, never twice by the server's or a test runner's own handlers

    context = _Context()

    console = logging.StreamHandler(stream or sys.stderr)
    console.name = "jobmagnet-console"
    console.setLevel(_level(level))
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    console.addFilter(context)

    file = _ProfileFileHandler()
    file.name = "jobmagnet-file"
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(profile)s/%(job)s] %(name)s: %(message)s",
                                        "%Y-%m-%d %H:%M:%S"))
    file.addFilter(context)

    logger.addHandler(console)
    logger.addHandler(file)
    _ready = True
    return logger


def reset() -> None:
    """Drop every handler; the next setup() starts clean (tests, or a fresh log folder)."""
    global _ready
    logger = logging.getLogger(NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    _ready = False


def close_profile(pid: str) -> None:
    """Close a deleted profile's log file, so its folder can go on Windows too."""
    for handler in logging.getLogger(NAME).handlers:
        if isinstance(handler, _ProfileFileHandler):
            handler.close_profile(pid)


def tail(lines: int = 200) -> list[str]:
    """The last `lines` lines of the current profile's log, for the API or a bug report."""
    path = log_path()
    if not path.exists():
        return []
    limit = max(1, min(int(lines), 2000))
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return [line.rstrip("\n") for line in deque(f, maxlen=limit)]
