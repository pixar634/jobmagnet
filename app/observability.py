"""Local AI observability, in the shape of Langfuse: a trace per job or request, a generation per model call.

Every run of the Claude or Gemini CLI records the brief it was given, the reply it wrote, the model's token
usage, the cost the CLI reported and its latency. A background job is one trace, with the job id as its session,
so a job's model calls (and any retries) group under one id; a model call made outside a job opens a trace of its
own, named after the module that asked for it.

Traces are appended as JSON lines to `<data dir>/ai_traces.jsonl`, rotated so they never grow without end. They
stay on this machine: no account, no server, nothing sent anywhere. Read them with `python scripts/traces.py` or
GET /api/observability. Set `JOBMAGNET_OBSERVABILITY=0` to record nothing.

The log is append-only and crash-safe: a trace is written when it starts and again when it ends, and generations
as each one finishes, so a killed run still leaves what it did. `read()` folds those records back into traces.
This module never raises: a failed write costs a DEBUG line, never the model run.
"""
import contextlib
import contextvars
import datetime as dt
import hashlib
import json
import math
import os
import sys
import threading
import time
import uuid

from . import logs

log = logs.get_logger("observability")

FILE_NAME = "ai_traces.jsonl"
MAX_BYTES = 8_000_000
BACKUPS = 3
MAX_FIELD = 8_000          # characters of one brief or reply kept; longer text keeps its head, tail, length and hash
VERSION = 1
OFF = {"0", "off", "false", "no"}

# USD per million tokens (list price when this was written). Only used when the CLI doesn't report its own cost;
# a run whose envelope has total_cost_usd uses that instead.
PRICES = {
    "opus": (15.0, 75.0),
    "sonnet": (3.0, 15.0),
    "haiku": (0.80, 4.0),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
}
CACHE_READ_RATE = 0.1      # a cached input token costs a tenth of an input token
CACHE_WRITE_RATE = 1.25    # writing one costs a quarter more

_current: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar("ai_trace", default=None)
_session: contextvars.ContextVar[str | None] = contextvars.ContextVar("ai_session", default=None)
_write_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("JOBMAGNET_OBSERVABILITY", "1").strip().lower() not in OFF


def trace_path():
    """The current profile's trace file."""
    from .profiles import data_dir
    return data_dir() / FILE_NAME


# --- what a run is called ------------------------------------------------------------------------------------

def trace_name() -> str:
    """The module that asked for a model run, e.g. "tailor": the name a trace is filed under when none is given."""
    frame = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module.startswith("app.") and module not in ("app.observability", "app.engines"):
            return module[len("app."):].replace(".", "_")
        if not module.startswith(("app.", "contextlib")):
            return module.rsplit(".", 1)[-1] or "model"
        frame = frame.f_back
    return "model"


# --- one model call ------------------------------------------------------------------------------------------

def _now() -> str:
    return dt.datetime.now().isoformat(timespec="milliseconds")


def _message(error: BaseException) -> str:
    return _clip(f"{type(error).__name__}: {error}", 500) or ""


def _clip(value, limit: int = MAX_FIELD):
    """A brief or reply, cut to its head and tail: enough to debug, never enough to fill the disk."""
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    head, tail = limit * 3 // 4, limit // 4
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{text[:head]}\n…[{len(text)} chars, sha256:{digest}]…\n{text[-tail:]}"


def _pct(values: list[float], p: float):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 1)


def _price(model: str):
    name = (model or "").lower()
    for key, pair in PRICES.items():
        if key in name:
            return pair
    return None


def _profile() -> str | None:
    from .profiles import active
    return active()


class Generation:
    """One model call: what went in, what came out, what it cost, and how long it took."""

    def __init__(self, name: str, *, model: str = "", engine: str = "", input=None, metadata: dict | None = None):
        self.id = uuid.uuid4().hex[:16]
        self.trace_id = ""
        self.name = name
        self.model = model
        self.engine = engine
        self.input = input
        self.output = None
        self.metadata = dict(metadata or {})
        self.usage: dict | None = None
        self.cost_usd: float | None = None
        self.cost_source = ""
        self.level = "DEFAULT"
        self.status_message = ""
        self.started_at = _now()
        self.ended_at = ""
        self._started = time.perf_counter()

    def set_output(self, output) -> None:
        self.output = output

    def add_usage(self, *, input=None, output=None, total=None, cache_read=None, cache_creation=None) -> None:
        """Add this call's token counts (several models in one run, e.g. Gemini's envelopes, add up)."""
        given = {"input": input, "output": output, "total": total, "cache_read": cache_read, "cache_creation": cache_creation}
        fresh = {k: int(v) for k, v in given.items() if v is not None}
        if not fresh:
            return
        usage = self.usage or {"input": 0, "output": 0, "total": 0, "cache_read": 0, "cache_creation": 0}
        for key, value in fresh.items():
            usage[key] = usage.get(key, 0) + value
        if total is None:
            usage["total"] = usage["input"] + usage["output"] + usage["cache_read"] + usage["cache_creation"]
        self.usage = usage
        if self.cost_usd is None:
            self._estimate()

    def set_cost(self, usd, source: str = "cli") -> None:
        try:
            self.cost_usd = round(float(usd), 6)
            self.cost_source = source
        except (TypeError, ValueError):
            pass

    def _estimate(self) -> None:
        pair = _price(self.model or self.engine)
        if not pair or not self.usage:
            return
        price_in, price_out = pair
        u = self.usage
        cost = (u.get("input", 0) * price_in + u.get("output", 0) * price_out
                + u.get("cache_read", 0) * price_in * CACHE_READ_RATE
                + u.get("cache_creation", 0) * price_in * CACHE_WRITE_RATE) / 1_000_000
        self.cost_usd, self.cost_source = round(cost, 6), "estimate"

    @property
    def latency_ms(self) -> float:
        return round((time.perf_counter() - self._started) * 1000, 1)

    def finish(self) -> None:
        self.ended_at = _now()

    def record(self) -> dict:
        return {"type": "generation", "version": VERSION, "id": self.id, "trace_id": self.trace_id,
                "name": self.name, "model": self.model, "engine": self.engine,
                "started_at": self.started_at, "ended_at": self.ended_at or _now(),
                "latency_ms": self.latency_ms, "input": _clip(self.input), "output": _clip(self.output),
                "usage": self.usage, "cost_usd": self.cost_usd, "cost_source": self.cost_source,
                "level": self.level, "status_message": self.status_message, "metadata": self.metadata}


class Trace:
    """One operation: a background job, a request, or a single model call with no job around it."""

    def __init__(self, name: str, *, session_id: str | None = None, user_id: str | None = None,
                 input=None, metadata: dict | None = None, tags: list | None = None):
        self.id = uuid.uuid4().hex[:16]
        self.name = name
        self.session_id = session_id
        self.user_id = user_id
        self.input = input
        self.output = None
        self.metadata = dict(metadata or {})
        self.tags = list(tags or [])
        self.started_at = _now()
        self.ended_at: str | None = None
        self.status = "running"
        self.status_message = ""
        self.generations: list[Generation] = []
        self.scores: list[dict] = []
        self._started = time.perf_counter()
        self._lock = threading.Lock()

    @property
    def duration_ms(self) -> float:
        return round((time.perf_counter() - self._started) * 1000, 1)

    def add(self, generation: Generation) -> None:
        generation.trace_id = self.id
        with self._lock:
            self.generations.append(generation)

    def usage(self) -> dict:
        total = {"input": 0, "output": 0, "total": 0, "cache_read": 0, "cache_creation": 0}
        for gen in self.generations:
            for key, value in (gen.usage or {}).items():
                total[key] = total.get(key, 0) + value
        return total

    def cost_usd(self) -> float:
        return round(sum(g.cost_usd for g in self.generations if g.cost_usd is not None), 6)

    def record(self, final: bool = False) -> dict:
        rec = {"type": "trace", "version": VERSION, "id": self.id, "name": self.name,
               "session_id": self.session_id, "user_id": self.user_id, "started_at": self.started_at,
               "input": _clip(self.input), "output": _clip(self.output), "metadata": self.metadata,
               "tags": self.tags, "status": self.status, "status_message": self.status_message,
               "usage": self.usage(), "cost_usd": self.cost_usd(), "generation_count": len(self.generations),
               "score_count": len(self.scores)}
        if final:
            rec |= {"final": True, "ended_at": self.ended_at, "duration_ms": self.duration_ms}
        return rec


# --- writing -------------------------------------------------------------------------------------------------

def _rotate(path) -> None:
    if not path.exists() or path.stat().st_size < MAX_BYTES:
        return
    for i in range(BACKUPS - 1, 0, -1):
        old = path.with_name(f"{path.name}.{i}")
        if old.exists():
            old.replace(path.with_name(f"{path.name}.{i + 1}"))
    path.replace(path.with_name(f"{path.name}.1"))


def _append(record: dict) -> None:
    if not enabled():
        return
    try:
        path = trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _write_lock:
            _rotate(path)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except (OSError, TypeError, ValueError) as e:
        log.debug("could not write an AI trace: %s", e)


def _persist(generation: Generation) -> None:
    _append(generation.record())
    log.debug("model run %s %s: %.0f ms, %s tokens, %s", generation.name, generation.model or generation.engine,
              generation.latency_ms, (generation.usage or {}).get("total", "?"),
              f"${generation.cost_usd:.4f}" if generation.cost_usd is not None else "cost unknown")


def _finish(t: Trace, implicit: bool = False) -> None:
    t.ended_at = _now()
    if t.status == "running":
        t.status = "ok"
    _append(t.record(final=True))
    usage = t.usage()
    write = log.debug if implicit else log.info
    write("trace %s %s: %d model run(s), %d->%d tokens, $%.4f, %.0f ms", t.id, t.name, len(t.generations),
          usage["input"], usage["output"], t.cost_usd(), t.duration_ms)


# --- context -------------------------------------------------------------------------------------------------

def current() -> Trace | None:
    return _current.get()


def bind_session(session_id: str) -> contextvars.Token:
    """Group the model calls made on this thread or request under one session id."""
    return _session.set(session_id)


def reset_session(token: contextvars.Token) -> None:
    _session.reset(token)


@contextlib.contextmanager
def trace(name: str, *, session_id: str | None = None, user_id: str | None = None,
          input=None, metadata: dict | None = None, tags: list | None = None):
    """One operation around every model call inside it: a job, a request, a scan."""
    t = Trace(name, session_id=session_id or _session.get(), user_id=user_id or _profile(), input=input,
              metadata=metadata, tags=tags)
    _append(t.record())
    token = _current.set(t)
    try:
        yield t
    except BaseException as e:
        t.status, t.status_message = "error", _message(e)
        raise
    finally:
        _current.reset(token)
        _finish(t)


@contextlib.contextmanager
def generation(name: str = "", *, model: str = "", engine: str = "", input=None, metadata: dict | None = None):
    """One model call. Opens an implicit trace when no job or request trace is bound."""
    t = _current.get()
    owns = t is None
    token = None
    if t is None:
        t = Trace(name or trace_name(), session_id=_session.get() or _job_id(), user_id=_profile(),
                  metadata={"implicit": True})
        _append(t.record())
        token = _current.set(t)
    gen = Generation(name or t.name, model=model, engine=engine, input=input, metadata=metadata)
    t.add(gen)
    try:
        yield gen
    except BaseException as e:
        gen.level, gen.status_message = "ERROR", _message(e)
        raise
    finally:
        gen.finish()
        _persist(gen)
        if owns:
            if gen.level == "ERROR":
                t.status, t.status_message = "error", gen.status_message
            _current.reset(token)
            _finish(t, implicit=True)


def score(name: str, value, *, data_type: str = "NUMERIC", comment: str = "",
          trace_id: str | None = None, observation_id: str | None = None) -> None:
    """A number on a trace or generation (a fit score, a match, an ATS result). Needs a trace to belong to."""
    t = _current.get()
    tid = trace_id or (t.id if t else None)
    if not tid:
        return
    rec = {"type": "score", "version": VERSION, "id": uuid.uuid4().hex[:16], "trace_id": tid,
           "observation_id": observation_id, "name": name, "value": value, "data_type": data_type,
           "comment": comment, "at": _now()}
    _append(rec)
    if t is not None and t.id == tid:
        with t._lock:
            t.scores.append(rec)


def _job_id() -> str | None:
    from . import logs as app_logs
    return app_logs.current_job()


# --- reading -------------------------------------------------------------------------------------------------

def _records():
    """Every record on disk, oldest first (rotated files included), skipping a half-written last line."""
    path = trace_path()
    names = [f"{path.name}.{i}" for i in range(BACKUPS, 0, -1)] + [path.name]
    for name in names:
        f = path.with_name(name)
        if not f.exists():
            continue
        try:
            with f.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict):
                        yield rec
        except OSError:
            continue


def read(days: int | None = None) -> list[dict]:
    """Every trace, newest first, with its generations and scores folded in."""
    since = None
    if days:
        since = (dt.datetime.now() - dt.timedelta(days=days)).isoformat(timespec="milliseconds")
    traces: dict[str, dict] = {}
    for rec in _records():
        kind = rec.get("type")
        tid = rec.get("trace_id") or rec.get("id")
        if not tid:
            continue
        if kind == "trace":
            if since and str(rec.get("started_at") or "") < since:
                continue
            t = traces.setdefault(tid, {"id": tid, "generations": [], "scores": []})
            t.update({k: v for k, v in rec.items() if k != "type"})
        elif kind == "generation":
            t = traces.setdefault(tid, {"id": tid, "name": rec.get("name"), "generations": [], "scores": []})
            t["generations"] = [g for g in t["generations"] if g.get("id") != rec.get("id")] + [rec]
        elif kind == "score":
            t = traces.setdefault(tid, {"id": tid, "generations": [], "scores": []})
            t["scores"].append(rec)
    for t in traces.values():
        t["generations"].sort(key=lambda g: g.get("started_at") or "")
    # A generation whose trace record fell outside the window (or was never written) is not a trace of its own.
    kept = [t for t in traces.values() if t.get("started_at")]
    return sorted(kept, key=lambda t: str(t.get("started_at") or ""), reverse=True)


def recent(limit: int = 20, days: int | None = 30) -> list[dict]:
    return read(days)[:max(1, min(int(limit), 500))]


def get(trace_id: str) -> dict | None:
    return next((t for t in read() if t.get("id") == trace_id), None)


def summary(days: int = 7) -> dict:
    """Totals for the last `days`: runs, errors, tokens, cost and latency, by model and by name."""
    traces = read(days)
    generations = [g for t in traces for g in t.get("generations") or []]
    tokens = {"input": 0, "output": 0, "total": 0, "cache_read": 0, "cache_creation": 0}
    by_model: dict[str, dict] = {}
    by_name: dict[str, dict] = {}
    for g in generations:
        usage = g.get("usage") or {}
        for key in tokens:
            tokens[key] += int(usage.get(key) or 0)
        model = g.get("model") or g.get("engine") or "unknown"
        name = g.get("name") or "unknown"
        for bucket, key in ((by_model, model), (by_name, name)):
            row = bucket.setdefault(key, {"runs": 0, "errors": 0, "tokens": 0, "cost_usd": 0.0, "ms": []})
            row["runs"] += 1
            row["errors"] += int(g.get("level") == "ERROR")
            row["tokens"] += int(usage.get("total") or 0)
            row["cost_usd"] += float(g.get("cost_usd") or 0)
            row["ms"].append(float(g.get("latency_ms") or 0))
    for bucket in (by_model, by_name):
        for row in bucket.values():
            row["avg_ms"] = round(sum(row.pop("ms")) / row["runs"], 1) if row["runs"] else None
            row["cost_usd"] = round(row["cost_usd"], 4)
    errors = sum(1 for t in traces if t.get("status") == "error")
    latencies = [float(g.get("latency_ms") or 0) for g in generations]
    return {
        "days": days, "path": str(trace_path()), "traces": len(traces), "generations": len(generations),
        "trace_errors": errors, "generation_errors": sum(1 for g in generations if g.get("level") == "ERROR"),
        "error_rate": round(errors / len(traces), 3) if traces else 0.0,
        "tokens": tokens, "cost_usd": round(sum(g.get("cost_usd") or 0 for g in generations), 4),
        "unpriced_runs": sum(1 for g in generations if g.get("cost_usd") is None),
        "latency_ms": {"p50": _pct(latencies, 50), "p95": _pct(latencies, 95), "max": round(max(latencies), 1) if latencies else None},
        "by_model": dict(sorted(by_model.items())), "by_name": dict(sorted(by_name.items())),
    }
