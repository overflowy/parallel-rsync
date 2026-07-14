#!/usr/bin/env python3

import argparse
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import yaml

_RICH_AVAILABLE = False
try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:
    pass


_PROGRESS_RE = re.compile(
    r"^\s*"
    r"(?P<bytes>[\d,]+)\s+"  # bytes transferred
    r"(?P<pct>\d+)%\s+"  # percentage
    r"(?P<speed>\S+/s)\s+"  # transfer speed
    r"(?P<eta>\S+)"  # ETA
    r"(?:\s+\(xfr#(?P<xfr>\d+)"  # files transferred
    r",\s*(?:to-chk|ir-chk)="
    r"(?P<remaining>\d+)/(?P<total>\d+)\))?"  # remaining/total files
)


def _parse_progress_line(line: str) -> dict | None:
    """Parse a single rsync --info=progress2 line into a dict."""
    m = _PROGRESS_RE.search(line)
    if not m:
        return None
    return {
        "bytes": int(m.group("bytes").replace(",", "")),
        "pct": int(m.group("pct")),
        "speed": m.group("speed"),
        "eta": m.group("eta"),
        "xfr": int(m.group("xfr")) if m.group("xfr") else 0,
        "remaining": int(m.group("remaining")) if m.group("remaining") else 0,
        "total": int(m.group("total")) if m.group("total") else 0,
    }


def default_config_candidates() -> list[Path]:
    """Return the XDG fallback config paths, in search order."""
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    config_dir = Path(xdg_config_home) / "parallel-rsync"
    return [config_dir / "config.yml", config_dir / "config.yaml"]


def resolve_setting(cli_value: int | None, cfg: dict, key: str, default: int) -> int:
    """Resolve a positive-integer setting: CLI flag > config key > default."""
    value = cli_value if cli_value is not None else cfg.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"'{key}' must be a positive integer (got {value!r}).")
    return value


def resolve_config_path(explicit: Path | None) -> Path:
    """Resolve the config file path, falling back to the XDG location."""
    if explicit is not None:
        return explicit
    candidates = default_config_candidates()
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No config file found. Pass -c/--config or create one of: {checked}"
    )


def load_config(path: Path) -> dict:
    """Load and validate the YAML configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "groups" not in cfg:
        raise ValueError("YAML must contain a top‑level 'groups' key.")
    if not isinstance(cfg["groups"], list):
        raise ValueError("'groups' must be a list of group definitions.")
    if "global_options" in cfg and not isinstance(cfg["global_options"], list):
        raise ValueError("'global_options' must be a list of rsync option strings.")
    for group in cfg["groups"]:
        excludes = group.get("exclude_options")
        if excludes is not None and (
            not isinstance(excludes, list) or not all(isinstance(e, str) for e in excludes)
        ):
            name = group.get("name", "<unnamed>")
            raise ValueError(f"Group '{name}': 'exclude_options' must be a list of strings.")
    return cfg


def ensure_dest_dir(dest: str, logger: logging.Logger, name: str) -> None:
    """Create the local destination directory if it doesn't exist."""
    if ":" in dest:
        logger.debug(f"[{name}] Remote destination, skipping local mkdir")
        return
    dest_path = Path(dest)
    if not dest_path.exists():
        logger.info(f"[{name}] Creating destination directory: {dest_path}")
        dest_path.mkdir(parents=True, exist_ok=True)


def _option_excluded(option: str, excludes: list[str]) -> bool:
    """Return True if *option* matches an exclusion entry.

    An entry matches on the exact string or on the option name before '='
    (so "--rsync-path" excludes "--rsync-path=sudo rsync").
    """
    return any(option == e or option.startswith(e + "=") for e in excludes)


def build_rsync_cmd(group: dict, global_options: list[str] | None = None) -> list[str]:
    """Construct the rsync command list for a given group."""
    src = group.get("src")
    dest = group.get("dest")
    group_options = group.get("options") or []
    excludes = group.get("exclude_options") or []
    if not src or not dest:
        raise ValueError(f"Group '{group.get('name', '<unnamed>')}' missing src or dest.")
    if not src.endswith("/"):
        src = src + "/"
    kept_globals = [o for o in (global_options or []) if not _option_excluded(o, excludes)]
    cmd = ["rsync"] + kept_globals + list(group_options) + [src, dest]
    return cmd


def extract_host(dest: str) -> str:
    """Extract the hostname from an rsync destination string."""
    if "::" in dest:
        host_part = dest.split("::", 1)[0]
        return host_part.split("@", 1)[1] if "@" in host_part else host_part
    if ":" in dest:
        colon_idx = dest.index(":")
        if colon_idx == 1 and dest[0].isalpha():
            return "local"
        host_part = dest.split(":", 1)[0]
        return host_part.split("@", 1)[1] if "@" in host_part else host_part
    return "local"


def setup_logging(log_level: str, log_file: str | None = None) -> logging.Logger:
    """Configure a logger.

    If *log_file* is provided, log messages are written to that file only —
    nothing is printed to the console (the rich progress display owns the
    terminal).  When no log file is given, a NullHandler is attached so
    that logging calls are silently discarded; the progress bars and
    summary table remain the sole user-facing output.
    """
    logger = logging.getLogger("parallel_rsync")
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    if log_file:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


_STATUS = {
    "waiting": "[dim]waiting[/dim]",
    "running": "[bold cyan]syncing[/bold cyan]",
    "done": "[bold green]✔  done[/bold green]",
    "failed": "[bold red]✖  failed[/bold red]",
    "timeout": "[bold yellow]⏱  timeout[/bold yellow]",
}


def _inject_progress2(cmd: list[str]) -> list[str]:
    """Ensure --info=progress2 is present so we can parse progress."""
    has_progress2 = any("--info=progress2" in arg for arg in cmd)
    if not has_progress2:
        # Insert right after 'rsync'
        return [cmd[0], "--info=progress2"] + cmd[1:]
    return list(cmd)


def run_rsync_live(
    group: dict,
    global_options: list[str],
    semaphores: dict[str, threading.Semaphore],
    logger: logging.Logger,
    progress: "Progress | None",
    task_id: "TaskID | None",
    timeout: int | None = None,
) -> dict:
    """Execute rsync for one group, streaming progress to the rich bar."""
    name = group.get("name", "unnamed")
    src = group.get("src", "")
    dest = group.get("dest", "")

    host = extract_host(src)
    if host == "local":
        host = extract_host(dest)
    semaphore = semaphores[host]

    def _log(msg: str, level: str = "info") -> None:
        """Log to file only (if configured). The progress bars handle console output."""
        getattr(logger, level)(msg)

    # -- waiting --
    if progress and task_id is not None:
        progress.update(task_id, description=f"{_STATUS['waiting']}  [bold]{name}[/bold]")

    _log(f"[{name}] Waiting for slot on host '{host}'")

    with semaphore:
        try:
            if group.get("mkdir_dest", False):
                ensure_dest_dir(dest, logger, name)

            cmd = build_rsync_cmd(group, global_options)
            cmd = _inject_progress2(cmd)
            cmd_str = shlex.join(cmd)

            # -- running --
            if progress and task_id is not None:
                progress.update(
                    task_id,
                    description=f"{_STATUS['running']}  [bold]{name}[/bold]",
                )

            _log(f"[{name}] Starting rsync on host '{host}': {cmd_str}")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            stdout_lines: list[str] = []
            stderr_lines: list[str] = []

            # Read stdout in a thread so we can enforce timeout
            def _read_stdout():
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\n\r")
                    stdout_lines.append(line)
                    parsed = _parse_progress_line(line)
                    if parsed and progress and task_id is not None:
                        progress.update(
                            task_id,
                            completed=parsed["pct"],
                            speed=parsed["speed"],
                            eta=parsed["eta"],
                        )

            def _read_stderr():
                assert proc.stderr is not None
                for raw_line in proc.stderr:
                    stderr_lines.append(raw_line.rstrip("\n\r"))

            t_out = threading.Thread(target=_read_stdout, daemon=True)
            t_err = threading.Thread(target=_read_stderr, daemon=True)
            t_out.start()
            t_err.start()

            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                t_out.join(timeout=2)
                t_err.join(timeout=2)
                _log(f"[{name}] rsync timed out after {timeout}s on host '{host}'", "error")
                if progress and task_id is not None:
                    progress.update(
                        task_id,
                        description=f"{_STATUS['timeout']}  [bold]{name}[/bold]",
                    )
                return {
                    "name": name,
                    "host": host,
                    "cmd": cmd_str,
                    "returncode": -2,
                    "stdout": "\n".join(stdout_lines),
                    "stderr": f"Timed out after {timeout}s",
                }

            t_out.join(timeout=5)
            t_err.join(timeout=5)

            rc = proc.returncode
            stdout_text = "\n".join(stdout_lines)
            stderr_text = "\n".join(stderr_lines)

            _log(f"[{name}] rsync completed with exit code {rc}")
            if stderr_text:
                _log(f"[{name}] STDERR:\n{stderr_text}", "warning")

            # -- done / failed --
            if progress and task_id is not None:
                if rc == 0:
                    progress.update(
                        task_id,
                        completed=100,
                        description=f"{_STATUS['done']}  [bold]{name}[/bold]",
                    )
                else:
                    progress.update(
                        task_id,
                        description=f"{_STATUS['failed']}  [bold]{name}[/bold]",
                    )

            return {
                "name": name,
                "host": host,
                "cmd": cmd_str,
                "returncode": rc,
                "stdout": stdout_text,
                "stderr": stderr_text,
            }

        except Exception as e:
            _log(f"[{name}] Exception while running rsync: {e}", "error")
            if progress and task_id is not None:
                progress.update(
                    task_id,
                    description=f"{_STATUS['failed']}  [bold]{name}[/bold]",
                )
            return {
                "name": name,
                "host": host,
                "cmd": shlex.join(build_rsync_cmd(group, global_options)),
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
            }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

_RSYNC_EXIT_CODES = {
    0: "Success",
    1: "Syntax or usage error",
    2: "Protocol incompatibility",
    3: "Errors selecting input/output files, dirs",
    4: "Requested action not supported",
    5: "Error starting client-server protocol",
    6: "Daemon unable to append to log-file",
    10: "Error in socket I/O",
    11: "Error in file I/O",
    12: "Error in rsync protocol data stream",
    13: "Errors with program diagnostics",
    14: "Error in IPC code",
    20: "Received SIGUSR1 or SIGINT",
    21: "Some error returned by waitpid()",
    22: "Error allocating core memory buffers",
    23: "Partial transfer due to error",
    24: "Partial transfer due to vanished source files",
    25: "The --max-delete limit stopped deletions",
    30: "Timeout in data send/receive",
    35: "Timeout waiting for daemon connection",
}

_STDERR_TAIL_LINES = 10


def _describe_exit(rc: int) -> str:
    """Human-readable label for a result's return code."""
    if rc == -2:
        return "timeout"
    if rc == -1:
        return "exception"
    meaning = _RSYNC_EXIT_CODES.get(rc)
    return f"exit {rc} ({meaning})" if meaning else f"exit {rc}"


def _failure_detail(result: dict) -> tuple[str, list[str], int]:
    """Return (header, stderr tail lines, omitted line count) for a failed result."""
    header = f"{result['name']} — {_describe_exit(result['returncode'])}"
    lines = [line for line in result["stderr"].splitlines() if line.strip()]
    omitted = max(0, len(lines) - _STDERR_TAIL_LINES)
    return header, lines[-_STDERR_TAIL_LINES:], omitted


def _print_failure_details(failures: list[dict]) -> None:
    """Print the command and stderr tail for each failed group."""
    if _RICH_AVAILABLE:
        console = Console()
        for r in sorted(failures, key=lambda x: x["name"]):
            header, tail, omitted = _failure_detail(r)
            console.print(f"[bold red]✖ {header}[/bold red]")
            console.print(f"  [dim]$ {r['cmd']}[/dim]")
            if omitted:
                console.print(f"  [dim]... ({omitted} earlier lines omitted, see --log-file)[/dim]")
            for line in tail:
                console.print(f"  {line}", markup=False, highlight=False)
            console.print()
    else:
        for r in sorted(failures, key=lambda x: x["name"]):
            header, tail, omitted = _failure_detail(r)
            print(f"[FAIL] {header}")
            print(f"  $ {r['cmd']}")
            if omitted:
                print(f"  ... ({omitted} earlier lines omitted, see --log-file)")
            for line in tail:
                print(f"  {line}")
            print()


def _print_summary(results: list[dict]) -> None:
    """Print a pretty summary table using rich (falls back to plain text)."""
    failures = [r for r in results if r["returncode"] != 0]
    successes = len(results) - len(failures)

    if _RICH_AVAILABLE:
        console = Console()
        console.print()

        table = Table(
            title="[bold]Summary[/bold]",
            show_lines=True,
            title_style="bold cyan",
            border_style="dim",
        )
        table.add_column("Group", style="bold")
        table.add_column("Host", style="dim")
        table.add_column("Exit Code", justify="center")
        table.add_column("Status", justify="center")

        for r in sorted(results, key=lambda x: x["name"]):
            rc = r["returncode"]
            if rc == 0:
                status = Text("✔ Success", style="bold green")
                rc_text = Text(str(rc), style="green")
            elif rc == -2:
                status = Text("⏱ Timeout", style="bold yellow")
                rc_text = Text("timeout", style="yellow")
            else:
                status = Text("✖ Failed", style="bold red")
                rc_text = Text(str(rc), style="red")
            table.add_row(r["name"], r["host"], rc_text, status)

        console.print(table)

        if failures:
            console.print(
                f"\n[bold red]✖ {len(failures)} job(s) failed[/bold red]  "
                f"[dim]|[/dim]  [bold green]✔ {successes} succeeded[/bold green]"
            )
        else:
            console.print(
                f"\n[bold green]✔ All {successes} job(s) completed successfully![/bold green]"
            )
        console.print()
    else:
        print(f"\n{'=' * 50}")
        print(f"  Summary: {successes} succeeded, {len(failures)} failed")
        print(f"{'=' * 50}")
        for r in sorted(results, key=lambda x: x["name"]):
            rc = r["returncode"]
            tag = "OK" if rc == 0 else "FAIL"
            print(f"  [{tag}] {r['name']} (host={r['host']}, exit={rc})")
        print()

    if failures:
        _print_failure_details(failures)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    started_at = datetime.now()
    t0 = time.monotonic()
    parser = argparse.ArgumentParser(description="Launch multiple rsync jobs in parallel.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to the YAML configuration file. If omitted, falls back to "
            "$XDG_CONFIG_HOME/parallel-rsync/config.yml (then config.yaml), "
            "with $XDG_CONFIG_HOME defaulting to ~/.config."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Maximum number of parallel rsync processes overall. "
            "Overrides the 'workers' config key (default: 4)."
        ),
    )
    parser.add_argument(
        "--max-per-host",
        type=int,
        default=None,
        help=(
            "Maximum concurrent rsync jobs per host. "
            "Overrides the 'max_per_host' config key (default: 2)."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional file path for logging output. If omitted, no log file is written.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each rsync process (default: no timeout).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Add '--dry-run' to every rsync command for testing.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the fancy progress bars (plain log output only).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    logger = setup_logging(args.log_level, args.log_file)

    try:
        config_path = resolve_config_path(args.config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        logger.error(str(exc))
        sys.exit(1)

    try:
        cfg = load_config(config_path)
    except Exception as exc:
        print(f"Error: failed to load config {config_path}: {exc}", file=sys.stderr)
        logger.error(f"Failed to load config {config_path}: {exc}")
        sys.exit(1)

    try:
        workers = resolve_setting(args.workers, cfg, "workers", 4)
        max_per_host = resolve_setting(args.max_per_host, cfg, "max_per_host", 2)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        logger.error(str(exc))
        sys.exit(1)

    logger.info("=== Parallel rsync started ===")
    logger.info(f"Config file: {config_path}")
    logger.info(f"Overall workers: {workers}")
    logger.info(f"Per-host concurrency limit: {max_per_host}")
    logger.info(f"Timeout: {args.timeout or 'none'}")
    logger.info(f"Dry-run mode: {'ON' if args.dry_run else 'OFF'}")

    groups = cfg["groups"]
    global_options: list[str] = cfg.get("global_options", [])

    if global_options:
        logger.info(f"Global rsync options: {shlex.join(global_options)}")

    # Inject dry-run
    if args.dry_run:
        if "--dry-run" not in global_options:
            global_options = list(global_options) + ["--dry-run"]
        patched = []
        for g in groups:
            g = dict(g)
            opts = list(g.get("options", []))
            if "--dry-run" not in opts:
                opts.append("--dry-run")
            g["options"] = opts
            patched.append(g)
        groups = patched

    # ------------------------------------------------------------------
    # Per-host semaphores
    # ------------------------------------------------------------------
    def _effective_host(g: dict) -> str:
        host = extract_host(g.get("src", ""))
        if host == "local":
            host = extract_host(g.get("dest", ""))
        return host

    hosts = {_effective_host(g) for g in groups}
    host_semaphores = {host: threading.Semaphore(max_per_host) for host in hosts}
    logger.info(f"Detected hosts: {', '.join(sorted(hosts))}")

    # ------------------------------------------------------------------
    # Build progress display
    # ------------------------------------------------------------------
    use_progress = _RICH_AVAILABLE and not args.no_progress

    progress = None
    task_ids: dict[str, "TaskID"] = {}

    if use_progress:
        progress = Progress(
            SpinnerColumn("dots"),
            TextColumn("{task.description}", markup=True),
            BarColumn(bar_width=30, complete_style="green", finished_style="bright_green"),
            TaskProgressColumn(),
            TextColumn("•", style="dim"),
            TextColumn("[cyan]{task.fields[speed]}[/cyan]", markup=True),
            TextColumn("•", style="dim"),
            TextColumn("[dim]ETA {task.fields[eta]}[/dim]", markup=True),
            TimeElapsedColumn(),
            expand=False,
            transient=False,
        )

        # Pre-create a task/bar for every group
        for g in groups:
            gname = g.get("name", "unnamed")
            tid = progress.add_task(
                description=f"{_STATUS['waiting']}  [bold]{gname}[/bold]",
                total=100,
                completed=0,
                speed="--",
                eta="--:--",
            )
            task_ids[gname] = tid

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    results: list[dict] = []

    def _run(g):
        gname = g.get("name", "unnamed")
        tid = task_ids.get(gname)
        return run_rsync_live(
            g, global_options, host_semaphores, logger, progress, tid, args.timeout
        )

    if use_progress:
        assert progress is not None
        with progress:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_name = {
                    executor.submit(_run, g): g.get("name", "unnamed") for g in groups
                }
                for future in as_completed(future_to_name):
                    results.append(future.result())
    else:
        # Fallback: no progress bars
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_name = {executor.submit(_run, g): g.get("name", "unnamed") for g in groups}
            for future in as_completed(future_to_name):
                result = future.result()
                results.append(result)
                name = result["name"]
                rc = result["returncode"]
                if rc != 0:
                    logger.error(f"[{name}] rsync exited with errors (code {rc})")
                else:
                    logger.info(f"[{name}] rsync completed successfully")

    _print_summary(results)

    duration = timedelta(seconds=round(time.monotonic() - t0))
    footer = f"Run started {started_at:%Y-%m-%d %H:%M:%S}, took {duration}"
    if _RICH_AVAILABLE:
        Console().print(f"[dim]{footer}[/dim]")
    else:
        print(footer)
    logger.info(footer)

    failures = [r for r in results if r["returncode"] != 0]
    logger.info(
        f"=== All rsync jobs finished: "
        f"{len(results) - len(failures)} succeeded, {len(failures)} failed ==="
    )
    if failures:
        for f in failures:
            logger.error(f"  FAILED: {f['name']} (code {f['returncode']})")
        sys.exit(1)


if __name__ == "__main__":
    main()
