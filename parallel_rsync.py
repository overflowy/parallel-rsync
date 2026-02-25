#!/usr/bin/env python3

import argparse
import logging
import re
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def build_rsync_cmd(group: dict, global_options: list[str] | None = None) -> list[str]:
    """Construct the rsync command list for a given group."""
    src = group.get("src")
    dest = group.get("dest")
    group_options = group.get("options", [])
    if not src or not dest:
        raise ValueError(f"Group '{group.get('name', '<unnamed>')}' missing src or dest.")
    if not src.endswith("/"):
        src = src + "/"
    merged_options = list(global_options or []) + list(group_options)
    cmd = ["rsync"] + merged_options + [src, dest]
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch multiple rsync jobs in parallel with fancy progress bars."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum number of parallel rsync processes overall (default: 4).",
    )
    parser.add_argument(
        "--max-per-host",
        type=int,
        default=2,
        help="Maximum concurrent rsync jobs per host (default: 2).",
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
    logger.info("=== Parallel rsync started ===")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Overall workers: {args.workers}")
    logger.info(f"Per-host concurrency limit: {args.max_per_host}")
    logger.info(f"Timeout: {args.timeout or 'none'}")
    logger.info(f"Dry-run mode: {'ON' if args.dry_run else 'OFF'}")

    # ------------------------------------------------------------------
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        logger.error(f"Failed to load config: {exc}")
        sys.exit(1)

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
    host_semaphores = {host: threading.Semaphore(args.max_per_host) for host in hosts}
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
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_name = {
                    executor.submit(_run, g): g.get("name", "unnamed") for g in groups
                }
                for future in as_completed(future_to_name):
                    results.append(future.result())
    else:
        # Fallback: no progress bars
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
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
