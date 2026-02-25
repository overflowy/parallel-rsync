#!/usr/bin/env python3

import argparse
import importlib
import logging
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


def get_rich_handler_class():
    """Load rich's console log handler if installed."""
    try:
        rich_logging = importlib.import_module("rich.logging")
    except ImportError:
        return None
    return getattr(rich_logging, "RichHandler", None)


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
    """
    Create the local destination directory if it doesn't exist.
    Only applies to local destinations (no ':' in dest).
    For remote destinations this is a no-op — use --mkpath on the remote side.
    """
    if ":" in dest:
        logger.debug(f"[{name}] Remote destination, skipping local mkdir")
        return
    dest_path = Path(dest)
    if not dest_path.exists():
        logger.info(f"[{name}] Creating destination directory: {dest_path}")
        dest_path.mkdir(parents=True, exist_ok=True)


def build_rsync_cmd(group: dict, global_options: list[str] | None = None) -> list[str]:
    """Construct the rsync command list for a given group.

    Global options are prepended, then per-group options follow so that
    per-group flags can override globals when rsync uses last-wins semantics.
    """
    src = group.get("src")
    dest = group.get("dest")
    group_options = group.get("options", [])
    if not src or not dest:
        raise ValueError(
            f"Group '{group.get('name', '<unnamed>')}' missing src or dest."
        )
    # Ensure src ends with a slash for directory sync semantics
    if not src.endswith("/"):
        src = src + "/"
    # Merge: global first, then per-group (last-wins for rsync flags)
    merged_options = list(global_options or []) + list(group_options)
    cmd = ["rsync"] + merged_options + [src, dest]
    return cmd


def extract_host(dest: str) -> str:
    """
    Extract the hostname from an rsync destination string.

    Handles:
      - Remote shell: user@host:/path  or  host:/path
      - Rsync daemon: user@host::module  or  host::module
      - Local: /path/to/dir

    Returns 'local' for non‑remote destinations.
    """
    # Rsync daemon syntax (double colon)
    if "::" in dest:
        host_part = dest.split("::", 1)[0]
        if "@" in host_part:
            return host_part.split("@", 1)[1]
        return host_part

    # Remote shell syntax (single colon, but skip Windows drive letters like C:\)
    if ":" in dest:
        colon_idx = dest.index(":")
        # A single letter before the colon is likely a Windows drive letter
        if colon_idx == 1 and dest[0].isalpha():
            return "local"
        host_part = dest.split(":", 1)[0]
        if "@" in host_part:
            return host_part.split("@", 1)[1]
        return host_part

    return "local"


def setup_logging(log_file: str, log_level: str) -> logging.Logger:
    """Configure a logger that writes to both a file and stdout."""
    logger = logging.getLogger("parallel_rsync")
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Prevent duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    rich_handler_class = get_rich_handler_class()

    # Console handler (colorized when rich is installed)
    if rich_handler_class is not None:
        ch = rich_handler_class(show_path=False, markup=False)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(message)s"))
    else:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def run_rsync(
    group: dict,
    global_options: list[str],
    semaphores: dict[str, threading.Semaphore],
    logger: logging.Logger,
    timeout: int | None = None,
) -> dict:
    """
    Execute rsync for a single group while respecting the per-host semaphore.
    Returns a dictionary with execution details.
    """
    name = group.get("name", "unnamed")
    src = group.get("src", "")
    dest = group.get("dest", "")
    # Throttle on whichever side is remote (prefer src, fall back to dest)
    host = extract_host(src)
    if host == "local":
        host = extract_host(dest)
    semaphore = semaphores[host]

    logger.info(f"[{name}] Waiting for slot on host '{host}'")
    with semaphore:
        try:
            if group.get("mkdir_dest", False):
                ensure_dest_dir(dest, logger, name)
            cmd = build_rsync_cmd(group, global_options)
            cmd_str = shlex.join(cmd)
            logger.info(f"[{name}] Starting rsync on host '{host}': {cmd_str}")

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout,
            )
            logger.info(f"[{name}] rsync completed with exit code {result.returncode}")

            if result.stdout:
                logger.info(f"[{name}] STDOUT:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"[{name}] STDERR:\n{result.stderr}")

            return {
                "name": name,
                "host": host,
                "cmd": cmd_str,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            logger.error(f"[{name}] rsync timed out after {timeout}s on host '{host}'")
            return {
                "name": name,
                "host": host,
                "cmd": shlex.join(build_rsync_cmd(group, global_options)),
                "returncode": -2,
                "stdout": "",
                "stderr": f"Timed out after {timeout}s",
            }
        except Exception as e:
            logger.error(f"[{name}] Exception while running rsync: {e}")
            return {
                "name": name,
                "host": host,
                "cmd": shlex.join(build_rsync_cmd(group, global_options)),
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
            }


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch multiple rsync jobs in parallel with logging and per‑host concurrency limits."
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
        default="parallel_rsync.log",
        help="File path for logging output (default: parallel_rsync.log).",
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
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------
    logger = setup_logging(args.log_file, args.log_level)
    if get_rich_handler_class() is None:
        logger.warning("rich is not installed; console logs will not be colorized")
    logger.info("=== Parallel rsync launcher started ===")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Overall workers: {args.workers}")
    logger.info(f"Per-host concurrency limit: {args.max_per_host}")
    logger.info(f"Timeout: {args.timeout or 'none'}")
    logger.info(f"Dry-run mode: {'ON' if args.dry_run else 'OFF'}")

    # ------------------------------------------------------------------
    # Load configuration
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

    # Apply dry‑run flag globally if requested (on copies to avoid mutating config)
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
    # Prepare per‑host semaphores
    # ------------------------------------------------------------------
    def _effective_host(g: dict) -> str:
        """Pick the remote side for throttling (prefer src, fall back to dest)."""
        host = extract_host(g.get("src", ""))
        if host == "local":
            host = extract_host(g.get("dest", ""))
        return host

    hosts = {_effective_host(g) for g in groups}
    host_semaphores = {host: threading.Semaphore(args.max_per_host) for host in hosts}
    logger.info(f"Detected hosts: {', '.join(sorted(hosts))}")

    # ------------------------------------------------------------------
    # Execute rsync jobs in parallel
    # ------------------------------------------------------------------
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_name = {
            executor.submit(
                run_rsync, g, global_options, host_semaphores, logger, args.timeout
            ): g.get("name", "unnamed")
            for g in groups
        }
        for future in as_completed(future_to_name):
            result = future.result()
            results.append(result)
            name = result["name"]
            rc = result["returncode"]
            if rc != 0:
                logger.error(f"[{name}] rsync exited with errors (code {rc})")
            else:
                logger.info(f"[{name}] rsync completed successfully")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    failures = [r for r in results if r["returncode"] != 0]
    logger.info(
        f"=== All rsync jobs have finished: "
        f"{len(results) - len(failures)} succeeded, {len(failures)} failed ==="
    )
    if failures:
        for f in failures:
            logger.error(f"  FAILED: {f['name']} (code {f['returncode']})")
        sys.exit(1)


if __name__ == "__main__":
    main()
