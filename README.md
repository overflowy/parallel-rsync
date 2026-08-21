# parallel-rsync

Parallel rsync launcher with fancy progress bars.

https://github.com/user-attachments/assets/fd146967-0946-4ee1-92a0-9f49864c7fc8

## Usage

```
parallel_rsync.py [-h] [-c CONFIG] [--workers WORKERS] [--max-per-host MAX_PER_HOST] [--log-file LOG_FILE]
                  [--log-level {DEBUG,INFO,WARNING,ERROR}] [--timeout TIMEOUT] [--dry-run] [--no-progress]

Launch multiple rsync jobs in parallel.

options:
  -h, --help            show this help message and exit
  -c, --config CONFIG   Path to the YAML configuration file. If omitted, falls back to
                        $XDG_CONFIG_HOME/parallel-rsync/config.yml (then config.yaml), with
                        $XDG_CONFIG_HOME defaulting to ~/.config.
  --workers WORKERS     Maximum number of parallel rsync processes overall. Overrides the
                        'workers' config key (default: 4).
  --max-per-host MAX_PER_HOST
                        Maximum concurrent rsync jobs per host. Overrides the 'max_per_host'
                        config key (default: 2).
  --log-file LOG_FILE   Optional file path for logging output. If omitted, no log file is written.
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Logging verbosity (default: INFO).
  --timeout TIMEOUT     Timeout in seconds for each rsync process (default: no timeout).
  --dry-run             Add '--dry-run' to every rsync command for testing.
  --no-progress         Disable the fancy progress bars (plain log output only).
```

The config file should have the following structure:

```yaml
workers: 8        # optional, overridden by --workers (default: 4)
max_per_host: 4   # optional, overridden by --max-per-host (default: 2)

global_options:
  - "-avz"
  - "--delete"
  - "--rsync-path=sudo rsync"

groups:
  - name: "web-assets"
    src: "/var/www/assets/"
    dest: "deploy@web01.example.com:/srv/www/assets/"
    options:
      - "--exclude=*.tmp"
  - name: "web-logs"
    src: "/var/log/nginx/"
    dest: "deploy@web01.example.com:/srv/www/logs/"
    exclude_options:
      - "--rsync-path"
    options:
      - "--progress"
```

Global options are prepended to each group's options. Per-group options can override or extend the global ones.

A group can drop specific global options with `exclude_options`. An entry matches a global option either exactly or by its name before the `=`, so `--rsync-path` drops `--rsync-path=sudo rsync`. Exclusions only apply to global options, never to the group's own `options`.

Exclusion matches single list entries only, so write global options that take a value in `--opt=value` form. If the value is a separate list entry (`["--rsync-path", "sudo rsync"]`), excluding `--rsync-path` would leave the orphaned value behind, and rsync would treat it as an extra source path.

### Collision check

All jobs run at once. Two jobs writing the same tree race each other, and with `--delete` one wipes what the other just wrote. So before any job starts, the run aborts if groups collide. That covers duplicate group names, two `dest` paths that are identical or nested, a `src` that overlaps another group's `dest`, and a group whose own `src` and `dest` overlap.

The check reads the options too. `--remove-source-files` turns a group's `src` into a write target. `--backup-dir`, `--partial-dir`, and `--temp-dir` add write targets on the dest side, and `--link-dest`, `--compare-dest`, and `--copy-dest` add read-only ones. Both `--opt=value` and the separate-argument spelling work. When a singular option appears twice, the last one wins, matching rsync, so a per-group value overrides a global one. `--port` tells daemon endpoints on different ports apart. Groups that only read the same tree pass.

Comparison is textual. Local paths go through `realpath` first, so symlinks and spellings like `//Volumes/X` or `a/../b` land on the same tree. Remote paths compare per host. `host:backup`, `host:./backup`, and `host:~/backup` count as equal, `alice@host:backup` and `bob@host:backup` do not, and daemon endpoints (`host::module`, `rsync://`) never mix with ssh paths. Two things it cannot see: that an ssh alias and the real hostname (or `localhost:` and a plain local path) reach the same machine, and that `/Volumes/Backup` and `/Volumes/backup` meet on a case-insensitive filesystem. It guards against config mistakes. It does not replace reading your config.

### Mount check

The launcher also checks local `src`/`dest` paths under known removable-media trees: `/Volumes/<name>` (macOS), `/mnt/<name>`, `/media/<user>/<label>`, and `/run/media/<user>/<label>` (Linux). If the implied mount point is not actually mounted, the run aborts. A stale directory at a mount point would otherwise let rsync back up onto the system drive (via `--mkpath`) or mirror an empty source over a good backup (via `--delete`).

## Requirements

- PyYAML (`pip install pyyaml`)
- Rich (`pip install rich`), optional. Without it you get plain log output instead of the progress bars and summary table.

## Install

```sh
task install

# or

uv tool install .
```

This installs a `parallel-rsync` executable on your PATH. Uninstall with `task uninstall`.

Alternatively, run it without installing:

```sh
uvx --from . parallel-rsync
```

## Build (universal binary)

```sh
task build

# or

uvx cosmofy bundle
```

This creates a `parallel-rsync` binary in `dist` that runs on any platform.

## License

MIT
