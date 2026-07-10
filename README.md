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

A group can drop specific global options with `exclude_options`: an entry matches a global option either exactly or by its name before the `=` (so `--rsync-path` drops `--rsync-path=sudo rsync`). Exclusions only apply to global options, never to the group's own `options`.

Note: exclusion matches single list entries only. Write global options that take a value in `--opt=value` form — if the value is a separate list entry (`["--rsync-path", "sudo rsync"]`), excluding `--rsync-path` would leave the orphaned value behind, which rsync would treat as an extra source path.

## Requirements

- PyYAML (`pip install pyyaml`)
- Optional for colorized console logs: Rich (`pip install rich`)

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

This will create a `parallel-rsync` binary in the `dist` directory that can be used on any platform.

## License

MIT
