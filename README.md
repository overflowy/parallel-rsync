# parallel-rsync

Parallel rsync launcher with fancy progress bars.

https://github.com/user-attachments/assets/fd146967-0946-4ee1-92a0-9f49864c7fc8

## Usage

```
parallel_rsync.py [-h] -c CONFIG [--workers WORKERS] [--max-per-host MAX_PER_HOST] [--log-file LOG_FILE]
                  [--log-level {DEBUG,INFO,WARNING,ERROR}] [--timeout TIMEOUT] [--dry-run] [--no-progress]

Launch multiple rsync jobs in parallel.

options:
  -h, --help            show this help message and exit
  -c, --config CONFIG   Path to the YAML configuration file.
  --workers WORKERS     Maximum number of parallel rsync processes overall (default: 4).
  --max-per-host MAX_PER_HOST
                        Maximum concurrent rsync jobs per host (default: 2).
  --log-file LOG_FILE   Optional file path for logging output. If omitted, no log file is written.
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Logging verbosity (default: INFO).
  --timeout TIMEOUT     Timeout in seconds for each rsync process (default: no timeout).
  --dry-run             Add '--dry-run' to every rsync command for testing.
  --no-progress         Disable the fancy progress bars (plain log output only).
```

The config file should have the following structure:

```yaml
global_options:
  - "-avz"
  - "--delete"

groups:
  - name: "web-assets"
    src: "/var/www/assets/"
    dest: "deploy@web01.example.com:/srv/www/assets/"
    options:
      - "--exclude=*.tmp"
  - name: "web-logs"
    src: "/var/log/nginx/"
    dest: "deploy@web01.example.com:/srv/www/logs/"
    options:
      - "--progress"
```

Global options are prepended to each group's options. Per-group options can override or extend the global ones.

## Requirements

- PyYAML (`pip install pyyaml`)
- Optional for colorized console logs: Rich (`pip install rich`)

## Build (universal binary)

```sh
task build

# or

uvx cosmofy bundle
```

This will create a `parallel-rsync` binary in the `dist` directory that can be used on any platform.

## License

MIT
