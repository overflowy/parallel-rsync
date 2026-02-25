# parallel-rsync

Parallel rsync launcher with logging and per-host concurrency limits.

<img width="2450" height="1142" alt="CleanShot 2026-02-25 at 18 57 37@2x" src="https://github.com/user-attachments/assets/042d47de-35fe-4bdd-8c4b-72f9b8859058" />


## Usage

```
parallel_rsync.py -c config.yaml [--workers N] [--max-per-host M]
                  [--log-file FILE] [--log-level LEVEL]
                  [--timeout SECS] [--dry-run]
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

## License

MIT
