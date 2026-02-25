# parallel-rsync

Parallel rsync launcher with logging and per-host concurrency limits.

```
Usage:
    parallel_rsync.py -c config.yaml [--workers N] [--max-per-host M]
                      [--log-file FILE] [--log-level LEVEL]
                      [--timeout SECS] [--dry-run]
```

The YAML file should have the following structure:

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

Requires: PyYAML (pip install pyyaml).
