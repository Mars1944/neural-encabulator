from __future__ import annotations

"""
Console helper for consistent, configurable terminal output.

User Guide
----------

Overview
- Provides a small Console class with leveled methods: info, warn, error, success, debug.
- Centralizes format/behavior across scripts (infer, train, viewer, etc.).
- Optional print hook to route built-in print() through the same formatter.

Quick Start
- Import and construct from your runtime config (dict):
    from console import console_from_config
    c = console_from_config(config)
    c.info("Hello")

Config Keys (top-level or under "inference" or "logging")
- log_color: bool      -> enable/disable ANSI color (default True; auto-disables if colorama missing)
- log_time: bool       -> prefix each line with HH:MM:SS timestamp
- log_level: str       -> "info" or "debug" (debug shows extra lines)
- log_hook_print: bool -> if True, replaces builtins.print with a formatted logger

Typical JSON snippet
{
  "inference": {
    "log_color": true,
    "log_time": false,
    "log_level": "info",
    "log_hook_print": false
  }
}

Notes
- If log_hook_print is enabled, most stray print() calls inherit console formatting automatically.
- No external dependency is required; color is best-effort via colorama when available.
"""

import sys
import builtins
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


try:
    import colorama  # type: ignore

    colorama.init()
    _HAS_COLOR = True
except Exception:  # pragma: no cover
    _HAS_COLOR = False


@dataclass
class ConsoleConfig:
    """Runtime options for the console.

    - use_color: enable color when colorama is available
    - timestamp: prefix each line with HH:MM:SS
    - level:     "info" or "debug"; debug prints extra lines
    - hook_print: replace builtins.print so legacy prints go through the console
    """

    use_color: bool = True
    timestamp: bool = False
    level: str = "info"  # info|debug
    hook_print: bool = False


class Console:
    """Lightweight console with consistent formatting and levels.

    Construct with ConsoleConfig, then call .info/.warn/.error/.success/.debug.
    """
    def __init__(self, cfg: Optional[ConsoleConfig] = None, stream=None) -> None:
        self.cfg = cfg or ConsoleConfig()
        self.stream = stream or sys.stdout
        self._orig_print = None  # type: ignore
        self._file = None  # type: ignore

    # Formatting helpers
    def _fmt(self, tag: str, msg: str, color: Optional[str]) -> str:
        """Build the final line with optional timestamp and color."""
        parts = []
        if self.cfg.timestamp:
            parts.append(datetime.now().strftime("%H:%M:%S"))
        parts.append(tag)
        prefix = " ".join(parts)
        text = f"{prefix} | {msg}"
        if self.cfg.use_color and _HAS_COLOR and color:
            return f"{color}{text}{colorama.Style.RESET_ALL}"
        return text

    def _writeln(self, s: str) -> None:
        """Write a line to the configured stream with flush; fallback to print on error."""
        try:
            self.stream.write(s + "\n")
            self.stream.flush()
        except Exception:
            # Fallback to builtins.print if stream write fails
            builtins.print(s)
        # Also write to log file if configured
        try:
            if self._file is not None:
                self._file.write(s + "\n")
                self._file.flush()
        except Exception:
            pass

    # Public logging methods
    def info(self, msg: str) -> None:
        self._writeln(self._fmt("[info]", msg, getattr(sys.modules.get('colorama', None), 'Fore', None).CYAN if _HAS_COLOR else None))

    def success(self, msg: str) -> None:
        self._writeln(self._fmt("[ok]", msg, getattr(sys.modules.get('colorama', None), 'Fore', None).GREEN if _HAS_COLOR else None))

    def warn(self, msg: str) -> None:
        self._writeln(self._fmt("[warn]", msg, getattr(sys.modules.get('colorama', None), 'Fore', None).YELLOW if _HAS_COLOR else None))

    def error(self, msg: str) -> None:
        self._writeln(self._fmt("[err]", msg, getattr(sys.modules.get('colorama', None), 'Fore', None).RED if _HAS_COLOR else None))

    def debug(self, msg: str) -> None:
        if self.cfg.level.lower() == "debug":
            self._writeln(self._fmt("[dbg]", msg, getattr(sys.modules.get('colorama', None), 'Fore', None).MAGENTA if _HAS_COLOR else None))

    def header(self, msg: str) -> None:
        self._writeln(self._fmt("[==]", msg, getattr(sys.modules.get('colorama', None), 'Style', None).BRIGHT if _HAS_COLOR else None))

    def line(self, msg: str = "") -> None:
        self._writeln(msg)

    # Tables
    def table(
        self,
        headers: "list[str]",
        rows: "list[list[str]]",
        align: "list[str] | None" = None,
        title: "str | None" = None,
        sep: str = " | ",
    ) -> None:
        """Pretty-print a simple table with dynamic column widths.

        - headers: list of column header strings
        - rows: list of row lists (strings); len of each row should match headers
        - align: optional list of alignment flags per column: 'l' (left) or 'r' (right)
        - title: optional title printed as a header before the table
        - sep: column separator (default ' | ')
        """
        try:
            if not headers:
                return
            ncol = len(headers)
            align = align if isinstance(align, list) and len(align) == ncol else ["l"] * ncol
            # Normalize rows to strings and correct column count
            norm_rows: list[list[str]] = []
            for r in rows:
                if not isinstance(r, (list, tuple)):
                    continue
                rr = [str(x) for x in r[:ncol]]
                if len(rr) < ncol:
                    rr += [""] * (ncol - len(rr))
                norm_rows.append(rr)

            # Compute column widths
            col_w = [0] * ncol
            for j in range(ncol):
                col_w[j] = max(len(str(headers[j])), max((len(rr[j]) for rr in norm_rows), default  = 0))

            def fmt_row(cols: list[str]) -> str:
                parts: list[str] = []
                for j, col in enumerate(cols):
                    if align[j] == "r":
                        parts.append(str(col).rjust(col_w[j]))
                    else:
                        parts.append(str(col).ljust(col_w[j]))
                return sep.join(parts)

            # Writer that forces magenta color for table content when color is enabled
            def _write_magenta(s: str) -> None:
                try:
                    if self.cfg.use_color and _HAS_COLOR:
                        Fore = getattr(sys.modules.get('colorama', None), 'Fore', None)
                        Style = getattr(sys.modules.get('colorama', None), 'Style', None)
                        white = getattr(Fore, 'MAGENTA', '') if Fore is not None else ''
                        reset = getattr(Style, 'RESET_ALL', '') if Style is not None else ''
                        self._writeln(f"{white}{s}{reset}")
                    else:
                        self._writeln(s)
                except Exception:
                    self._writeln(s)

            if title:
                self.header(title)
            _write_magenta(fmt_row(headers))
            _write_magenta("-" * (sum(col_w) + len(sep) * (ncol - 1)))
            for rr in norm_rows:
                _write_magenta(fmt_row(rr))
        except Exception:
            # Fallback: try to print minimally if any error
            try:
                if title:
                    self.header(title)
                self.line(sep.join(headers))
                for r in rows:
                    self.line(sep.join([str(x) for x in r]))
            except Exception:
                pass

    # Print hook
    def install_print_hook(self) -> None:
        """Replace builtins.print with a hook that routes to this console.

        This keeps existing print calls consistent with the console format
        without touching every call site.
        """
        if self._orig_print is not None:
            return
        self._orig_print = builtins.print  # type: ignore

        def _hook(*args: Any, **kwargs: Any) -> None:
            """Hook that formats print() messages using the console style."""
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            text = sep.join(str(a) for a in args)
            # Route to info() for consistency; preserve 'end' behavior
            line = self._fmt("[log]", text, getattr(sys.modules.get('colorama', None), 'Fore', None).WHITE if _HAS_COLOR else None)
            if end == "\n":
                self._writeln(line)
            else:
                try:
                    self.stream.write(line + end)
                    self.stream.flush()
                except Exception:
                    self._orig_print(line, end=end)  # type: ignore

        builtins.print = _hook  # type: ignore

    def uninstall_print_hook(self) -> None:
        """Restore the original builtins.print if previously hooked."""
        if self._orig_print is not None:
            builtins.print = self._orig_print  # type: ignore
            self._orig_print = None

    # File logging
    def set_log_file(self, path: "str | builtins.object") -> None:
        """Open or switch the log file path used for tee'ing output."""
        try:
            from pathlib import Path as _Path
            p = _Path(str(path))
            p.parent.mkdir(parents=True, exist_ok=True)
            # Close previous
            try:
                if self._file is not None:
                    self._file.close()
            except Exception:
                pass
            self._file = p.open("a", encoding="utf-8")
        except Exception:
            self._file = None


_global_console: Optional[Console] = None


def console_from_config(config: Optional[dict[str, Any]] = None) -> Console:
    """Create (or reuse) a global Console configured from a config dict.

    Reads keys from:
      - top-level of config
      - config["inference"] (if present)
      - config["logging"] (if present)
    """
    global _global_console
    if _global_console is not None:
        return _global_console
    cfg = ConsoleConfig()
    if isinstance(config, dict):
        log_cfg = config.get("logging", {}) if isinstance(config.get("logging", {}), dict) else {}
        # Also allow keys at top-level or under inference
        for scope in (config, config.get("inference", {}), log_cfg):
            if not isinstance(scope, dict):
                continue
            cfg.use_color = bool(scope.get("log_color", cfg.use_color))
            cfg.timestamp = bool(scope.get("log_time", cfg.timestamp))
            lvl = scope.get("log_level", cfg.level)
            if isinstance(lvl, str):
                cfg.level = lvl
            cfg.hook_print = bool(scope.get("log_hook_print", cfg.hook_print))
    _global_console = Console(cfg)
    if cfg.hook_print:
        _global_console.install_print_hook()
    # Optional file logging
    try:
        # Accept keys from any scope
        scopes = [config or {}, (config or {}).get("inference", {}), (config or {}).get("logging", {})]
        log_to_file = False
        log_file = None
        cfg_dir = (config or {}).get("_config_dir", None)
        out_root = (config or {}).get("outputs_root", None)
        for sc in scopes:
            if isinstance(sc, dict):
                if "log_to_file" in sc:
                    log_to_file = bool(sc.get("log_to_file", log_to_file))
                if "log_file" in sc and not log_file:
                    log_file = sc.get("log_file")
        if log_to_file:
            from pathlib import Path as _Path
            from datetime import datetime as _dt
            if isinstance(log_file, str) and log_file.strip():
                p = _Path(log_file)
                if not p.is_absolute() and isinstance(cfg_dir, str) and cfg_dir:
                    p = (_Path(cfg_dir) / p).resolve()
            else:
                base = None
                if isinstance(out_root, str) and out_root.strip():
                    pr = _Path(out_root)
                    if not pr.is_absolute() and isinstance(cfg_dir, str) and cfg_dir:
                        pr = (_Path(cfg_dir) / pr).resolve()
                    base = pr
                else:
                    base = _Path(cfg_dir) if isinstance(cfg_dir, str) and cfg_dir else _Path(".")
                ts = _dt.now().strftime("%Y%m%d-%H%M%S")
                p = (base / "logs" / f"run-{ts}.log").resolve()
            _global_console.set_log_file(str(p))
    except Exception:
        pass
    return _global_console


def get_console() -> Console:
    """Return the global console if initialized; otherwise a default Console."""
    return _global_console or Console()
