"""Desktop window for the scanner, for people who would rather not use a
terminal.

    hayward-gui

Built on tkinter, which ships with Python, so the desktop app adds no
dependency and no packaging story.

Two things govern how this looks. It uses the platform's native ttk theme
rather than forcing one, because a forced theme is what makes a Python GUI
look like a Python GUI. And severity is carried by a small coloured marker
against otherwise plain text, because a table where every row is a different
colour is harder to read, not easier.

The scan runs on a worker thread and reports through a queue, since a large
directory takes long enough that a frozen window would look like a crash.
Nothing here loads or executes a model; it calls the same scanner the command
line does.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, messagebox, ttk

from hayward import __version__
from hayward.findings import Finding, Severity, is_coverage_gap
from hayward.report import SUFFIXES, render
from hayward.scanner import ModelFileScanner

# Muted rather than saturated. These sit next to body text all day.
_SEVERITY_COLOUR = {
    Severity.CRITICAL: "#c0392b",
    Severity.HIGH: "#d35400",
    Severity.MEDIUM: "#b7950b",
    Severity.LOW: "#2874a6",
    Severity.INFO: "#7b8794",
}

_INK = "#1f2933"
_MUTED = "#7b8794"
_HAIRLINE = "#dfe3e8"
_SURFACE = "#ffffff"

_EMPTY_TITLE = "No scan yet"
_EMPTY_BODY = (
    "Choose a model file or a folder to scan.\n"
    "Files are read as bytes. Nothing is loaded or executed."
)


def _common_root(paths: list[Path]) -> Path:
    if len(paths) == 1:
        return paths[0]
    try:
        return Path(os.path.commonpath(paths))
    except ValueError:
        return paths[0].parent


class HaywardApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.queue: queue.Queue = queue.Queue()
        self.scanning = False
        self.findings: list[Finding] = []
        self.visible: list[Finding] = []
        self.target: Path | None = None

        root.title("Hayward")
        root.geometry("1040x700")
        root.minsize(820, 520)
        root.configure(background=_SURFACE)

        self._init_style()
        self._build_header()
        self._build_body()
        self._build_footer()
        self._enable_drop()
        self._show_empty()

        root.after(80, self._drain_queue)

    # ── appearance ──────────────────────────────────────────────────

    def _init_style(self) -> None:
        style = ttk.Style()
        # Keep the native theme where there is one. "clam" is only a fallback
        # for bare X11 builds, where the alternatives look worse.
        if sys.platform == "darwin" and "aqua" in style.theme_names():
            style.theme_use("aqua")
        elif sys.platform.startswith("win") and "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")

        base = font.nametofont("TkDefaultFont")
        family = base.actual("family")
        size = base.actual("size")

        self.f_title = font.Font(family=family, size=size + 5, weight="bold")
        self.f_body = font.Font(family=family, size=size)
        self.f_small = font.Font(family=family, size=max(size - 1, 9))
        self.f_mono = font.Font(family="Menlo" if sys.platform == "darwin" else "Courier",
                                size=max(size - 1, 9))

        style.configure("Surface.TFrame", background=_SURFACE)
        style.configure("Hairline.TFrame", background=_HAIRLINE)
        style.configure("Title.TLabel", background=_SURFACE,
                        foreground=_INK, font=self.f_title)
        style.configure("Body.TLabel", background=_SURFACE,
                        foreground=_INK, font=self.f_body)
        style.configure("Muted.TLabel", background=_SURFACE,
                        foreground=_MUTED, font=self.f_small)
        style.configure("Count.TLabel", background=_SURFACE,
                        foreground=_INK, font=self.f_small)

        # Rows need air. The default rowheight is built for 1998.
        style.configure("Results.Treeview", background=_SURFACE,
                        fieldbackground=_SURFACE, foreground=_INK,
                        rowheight=30, borderwidth=0, font=self.f_body)
        style.configure("Results.Treeview.Heading", font=self.f_small)
        style.layout("Results.Treeview", style.layout("Treeview"))

    def _hairline(self, parent: tk.Misc) -> None:
        ttk.Frame(parent, style="Hairline.TFrame", height=1).pack(fill="x")

    # ── layout ──────────────────────────────────────────────────────

    def _build_header(self) -> None:
        bar = ttk.Frame(self.root, style="Surface.TFrame", padding=(24, 20, 24, 16))
        bar.pack(fill="x")

        left = ttk.Frame(bar, style="Surface.TFrame")
        left.pack(side="left")
        ttk.Label(left, text="Hayward", style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, text=f"Security scanner for model files  ·  {__version__}",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 0))

        right = ttk.Frame(bar, style="Surface.TFrame")
        right.pack(side="right")
        ttk.Button(right, text="Scan folder", command=self._pick_dir).pack(side="right")
        ttk.Button(right, text="Scan file", command=self._pick_file).pack(
            side="right", padx=(0, 8))
        self.export_button = ttk.Button(right, text="Export report",
                                        command=self._export, state="disabled")
        self.export_button.pack(side="right", padx=(0, 16))

        self._hairline(self.root)

    def _build_body(self) -> None:
        self.body = ttk.Frame(self.root, style="Surface.TFrame")
        self.body.pack(fill="both", expand=True)

        # Empty state and results share the same slot.
        self.empty = ttk.Frame(self.body, style="Surface.TFrame", padding=(24, 90))
        ttk.Label(self.empty, text=_EMPTY_TITLE, style="Body.TLabel").pack()
        ttk.Label(self.empty, text=_EMPTY_BODY, style="Muted.TLabel",
                  justify="center").pack(pady=(6, 0))

        self.results = ttk.Frame(self.body, style="Surface.TFrame")

        table = ttk.Frame(self.results, style="Surface.TFrame", padding=(16, 8, 16, 0))
        table.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(
            table, columns=("severity", "rule", "file"),
            show="headings", style="Results.Treeview", selectmode="browse",
        )
        self.tree.heading("severity", text="  SEVERITY", anchor="w")
        self.tree.heading("rule", text="RULE", anchor="w")
        self.tree.heading("file", text="FILE", anchor="w")
        self.tree.column("severity", width=130, minwidth=110, stretch=False, anchor="w")
        self.tree.column("rule", width=170, minwidth=150, stretch=False, anchor="w")
        self.tree.column("file", minwidth=240, anchor="w")

        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        for severity, colour in _SEVERITY_COLOUR.items():
            self.tree.tag_configure(severity.value, foreground=colour)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Detail pane. Fixed height so the table does not jump as selection
        # moves between a one-line finding and a long one.
        self._hairline(self.results)
        detail = ttk.Frame(self.results, style="Surface.TFrame",
                           padding=(24, 14, 24, 16), height=132)
        detail.pack(fill="x")
        detail.pack_propagate(False)

        self.detail_rule = tk.StringVar(value="")
        self.detail_text = tk.StringVar(value="")
        ttk.Label(detail, textvariable=self.detail_rule,
                  style="Muted.TLabel", font=self.f_mono).pack(anchor="w")
        ttk.Label(detail, textvariable=self.detail_text, style="Body.TLabel",
                  wraplength=940, justify="left").pack(anchor="w", pady=(6, 0))

    def _build_footer(self) -> None:
        self._hairline(self.root)
        bar = ttk.Frame(self.root, style="Surface.TFrame", padding=(24, 12))
        bar.pack(fill="x")

        self.summary = tk.StringVar(value="Ready")
        ttk.Label(bar, textvariable=self.summary, style="Count.TLabel").pack(side="left")

        self.show_info = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Show unknowns", variable=self.show_info,
                        command=self._refill).pack(side="right")

    def _enable_drop(self) -> None:
        """Register for file drops when tkdnd is present.

        tkdnd is an optional Tk extension, absent from a stock install, so the
        buttons remain the guaranteed way in.
        """
        try:
            self.root.tk.call("package", "require", "tkdnd")
            self.root.tk.call("tkdnd::drop_target", "register", self.root, "DND_Files")
            self.root.bind("<<Drop>>", self._on_drop)
        except tk.TclError:
            pass

    def _show_empty(self) -> None:
        self.results.pack_forget()
        self.empty.pack(fill="both", expand=True)

    def _show_results(self) -> None:
        self.empty.pack_forget()
        self.results.pack(fill="both", expand=True)

    # ── actions ─────────────────────────────────────────────────────

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a model file",
            filetypes=[
                ("Model files", "*.pt *.pth *.pkl *.pickle *.ckpt *.th *.bin "
                                "*.safetensors *.gguf *.h5 *.hdf5 *.keras *.onnx "
                                "*.pb *.npy *.npz *.joblib *.tflite *.skops "
                                "*.pmml *.mar *.nemo *.7z"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._start([Path(path)])

    def _pick_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose a folder of models")
        if path:
            self._start([Path(path)])

    def _on_drop(self, event: tk.Event) -> None:
        raw = self.root.tk.splitlist(event.data)  # type: ignore[attr-defined]
        if raw:
            self._start([Path(r) for r in raw])

    def _start(self, targets: list[Path]) -> None:
        if self.scanning:
            return
        self.scanning = True
        self.target = _common_root(targets)
        self.findings = []
        self.tree.delete(*self.tree.get_children())
        self.detail_rule.set("")
        self.detail_text.set("")
        label = self.target.name if len(targets) == 1 else f"{len(targets)} files"
        self.summary.set(f"Scanning {label} ...")
        self._show_results()
        threading.Thread(target=self._worker, args=(targets,), daemon=True).start()

    def _worker(self, targets: list[Path]) -> None:
        scanner = ModelFileScanner()
        findings: list[Finding] = []
        try:
            for target in targets:
                if target.is_file():
                    findings.extend(scanner.scan_file(target))
                else:
                    findings.extend(scanner.scan_directory(target))
        except OSError as exc:
            self.queue.put(("error", str(exc)))
            return
        except Exception as exc:
            self.queue.put(("error", f"scan crashed: {exc}"))
            return
        self.queue.put(("done", findings))

    # ── results ─────────────────────────────────────────────────────

    def _drain_queue(self) -> None:
        try:
            kind, payload = self.queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self.scanning = False
            if kind == "error":
                self.summary.set(f"Could not scan: {payload}")
            else:
                self.findings = payload
                self._refill()
        self.root.after(80, self._drain_queue)

    def _refill(self) -> None:
        self.tree.delete(*self.tree.get_children())

        self.visible = sorted(
            (f for f in self.findings
             if self.show_info.get() or f.severity is not Severity.INFO),
            key=lambda f: (f.severity_order, f.file_path),
        )
        for index, finding in enumerate(self.visible):
            self.tree.insert(
                "", "end", iid=str(index),
                values=(
                    f"  ●  {finding.severity.value.upper()}",
                    finding.rule_id,
                    self._display_path(finding),
                ),
                tags=(finding.severity.value,),
            )

        self._update_summary()
        self.export_button.configure(
            state="normal" if self.target is not None else "disabled")
        if self.visible:
            self.tree.selection_set("0")
            self.tree.focus("0")
        else:
            self.detail_rule.set("")
            self.detail_text.set(
                "Nothing to show at this filter."
                if self.findings else "No findings."
            )

    def _display_path(self, finding: Finding) -> str:
        path = Path(finding.file_path)
        if self.target:
            root = self.target if self.target.is_dir() else self.target.parent
            try:
                return str(path.relative_to(root))
            except ValueError:
                pass
        return str(path)

    def _update_summary(self) -> None:
        if not self.findings:
            self.summary.set(f"No findings in {self.target.name if self.target else ''}")
            return
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.severity.value] = counts.get(finding.severity.value, 0) + 1
        order = ("critical", "high", "medium", "low", "info")
        parts = [f"{counts[s]} {s}" for s in order if s in counts]
        gaps = sum(1 for f in self.findings if is_coverage_gap(f))
        line = "   ".join(parts)
        if gaps:
            line += f"      {gaps} file(s) not fully read"
        self.summary.set(line)

    def _export(self) -> None:
        if self.target is None:
            return
        default = f"hayward-{self.target.name or 'scan'}"
        path = filedialog.asksaveasfilename(
            title="Export report",
            initialfile=f"{default}.html",
            defaultextension=".html",
            filetypes=[
                ("HTML report", "*.html"),
                ("Markdown", "*.md"),
                ("JSON", "*.json"),
            ],
        )
        if not path:
            return

        suffix = Path(path).suffix.lower()
        fmt = next((k for k, v in SUFFIXES.items() if v == suffix), "html")
        root = self.target if self.target.is_dir() else self.target.parent
        try:
            Path(path).write_text(
                render(fmt, self.findings, root, __version__), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.summary.set(f"Report written to {Path(path).name}")

    def _on_select(self, _event: tk.Event) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        finding = self.visible[int(selection[0])]
        cwe = "   " + ", ".join(f"CWE-{c}" for c in finding.cwe_ids) if finding.cwe_ids else ""
        self.detail_rule.set(f"{finding.rule_id}{cwe}")
        self.detail_text.set(finding.message)


def main() -> int:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(
            f"hayward-gui: could not open a window ({exc}). If there is no "
            "display available, use the command line instead: hayward scan <target>",
            file=sys.stderr,
        )
        return 1
    HaywardApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
