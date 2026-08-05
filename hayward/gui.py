"""Desktop window for the security scanner, for people who would rather not
use a terminal.

    hayward-gui

Built on tkinter, which ships with Python, so the GUI adds no dependency and
no packaging story. Drop a file or a folder on the window, or use the button.

The scan runs on a worker thread and reports back through a queue, because a
large directory takes long enough that a frozen window would look like a
crash. Nothing here loads or executes a model; it calls the same scanner the
CLI does.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from hayward import __version__
from hayward.findings import Finding, Severity, is_coverage_gap
from hayward.scanner import ModelFileScanner

_SEVERITY_COLOUR = {
    Severity.CRITICAL: "#b3261e",
    Severity.HIGH: "#c2410c",
    Severity.MEDIUM: "#a16207",
    Severity.LOW: "#1d4ed8",
    Severity.INFO: "#6b7280",
}

_INTRO = (
    "Drop a model file or a folder here, or use Choose.\n\n"
    "Nothing is loaded or executed. Files are read as bytes."
)


class HaywardApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.queue: queue.Queue = queue.Queue()
        self.scanning = False

        root.title(f"Hayward {__version__}")
        root.geometry("980x620")
        root.minsize(760, 460)

        self._build_toolbar()
        self._build_table()
        self._build_status()
        self._enable_drop()

        self.detail = tk.StringVar(value=_INTRO)
        detail_box = ttk.Label(
            root, textvariable=self.detail, wraplength=940,
            justify="left", padding=(12, 10),
        )
        detail_box.pack(fill="x", side="bottom")

        root.after(100, self._drain_queue)

    # ── layout ──────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=(12, 10))
        bar.pack(fill="x")

        ttk.Button(bar, text="Choose file", command=self._pick_file).pack(side="left")
        ttk.Button(bar, text="Choose folder", command=self._pick_dir).pack(
            side="left", padx=(8, 0))

        self.show_info = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            bar, text="Show unknowns (INFO)", variable=self.show_info,
            command=self._refill,
        ).pack(side="right")

    def _build_table(self) -> None:
        frame = ttk.Frame(self.root, padding=(12, 0))
        frame.pack(fill="both", expand=True)

        columns = ("severity", "rule", "file")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=18)
        self.tree.heading("severity", text="Severity")
        self.tree.heading("rule", text="Rule")
        self.tree.heading("file", text="File")
        self.tree.column("severity", width=110, anchor="w", stretch=False)
        self.tree.column("rule", width=170, anchor="w", stretch=False)
        self.tree.column("file", anchor="w")

        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        for severity, colour in _SEVERITY_COLOUR.items():
            self.tree.tag_configure(severity.value, foreground=colour)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

    def _build_status(self) -> None:
        self.status = tk.StringVar(value="Ready")
        ttk.Label(
            self.root, textvariable=self.status, padding=(12, 6),
            relief="flat", anchor="w",
        ).pack(fill="x")

    def _enable_drop(self) -> None:
        """Register for file drops when tkdnd is present.

        tkdnd is an optional Tk extension and is absent on a stock install, so
        the buttons remain the guaranteed path in.
        """
        try:
            self.root.tk.call("package", "require", "tkdnd")
            self.root.tk.call("tkdnd::drop_target", "register", self.root, "DND_Files")
            self.root.bind("<<Drop>>", self._on_drop)
        except tk.TclError:
            pass

    # ── actions ─────────────────────────────────────────────────────

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a model file",
            filetypes=[
                ("Model files", "*.pt *.pth *.pkl *.bin *.safetensors *.gguf "
                                "*.h5 *.keras *.onnx *.pb *.npy *.npz *.joblib "
                                "*.tflite *.skops *.pmml *.mar *.nemo *.7z"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._start(Path(path))

    def _pick_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose a folder of models")
        if path:
            self._start(Path(path))

    def _on_drop(self, event: tk.Event) -> None:
        raw = self.root.tk.splitlist(event.data)
        if raw:
            self._start(Path(raw[0]))

    def _start(self, target: Path) -> None:
        if self.scanning:
            return
        self.scanning = True
        self.target = target
        self.tree.delete(*self.tree.get_children())
        self.findings: list[Finding] = []
        self.detail.set("")
        self.status.set(f"Scanning {target} ...")
        threading.Thread(target=self._worker, args=(target,), daemon=True).start()

    def _worker(self, target: Path) -> None:
        scanner = ModelFileScanner()
        try:
            findings = (
                scanner.scan_file(target) if target.is_file()
                else scanner.scan_directory(target)
            )
            self.queue.put(("done", findings))
        except OSError as exc:
            self.queue.put(("error", str(exc)))

    # ── results ─────────────────────────────────────────────────────

    def _drain_queue(self) -> None:
        try:
            kind, payload = self.queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self.scanning = False
            if kind == "error":
                self.status.set(f"Could not scan: {payload}")
            else:
                self.findings = payload
                self._refill()
        self.root.after(100, self._drain_queue)

    def _refill(self) -> None:
        findings = getattr(self, "findings", [])
        self.tree.delete(*self.tree.get_children())

        shown = [
            f for f in findings
            if self.show_info.get() or f.severity is not Severity.INFO
        ]
        for index, finding in enumerate(
            sorted(shown, key=lambda f: (f.severity_order, f.file_path))
        ):
            try:
                where = Path(finding.file_path).relative_to(self.target)
            except (ValueError, AttributeError):
                where = Path(finding.file_path).name
            self.tree.insert(
                "", "end", iid=str(index),
                values=(finding.severity.value.upper(), finding.rule_id, str(where)),
                tags=(finding.severity.value,),
            )
        self._sorted = sorted(shown, key=lambda f: (f.severity_order, f.file_path))

        actionable = sum(1 for f in findings if f.severity is not Severity.INFO)
        gaps = sum(1 for f in findings if is_coverage_gap(f))
        parts = [f"{actionable} finding(s) above INFO"]
        if gaps:
            parts.append(f"{gaps} file(s) not fully read")
        if not findings:
            parts = ["Nothing found"]
        self.status.set(" | ".join(parts))
        if not shown:
            self.detail.set(_INTRO)

    def _on_select(self, _event: tk.Event) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        finding = self._sorted[int(selection[0])]
        cwe = (
            "  CWE: " + ", ".join(f"CWE-{c}" for c in finding.cwe_ids)
            if finding.cwe_ids else ""
        )
        self.detail.set(
            f"{finding.rule_id}  ({finding.severity.value}){cwe}\n\n{finding.message}"
        )


def main() -> int:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    HaywardApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
