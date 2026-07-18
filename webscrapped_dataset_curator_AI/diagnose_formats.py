#!/usr/bin/env python3
"""
diagnose_formats.py

Run this directly (no MCP, no agent, no Ollama) to see which content
formats are actually usable in your environment before kicking off a real
scrape. Checks each optional dependency group independently so a missing
package for one format (e.g. faster-whisper for video ASR) doesn't hide
whether the others (PDF, DOCX, ...) are fine.

Usage:
    python diagnose_formats.py
"""
import importlib
import shutil
import sys

CHECKS = [
    ("html", ["trafilatura", "readability"], []),
    ("pdf (text layer)", ["pdfplumber"], []),
    ("pdf (OCR fallback, scanned pages)", ["pdf2image", "pytesseract"], ["pdftoppm", "tesseract"]),
    ("docx", ["docx"], []),
    ("pptx", ["pptx"], []),
    ("xlsx / csv", ["openpyxl"], []),
    ("image OCR", ["PIL", "pytesseract"], ["tesseract"]),
    ("video captions", ["yt_dlp"], []),
    ("video/audio ASR fallback", ["yt_dlp", "faster_whisper"], ["ffmpeg"]),
]

print("Checking optional dependencies for each supported content format...\n")

any_missing = False
for label, modules, binaries in CHECKS:
    missing_modules = []
    for m in modules:
        try:
            importlib.import_module(m)
        except ImportError:
            missing_modules.append(m)
    missing_binaries = [b for b in binaries if shutil.which(b) is None]

    if not missing_modules and not missing_binaries:
        print(f"OK    {label}")
    else:
        any_missing = True
        parts = []
        if missing_modules:
            parts.append(f"pip install {' '.join(missing_modules)}")
        if missing_binaries:
            parts.append(f"install system package(s) for: {', '.join(missing_binaries)}")
        print(f"MISS  {label:<38} -- {'; '.join(parts)}")

print()
if any_missing:
    print(
        "Formats marked MISS will return a clear {\"error\": \"...\"} from "
        "extract_content instead of silently producing nothing -- but "
        "they won't contribute data until the listed dependency is "
        "installed. See requirements.txt for the full optional-dependency "
        "breakdown."
    )
else:
    print("All optional format dependencies are installed.")

sys.exit(1 if any_missing else 0)
