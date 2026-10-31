# PDF processing scripts — baseline before line-grouping experiment

This is the last production script set, based on the optimized group-neighbor filter.
No new 3-em line extension or group merging has been implemented in this baseline.

## Main scripts

1. rotate_pdf.py — rotate PDF pages using metadata.
2. split_book_spreads.py — split book spreads into individual pages.
3. clean_scanned_pdf.py — detect/deskew/crop pages and assemble the cleaned PDF.

## Imported support modules

- image_enhancement.py — grayscale local correction, histogram contrast, text masks and JPEG encoding.
- text_detection.py — candidate ink detection.
- text_filtering.py — optimized group-based height-neighbor filtering.
- remove_vertical_scan_artifacts.py — optional vertical edge cleanup.
- remove_slanted_edge_artifacts.py — optional slanted edge cleanup.

## Dependencies

pip install pymupdf opencv-python numpy scipy

Keep all eight .py files together. The source PDFs are not included; the originals remain in /mnt/data.
