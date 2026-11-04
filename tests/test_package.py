"""Smoke and packaging checks; run with `python -m unittest discover -s tests`."""
import unittest
from pathlib import Path
from unittest.mock import patch, call
from scan_cleaner import process_pdf
from scan_cleaner import cli


class PackageSmokeTests(unittest.TestCase):
    def test_public_api(self):
        self.assertTrue(callable(process_pdf))
        self.assertTrue(callable(cli.main))

    def test_refuses_missing_source(self):
        with self.assertRaises(FileNotFoundError):
            process_pdf('definitely-not-present-source.pdf','output.pdf')

    def test_rejects_dpi_upsampling(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp) / 'input.pdf'
            src.touch()
            with self.assertRaises(ValueError):
                process_pdf(src, Path(temp) / 'output.pdf', processing_dpi=200, output_dpi=320)

    def test_does_not_silently_drop_gutter_sidecar(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            src = Path(temp) / 'input.pdf'
            src.touch()
            with patch('scan_cleaner.pipeline._run_module') as run:
                with self.assertRaises(FileNotFoundError):
                    process_pdf(src, Path(temp) / 'output.pdf')
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0], 'split_book_spreads')
