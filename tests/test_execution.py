import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from jace_rec.io import Project
from jace_rec.release_execution import gpu_policy_root, snapshot

ROOT = Path(__file__).resolve().parents[1]


class ExecutionTests(unittest.TestCase):
    def test_frozen_snapshot_rejects_modified_code_and_existing_results(self):
        project = Project(ROOT/'configs/current-toys-2023.json')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'existing'
            output.mkdir()
            (output/'predictions.jsonl').write_text('preserve')
            with self.assertRaises(FileExistsError):
                snapshot(project, output)
            self.assertEqual((output/'predictions.jsonl').read_text(), 'preserve')
            code = snapshot(project, Path(tmp)/'fresh')
            self.assertEqual(snapshot(project, Path(tmp)/'fresh'), code)
            (code/'src/jace_rec/gating.py').write_text('changed')
            with self.assertRaises(ValueError):
                snapshot(project, Path(tmp)/'fresh')

    def test_nested_checkout_reuses_shared_policy_and_explicit_root(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            parent = Path(tmp)
            (parent/'configs').mkdir()
            (parent/'configs/gpu-account-policy.json').write_text('{}')
            nested = parent/'release/JACE-Rec'
            nested.mkdir(parents=True)
            self.assertEqual(gpu_policy_root(nested), parent)
            with patch.dict(os.environ, {'JACE_GPU_POLICY_ROOT': str(parent/'shared')}):
                self.assertEqual(gpu_policy_root(nested), parent/'shared')
