import struct
from pathlib import Path
import tempfile
import unittest

import numpy as np

from neuron.backend import read_trace
from neuron.search import alpha_for, evidence_decision, lexical_score, rank_candidates, validate_dataset
from neuron.topics import topic_data


class SearchTests(unittest.TestCase):
    def test_contrast_finds_increased_and_suppressed_channels(self):
        positive = [{2: np.array([10., -9., 2.])}, {2: np.array([12., -11., 4.])}]
        negative = [{2: np.array([0., 0., 2.])}, {2: np.array([2., 2., 4.])}]
        ranked = rank_candidates(positive, negative, 2)
        self.assertEqual([c['neuron'] for c in ranked], [1, 0])
        self.assertEqual([c['direction'] for c in ranked], [-1, 1])

    def test_column_normalization_gives_requested_residual_fraction(self):
        column = np.array([3., 4.])
        alpha = alpha_for(-0.5, 20., np.linalg.norm(column))
        self.assertEqual(alpha, -2.)
        self.assertEqual(np.linalg.norm(alpha * column), 10.)
        for norm in [0, float('nan'), float('inf'), -1]:
            with self.assertRaises(ValueError):
                alpha_for(1, 20, norm)

    def test_null_and_random_control_prevent_discovery_claim(self):
        self.assertEqual(evidence_decision(0, 0, 0, 0, 1)['status'], 'no_evidence')
        self.assertEqual(evidence_decision(0, 0.75, 0, 0.75, 1)['status'], 'no_evidence')
        self.assertEqual(evidence_decision(0, 0.75, 0, 0, 0.5)['status'], 'no_evidence')
        self.assertEqual(evidence_decision(0, 0.75, 0, 0, 1)['status'], 'candidate')

    def test_scoring_respects_word_boundaries(self):
        self.assertEqual(lexical_score('This GPU renders graphics.', ['gpu']), 1)
        self.assertEqual(lexical_score('a gpupdate command', ['gpu']), 0)
        self.assertEqual(lexical_score('a GRAPHICS CARD', ['graphics card']), 1)

    def test_data_split_leakage_rejected(self):
        data = topic_data('gpu')
        validate_dataset(data)
        data['heldout'] = data['tune'][:]
        with self.assertRaises(ValueError):
            validate_dataset(data)

    def test_generic_bootstrap_is_labeled(self):
        data = topic_data('volcano')
        validate_dataset(data)
        self.assertIn('bootstrap', data['quality'])
        self.assertEqual(data['terms'], ['volcano'])


class TraceTests(unittest.TestCase):
    def parse(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace'
            path.write_bytes(data)
            return read_trace(path)

    def test_microbatches_are_weighted_by_token_count(self):
        record1 = struct.pack('<4I', 1, 4, 9216, 2) + np.full(9216, 3, dtype='<f4').tobytes()
        record2 = struct.pack('<4I', 1, 4, 9216, 1) + np.full(9216, 9, dtype='<f4').tobytes()
        residual = struct.pack('<4I', 2, 4, 1, 2) + struct.pack('<2f', 3, 4)
        change = struct.pack('<4I', 3, 4, 3, 2) + struct.pack('<3f', 9, 1, 5)
        means, norms, interventions = self.parse(b'NEURON01' + record1 + record2 + residual + change)
        np.testing.assert_array_equal(means[4], 5)
        self.assertEqual(norms[4], [3, 4])
        self.assertEqual(interventions[0]['after'] - interventions[0]['before'], 4)

    def test_truncation_and_unknown_record_rejected(self):
        for raw in [b'', b'NEURON01x', b'NEURON01' + struct.pack('<4I', 1, 0, 9216, 1),
                    b'NEURON01' + struct.pack('<4I', 9, 0, 1, 1),
                    b'NEURON01' + struct.pack('<4I', 2, 0, 1, 1) + struct.pack('<f', float('nan'))]:
            with self.assertRaises(ValueError):
                self.parse(raw)


if __name__ == '__main__':
    unittest.main()
