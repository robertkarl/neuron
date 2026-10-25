import struct
import tempfile
from pathlib import Path
import unittest

import numpy as np

from neuron.backend import read_trace
from neuron.household import summarize_effect, TUNE, CONFIRM, CALIBRATION, FREE, OFF_TOPIC


class ProbabilityTests(unittest.TestCase):
    def test_targeted_logprob_records_do_not_contaminate_activation_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace'
            path.write_bytes(b'NEURON01' + struct.pack('<4I', 4, 0, 4, 1)
                             + struct.pack('<4f', 10183, -4, 27056, -6))
            logprobs = []
            self.assertEqual(read_trace(path, logprobs), ({}, {}, []))
            self.assertEqual(logprobs, [{10183: -4., 27056: -6.}])

    def test_invalid_probabilities_and_token_ids_are_rejected(self):
        for token, probability in [(1, 1), (248320, -1), (3.5, -1)]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'trace'
                path.write_bytes(b'NEURON01' + struct.pack('<4I', 4, 0, 2, 1)
                                 + struct.pack('<2f', token, probability))
                with self.assertRaises(ValueError):
                    read_trace(path)


class CausalScreenTests(unittest.TestCase):
    def setUp(self):
        self.baseline = np.full((10, 10), -8.)
        self.positive = self.baseline.copy()
        self.positive[:, 3] += 3.
        self.negative = self.baseline.copy()
        self.negative[:, 3] -= 2.
        self.controls = [self.baseline + .1, self.baseline - .1, self.baseline + .2]

    def test_accepts_consistent_specific_causal_effect(self):
        result = summarize_effect(self.baseline, self.positive, self.negative, self.controls, 3)
        self.assertEqual(result['status'], 'accepted_token_steering')
        self.assertEqual(result['median_logprob_lift'], 3.)
        self.assertGreater(result['steered_probability'], result['baseline_probability'])

    def test_rejects_broad_or_random_channel_effects(self):
        broad = self.baseline + 3
        self.assertEqual(summarize_effect(self.baseline,broad,self.negative,self.controls,3)['status'], 'rejected')
        self.controls[0] = self.positive.copy()
        self.assertEqual(summarize_effect(self.baseline,self.positive,self.negative,self.controls,3)['status'], 'rejected')

    def test_rejects_inconsistent_context_effects(self):
        self.positive[:3, 3] = -8.
        self.assertEqual(summarize_effect(self.baseline,self.positive,self.negative,self.controls,3)['status'], 'rejected')

    def test_confirmation_prompts_are_disjoint(self):
        prior = {x[0] for x in TUNE} | set(CALIBRATION)
        self.assertFalse(prior & {x[0] for x in CONFIRM})
        self.assertFalse({x[0] for x in CONFIRM} & set(FREE))
        self.assertFalse({x[0] for x in CONFIRM} & {x[0] for x in OFF_TOPIC})


if __name__ == '__main__':
    unittest.main()
