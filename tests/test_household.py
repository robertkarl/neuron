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

class DoseRefinementTests(unittest.TestCase):
    def test_damage_and_weak_effect_are_disqualified(self):
        from neuron.refine_household import choose_dose
        safe=dict(dose=.1,task_correct=3,median_lift=3.,fraction_lift_at_least_one=1.,specificity=1.)
        damaged={**safe,'dose':.2,'task_correct':2,'median_lift':6.}
        weak={**safe,'dose':.025,'median_lift':1.}
        self.assertEqual(choose_dose([safe,damaged,weak]),safe)
        self.assertIsNone(choose_dose([damaged,weak]))

    def test_second_confirmation_is_fresh(self):
        from neuron.refine_household import FRESH_CONFIRM, FRESH_FREE, FRESH_TASKS, TUNE_TASKS
        old={x[0] for x in TUNE+CONFIRM+OFF_TOPIC}|set(CALIBRATION)|set(FREE)
        new={x[0] for x in FRESH_CONFIRM+FRESH_TASKS}|set(FRESH_FREE)
        self.assertFalse(old&new)
        self.assertFalse({x[0] for x in TUNE_TASKS}&new)
