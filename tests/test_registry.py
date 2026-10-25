"""Audit the published registry against saved experiment evidence."""
import json
import unittest

import numpy as np

from neuron.backend import ROOT, digest
from neuron.household import ITEMS, summarize_effect
from neuron.refine_household import answer_correct
from neuron.search import alpha_for


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads((ROOT/'data/household-neurons.json').read_text())
        cls.evidence_path = ROOT/cls.registry['evidence']
        cls.evidence = json.loads(cls.evidence_path.read_text())

    def test_provenance_and_model_identity(self):
        self.assertEqual(self.registry['evidence_sha256'], digest(self.evidence_path))
        self.assertEqual(self.registry['model'], json.loads((ROOT/'models.lock.json').read_text()))
        self.assertEqual(self.registry['hook_sha256'], digest(ROOT/'backend/neuron-hook.h'))
        self.assertEqual(set(self.registry['neurons']), set(ITEMS))

    def test_every_coordinate_and_offset_matches_measured_intervention(self):
        for item, entry in self.registry['neurons'].items():
            with self.subTest(item=item):
                result = self.evidence['results'][item]
                coordinate = result['coordinate']
                self.assertEqual((entry['layer'], entry['neuron']), (coordinate['layer'], coordinate['neuron']))
                self.assertEqual(entry['signed_dose'], result['dose']*coordinate['direction'])
                self.assertAlmostEqual(entry['alpha'], alpha_for(entry['signed_dose'], entry['residual_median'], entry['column_norm']))
                self.assertEqual(entry['alpha'], result['alpha'])
                self.assertEqual(self.evidence['tokenization'][item][item], [entry['bare_token']])
                self.assertEqual(self.evidence['tokenization'][item][' '+item], [entry['space_token']])

    def test_all_ten_acceptances_recompute_from_raw_evidence(self):
        for i, item in enumerate(ITEMS):
            with self.subTest(item=item):
                result = self.evidence['results'][item]
                decision = summarize_effect(self.evidence['confirmation_baseline'], result['positive'], result['negative'], result['random_values'], i)
                self.assertEqual(decision['status'], 'accepted_token_steering')
                self.assertEqual(sum(answer_correct(x['text'], x['expected']) for x in result['off_topic']), 4)
                lift = np.asarray(result['positive'])[:,i] - np.asarray(self.evidence['confirmation_baseline'])[:,i]
                self.assertAlmostEqual(float(np.median(np.exp(lift))), self.registry['neurons'][item]['median_probability_ratio'])
