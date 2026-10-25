import unittest
from neuron.tweet_repro import RECIPE, verify_interventions


class TweetRecipeTests(unittest.TestCase):
    def test_coordinates_and_signs_match_supplied_recipe(self):
        self.assertEqual(RECIPE,[dict(layer=14,neuron=8949,dose=1.),dict(layer=13,neuron=7956,dose=-1.)])

    def test_both_channels_and_token_phases_are_verified(self):
        steering=[dict(layer=14,neuron=8949,alpha=23.),dict(layer=13,neuron=7956,alpha=-22.)]
        records=[dict(layer=x['layer'],neuron=x['neuron'],before=1.,after=1.+x['alpha'],count=count)
                 for x in steering for count in [20,1,1]]
        evidence=verify_interventions(records,steering)
        self.assertTrue(all(x['prefill'] and x['decode'] and x['evaluated_tokens']==22 for x in evidence))
        with self.assertRaises(RuntimeError):
            verify_interventions(records[:3],steering)
        records[-1]['after']+=1
        with self.assertRaises(RuntimeError):
            verify_interventions(records,steering)
