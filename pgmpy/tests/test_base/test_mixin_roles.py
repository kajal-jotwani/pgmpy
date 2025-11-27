#!/usr/bin/env python3
import unittest

from pgmpy.base import DAG


class TestGraphRolesMixin(unittest.TestCase):
    def setUp(self):
        self.G = DAG(ebunch=[("X", "Y"), ("Z", "Y")])
        self.G.add_node("U")

    def test_with_role_single_variable(self):
        self.G.with_role(role="exposures", variables="X", inplace=True)

        self.assertIn("roles", self.G.nodes["X"])
        self.assertEqual(self.G.nodes["X"]["roles"], {"exposures"})
        self.assertEqual(set(self.G.get_role("exposures")), {"X"})
        self.assertTrue(self.G.has_role("exposures"))

    def test_with_role_multiple_variables(self):
        self.G.with_role(role="outcomes", variables={"Y", "Z"}, inplace=True)

        self.assertEqual(self.G.nodes["Y"]["roles"], {"outcomes"})
        self.assertEqual(self.G.nodes["Z"]["roles"], {"outcomes"})
        self.assertEqual(set(self.G.get_role("outcomes")), {"Y", "Z"})

    def test_with_role_adds_without_overwriting_existing_roles(self):
        self.G.with_role(role="exposures", variables="X", inplace=True)
        self.G.with_role(role="outcomes", variables={"X", "Y"}, inplace=True)

        self.assertEqual(self.G.nodes["X"]["roles"], {"exposures", "outcomes"})
        self.assertEqual(self.G.nodes["Y"]["roles"], {"outcomes"})

    def test_with_role_raises_for_missing_variable(self):
        with self.assertRaises(ValueError):
            self.G.with_role(role="exposures", variables="MISSING", inplace=True)

    def test_get_roles_and_get_role_dict(self):
        self.G.with_role(role="exposures", variables="X", inplace=True)
        self.G.with_role(role="outcomes", variables={"Y", "Z"}, inplace=True)
        self.G.with_role(role="latents", variables="U", inplace=True)

        roles = set(self.G.get_roles())
        self.assertEqual(roles, {"exposures", "outcomes", "latents"})

        role_dict = self.G.get_role_dict()
        self.assertEqual(set(role_dict.keys()), roles)
        self.assertEqual(set(role_dict["exposures"]), {"X"})
        self.assertEqual(set(role_dict["outcomes"]), {"Y", "Z"})
        self.assertEqual(set(role_dict["latents"]), {"U"})

    def test_without_role_specific_variables(self):
        self.G.with_role("exposures", {"X", "Z"}, inplace=True)
        self.G.with_role("outcomes", {"Y"}, inplace=True)

        self.G.without_role(role="exposures", variables="X", inplace=True)

        self.assertNotIn("exposures", self.G.nodes["X"].get("roles", set()))
        self.assertIn("exposures", self.G.nodes["Z"]["roles"])
        self.assertIn("outcomes", self.G.nodes["Y"]["roles"])

    def test_without_role_removes_roles_attr_when_last_role_removed(self):
        self.G.with_role("exposures", "X", inplace=True)

        # After removing the only role, "roles" attribute should disappear
        self.G.without_role(role="exposures", variables="X", inplace=True)
        self.assertNotIn("roles", self.G.nodes["X"])

    def test_without_role_all_variables_when_variables_none(self):
        self.G.with_role("exposures", {"X", "Z"}, inplace=True)
        self.G.with_role("outcomes", {"Y"}, inplace=True)

        self.G.without_role(role="exposures", variables=None, inplace=True)

        self.assertNotIn("exposures", self.G.nodes["X"].get("roles", set()))
        self.assertNotIn("exposures", self.G.nodes["Z"].get("roles", set()))
        self.assertIn("outcomes", self.G.nodes["Y"]["roles"])

    def test_latents_property_and_observed(self):
        G = DAG(ebunch=[("a", "b")], latents="a")

        self.assertEqual(G.latents, {"a"})
        self.assertEqual(G.observed, {"b"})

        # Setting latents again should replace the old latent set
        G.latents = {"b"}
        self.assertEqual(G.latents, {"b"})
        self.assertEqual(G.observed, {"a"})

    def test_observed_when_no_latents(self):
        G = DAG(ebunch=[("a", "b")])
        self.assertEqual(G.latents, set())
        self.assertEqual(G.observed, {"a", "b"})

    def test_exposures_and_outcomes_properties(self):
        G = DAG(ebunch=[("X", "Y")])

        G.exposures = "X"
        G.outcomes = {"Y"}

        self.assertEqual(G.exposures, {"X"})
        self.assertEqual(G.outcomes, {"Y"})

        # Changing exposures should replace the previous exposures role
        G.exposures = {"Y"}
        self.assertEqual(G.exposures, {"Y"})
        self.assertEqual(G.outcomes, {"Y"})

    def test_is_valid_causal_structure_raises_when_missing_roles(self):
        G = DAG(ebunch=[("X", "Y")])

        # No roles at all
        with self.assertRaises(ValueError):
            G.is_valid_causal_structure()

        # Only exposures
        G.exposures = "X"
        with self.assertRaises(ValueError):
            G.is_valid_causal_structure()

        # Only outcomes
        G = DAG(ebunch=[("X", "Y")])
        G.outcomes = "Y"
        with self.assertRaises(ValueError):
            G.is_valid_causal_structure()

    def test_is_valid_causal_structure_passes_with_exposures_and_outcomes(self):
        G = DAG(ebunch=[("X", "Y")])
        G.exposures = "X"
        G.outcomes = "Y"

        self.assertTrue(G.is_valid_causal_structure())
