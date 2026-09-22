"""Tests for the ID* algorithm of Shpitser & Pearl (JMLR 2008), Section 4."""

import pytest

from pgmpy.base import ADMG, DAG
from pgmpy.identification import Counterfactual, CounterfactualEvent, IDStar
from pgmpy.identification.probability_expression import (
    ConstantNode,
    MarginalNode,
    ProbabilityExpressionTree,
    ProbabilityNode,
    ProductNode,
)

# Fig. 9 (a) of the paper, read off the figure and the make-cg walkthrough on pp. 1959-1960: "the node pairs Dx, D,
# and Xd, X have the same functional mechanisms, and the same parent set (in this case the parents are unobservable
# nodes Ud for the first pair, and U for the second)" fixes D and X as roots, and U as the unobserved parent shared
# by X and Y; "the parents of Wd, W are X and Uw" gives X -> W; "Z, Zx share the parent D" gives D -> Z; and "Y, Yx
# differ on their W-derived parent" together with the Z-derived parent of Y gives W -> Y and Z -> Y.
FIGURE_9A = [
    ("X", "W", "->"),
    ("W", "Y", "->"),
    ("D", "Z", "->"),
    ("Z", "Y", "->"),
    ("X", "Y", "<>"),
]

BOW_ARC = [("X", "Y", "->"), ("X", "Y", "<>")]
CHAIN = [("X", "M", "->"), ("M", "Y", "->")]
FRONT_DOOR = [("X", "M", "->"), ("M", "Y", "->"), ("X", "Y", "<>")]
BACK_DOOR = [("Z", "X", "->"), ("Z", "Y", "->"), ("X", "Y", "->")]

# Lemma 28: X is a parent of Y and Z, and Y and Z are joined by a bidirected path. With no node on that path the
# counterfactual graph is the zig-zag of Fig. 13 (b) at its shortest; W1 and W2 give the figure itself.
ZIG_ZAG = [("X", "Y", "->"), ("X", "Z", "->"), ("Y", "Z", "<>")]
FIGURE_13A = [
    ("X", "Y", "->"),
    ("X", "Z", "->"),
    ("Y", "W1", "<>"),
    ("W1", "W2", "<>"),
    ("W2", "Z", "<>"),
]


def identify(edge_list, event):
    """Run ID* on an ADMG built from `edge_list`."""
    return IDStar().identify(ADMG(edge_list=edge_list), event=event)


def event(variable, value, **intervention):
    """Build a counterfactual event, with the action written as keyword arguments."""
    return CounterfactualEvent(variable, value, intervention=intervention)


class TestIDStarPaperExamples:
    """Cases whose expected output is stated in the paper itself."""

    def test_counterfactual_graph_matches_figure_9c(self):
        """The make-cg walkthrough of pp. 1959-1960, which builds Fig. 9 (c) from Fig. 9 (b) for the query
        P(y_x | x', z_d, d).

        The narrative fixes the whole result: Dx and D merge, Xd and X merge, Wd and W merge, Z, Zx and Zd merge,
        Y and Yd merge, while "W and Wx differ in their X-derived parent, and Y, and Yx differ on their W-derived
        parent" leaves those two pairs apart. The query is relabelled to P(y_x | x', z, d), and "we remove nodes W
        and Y (and their adjacent edges) from consideration".
        """
        algorithm = IDStar()
        conjunction = frozenset([event("Y", "y", X="x"), event("X", "x'"), event("Z", "z", D="d"), event("D", "d")])

        graph, relabelled = algorithm._make_counterfactual_graph(ADMG(edge_list=FIGURE_9A), conjunction)

        # The fixed x is drawn with an overline in the figure; the rest of Fig. 9 (c) is D, X, Z, Wx and Yx.
        assert sorted(map(str, graph.nodes())) == ["D", "W_{X=x}", "X", "X_{X=x}", "Y_{X=x}", "Z"]
        assert sorted(map(str, graph.get_edges(data=True))) == [
            "(D, Z, '->')",
            "(W_{X=x}, Y_{X=x}, '->')",
            "(X, Y_{X=x}, '<>')",
            "(X_{X=x}, W_{X=x}, '->')",
            "(Z, Y_{X=x}, '->')",
        ]
        # Zd has been renamed to Z, so the query reads P(y_x, x', z, d).
        assert {str(node): value for node, value in relabelled.items()} == {
            "Y_{X=x}": "y",
            "X": "x'",
            "Z": "z",
            "D": "d",
        }

    def test_line_6_worked_example(self):
        r"""The trace of P(y_{x,z} | x') on p. 1963, which ends: "our query is identifiable as

        .. math::

            P' = \sum_w P_{z,w}(y, x') P_x(w)

        The two C-components are :math:`\{Y_{x,z}, X\}` and :math:`\{W_{x,z}\}`, and the subscript x of
        :math:`Y_{x,z,w}` is redundant once W is fixed, which is why the first factor is :math:`P_{z,w}(y, x')`
        rather than :math:`P_{x,z,w}(y, x')`.
        """
        result = identify(FIGURE_9A, [event("Y", "y", X="x", Z="z"), event("X", "x'")])
        assert result.to_latex() == (r"\sum_{w} P(W = w \mid do(X = x)) P(X = x', Y = y \mid do(W = w, Z = z))")

    def test_probability_of_necessity_and_sufficiency_is_not_identifiable(self):
        """Lemma 27: "Assume X is a parent of Y in G. Then P,G |/= id P(y_x, y'_x')". The counterfactual graph is
        the w-graph of Fig. 8 (b): Yx and Yx' are joined by the unobserved parent they share, and the two actions
        set X to conflicting values."""
        algorithm = IDStar()
        conjunction = [event("Y", "y", X="x"), event("Y", "y'", X="x'")]

        assert algorithm.identify(ADMG(edge_list=[("X", "Y", "->")]), event=conjunction) is False

        witness = algorithm.hedge_
        assert sorted(map(str, witness.nodes())) == ["X_{X=x'}", "X_{X=x}", "Y_{X=x'}", "Y_{X=x}"]
        assert sorted(map(str, witness.get_edges(data=True))) == [
            "(X_{X=x'}, Y_{X=x'}, '->')",
            "(X_{X=x}, Y_{X=x}, '->')",
            "(Y_{X=x'}, Y_{X=x}, '<>')",
        ]

    def test_lemma_27_also_rules_out_the_one_world_form(self):
        """The second query of Lemma 27, P(y_x, y'), where one of the two counterfactuals is an observation."""
        assert identify([("X", "Y", "->")], [event("Y", "y", X="x"), event("Y", "y'")]) is False

    @pytest.mark.parametrize(
        ("name", "edge_list", "conjunction"),
        [
            ("no node on the bidirected path", ZIG_ZAG, [event("Y", "y", X="x"), event("Z", "z", X="x'")]),
            ("one world observed", ZIG_ZAG, [event("Y", "y", X="x"), event("Z", "z")]),
            (
                "figure 13",
                FIGURE_13A,
                [
                    event("Y", "y", X="x"),
                    event("W1", "w1"),
                    event("W2", "w2"),
                    event("Z", "z", X="x'"),
                ],
            ),
        ],
    )
    def test_zig_zag_graphs_are_not_identifiable(self, name, edge_list, conjunction):
        """Lemma 28: "Assume G is such that X is a parent of Y and Z, and Y and Z are connected by a bidirected
        path with observable nodes W1, ..., Wk on the path. Then P,G |/= id P(y_x, w1, ..., wk, z_x'),
        P(y_x, w1, ..., wk, z)"."""
        assert identify(edge_list, conjunction) is False


class TestIDStarBaseCases:
    """Lines 1, 2, 3 and 5, which answer without looking at the graph at all."""

    def test_empty_conjunction_has_probability_one(self):
        """Line 1."""
        assert identify(CHAIN, []).root == ConstantNode(1)

    def test_effectiveness_violation_has_probability_zero(self):
        """Line 2: X cannot attain x' in a world where it has been forced to x."""
        assert identify(CHAIN, [event("X", "x'", X="x"), event("Y", "y")]).root == ConstantNode(0)

    def test_a_tautological_event_is_dropped(self):
        """Line 3: X attains x in a world where it has been forced to x, so the event says nothing."""
        assert identify(CHAIN, [event("X", "x", X="x")]).root == ConstantNode(1)
        assert identify(CHAIN, [event("X", "x", X="x"), event("Y", "y")]) == identify(CHAIN, [event("Y", "y")])

    def test_two_events_on_one_counterfactual_variable_conflict(self):
        """The same counterfactual variable cannot attain two values."""
        assert identify(CHAIN, [event("Y", "y"), event("Y", "y'")]).root == ConstantNode(0)

    def test_merging_can_expose_an_inconsistency(self):
        """Line 5: Z is not a descendant of X, so Z and Z_x are the same random variable and make-cg merges them.
        The query then assigns one variable two values."""
        graph = [("Z", "Y", "->"), ("X", "Y", "->")]
        assert identify(graph, [event("Z", "z"), event("Z", "z'", X="x")]).root == ConstantNode(0)
        # With the same value on both, the merge goes through and leaves a single event behind.
        assert identify(graph, [event("Z", "z"), event("Z", "z", X="x")]).to_latex() == "P(Z = z)"


class TestIDStarIdentifiableQueries:
    """Identifiable queries, asserted against their exact closed-form expression."""

    def test_a_plain_causal_effect(self):
        """A query mentioning a single world is an interventional quantity, and ID* returns it as one."""
        assert identify(BOW_ARC, [event("Y", "y", X="x")]).to_latex() == r"P(Y = y \mid do(X = x))"

    def test_effect_of_treatment_on_the_treated_in_the_front_door_graph(self):
        """P(y_x, x'), the numerator of the effect of treatment on the treated. The mediator is the only node
        outside the C-component of Yx and X, so it is what the two terms are joined over."""
        result = identify(FRONT_DOOR, [event("Y", "y", X="x"), event("X", "x'")])
        assert result.to_latex() == (r"\sum_{m} P(M = m \mid do(X = x)) P(X = x', Y = y \mid do(M = m))")

    def test_effect_of_treatment_on_the_treated_in_the_back_door_graph(self):
        result = identify(BACK_DOOR, [event("Y", "y", X="x"), event("X", "x'")])
        assert result.to_latex() == (r"\sum_{z} P(X = x' \mid do(Z = z)) P(Y = y \mid do(X = x, Z = z)) P(Z = z)")

    def test_two_events_in_the_same_world(self):
        """Both events speak of the same submodel, so nothing counterfactual is left: the answer is a product of
        terms of one interventional distribution."""
        result = identify(BACK_DOOR, [event("Y", "y", X="x"), event("Z", "z")])
        assert result.to_latex() == r"P(Y = y \mid do(X = x, Z = z)) P(Z = z)"

    def test_an_observational_query(self):
        """A query with no action at all is still answered in terms of experimental distributions, since that is
        the information ID* identifies from."""
        result = identify(CHAIN, [event("Y", "y"), event("X", "x")])
        assert result.to_latex() == (r"\sum_{m} P(M = m \mid do(X = x)) P(X = x) P(Y = y \mid do(M = m))")

    def test_a_free_node_is_summed_over(self):
        """M is an ancestor of the query that the query does not mention, so Line 6 sums over its value. So is X,
        which is summed over under a name of its own."""
        result = identify(CHAIN, [event("Y", "y")])
        assert result.to_latex() == (r"\sum_{m, x} P(M = m \mid do(X = x)) P(X = x) P(Y = y \mid do(M = m))")

    def test_returns_an_expression_tree(self):
        result = identify(FRONT_DOOR, [event("Y", "y", X="x"), event("X", "x'")])
        assert isinstance(result, ProbabilityExpressionTree)
        assert isinstance(result.root, MarginalNode)
        assert isinstance(result.root.children[0], ProductNode)
        assert all(isinstance(leaf, ProbabilityNode) for leaf in result.find_leaves())


class TestIDStarNonIdentifiable:
    """Queries blocked by a conflict, which Line 8 throws FAIL on."""

    def test_effect_of_treatment_on_the_treated_in_the_bow_arc(self):
        """X and Yx are confounded and end up in one C-component, where the action sets X to x while the other
        event observes it to be x'."""
        algorithm = IDStar()
        assert algorithm.identify(ADMG(edge_list=BOW_ARC), event=[event("Y", "y", X="x"), event("X", "x'")]) is False

        # Theorem 32: the witness is the counterfactual graph, and the C-component carrying the conflict is in it.
        witness = algorithm.hedge_
        assert sorted(map(str, witness.nodes())) == ["X", "X_{X=x}", "Y_{X=x}"]
        assert ("X", "Y_{X=x}", "<>") in [
            (str(u), str(v), edge_type) for u, v, edge_type in witness.get_edges(data=True)
        ]

    def test_a_counterfactual_mediator_alongside_an_observed_outcome(self):
        """M and Mx share an unobserved parent and are forced apart by conflicting actions on X."""
        assert identify(FRONT_DOOR, [event("M", "m", X="x"), event("Y", "y")]) is False

    def test_failure_propagates_out_of_the_line_6_decomposition(self):
        """Only one of the C-components of the bow arc query hedges, and that FAIL must abort the whole product
        rather than being swallowed."""
        assert identify(BOW_ARC + [("X", "Z", "->")], [event("Y", "y", X="x"), event("X", "x'")]) is False

    def test_hedge_is_reset_between_runs(self):
        algorithm = IDStar()
        assert algorithm.identify(ADMG(edge_list=BOW_ARC), event=[event("Y", "y", X="x"), event("X", "x'")]) is False
        assert algorithm.hedge_ is not None
        assert algorithm.identify(ADMG(edge_list=BOW_ARC), event=[event("Y", "y", X="x")]) is not False
        assert algorithm.hedge_ is None


class TestMakeCounterfactualGraph:
    """The construction of Fig. 10, tested directly."""

    def test_parallel_worlds_share_their_unobserved_parents(self):
        """The copies of a variable in two worlds are always confounded: they are functions of the same
        unobserved parent. Without that edge the w-graph would fall apart into two independent worlds."""
        algorithm = IDStar()
        graph = algorithm._parallel_worlds_graph(ADMG(edge_list=[("X", "Y", "->")]), [(("X", "x"),), (("X", "x'"),)])

        assert graph.has_edge(Counterfactual("Y", {"X": "x"}), Counterfactual("Y", {"X": "x'"}), "<>")

    def test_an_intervention_cuts_every_arrow_into_the_variable_it_fixes(self):
        """Both the directed edges and the bidirected ones, since a bidirected edge is an unobserved parent."""
        algorithm = IDStar()
        graph = algorithm._parallel_worlds_graph(ADMG(edge_list=BOW_ARC), [(), (("X", "x"),)])

        fixed = Counterfactual("X", {"X": "x"})
        assert graph.get_parents(fixed) == set()
        assert graph.get_spouses(fixed) == set()
        # The same variable is free in the other world, where its unobserved parent survives -- and, being shared
        # by the worlds, confounds it with the copy of Y in each of them.
        assert graph.get_spouses(Counterfactual("X")) == {
            Counterfactual("Y"),
            Counterfactual("Y", {"X": "x"}),
        }

    def test_an_observed_parent_can_stand_in_for_an_intervened_one(self):
        """The Z, Zx, Zd triplet of p. 1960: "in our counterfactual query ... the variable D happens to be observed
        to attain the value d, the same as the intervention value for the parent of Zd. This implies that for the
        purposes of the Z, Zx, Zd triplet, their D-derived parents share the same value, which allows us to
        conclude they are the same random variable"."""
        algorithm = IDStar()
        graph, _ = algorithm._make_counterfactual_graph(
            ADMG(edge_list=FIGURE_9A),
            frozenset([event("Y", "y", X="x"), event("X", "x'"), event("Z", "z", D="d"), event("D", "d")]),
        )

        assert Counterfactual("Z") in graph.nodes()
        assert Counterfactual("Z", {"D": "d"}) not in graph.nodes()

        # Without the observation of D, the D-derived parents of Z and Zd have no value in common and the two stay
        # apart.
        graph, _ = algorithm._make_counterfactual_graph(
            ADMG(edge_list=FIGURE_9A), frozenset([event("Z", "z", D="d"), event("Z", "z'")])
        )
        assert Counterfactual("Z", {"D": "d"}) in graph.nodes()
        assert Counterfactual("Z") in graph.nodes()

    def test_the_graph_is_restricted_to_the_ancestors_of_the_query(self):
        """W and Y are dropped from Fig. 9 (c) because nothing in the relabelled query descends from them."""
        algorithm = IDStar()
        graph, _ = algorithm._make_counterfactual_graph(ADMG(edge_list=FIGURE_9A), frozenset([event("Z", "z", D="d")]))
        assert sorted(map(str, graph.nodes())) == ["D_{D=d}", "Z_{D=d}"]

    def test_subscript_is_read_off_the_graph(self):
        """A node keeps the name of the world it was created in, so what it is taken under is the set of fixed
        nodes it descends from, the An(w) cap sub(gamma) of p. 1958."""
        algorithm = IDStar()
        graph, _ = algorithm._make_counterfactual_graph(
            ADMG(edge_list=FIGURE_9A),
            frozenset([event("Y", "y", X="x"), event("X", "x'"), event("Z", "z", D="d"), event("D", "d")]),
        )

        assert algorithm._subscript(graph, Counterfactual("Y", {"X": "x"})) == {("X", "x")}
        # Z was merged out of the world that fixes D, so it is no longer taken under any action at all.
        assert algorithm._subscript(graph, Counterfactual("Z")) == set()


class TestCounterfactualEvent:
    def test_counterfactual_variable(self):
        node = Counterfactual("Y", {"X": "x", "Z": "z"})
        assert node.variable == "Y"
        assert node.action == {"X": "x", "Z": "z"}
        assert repr(node) == "Y_{X=x, Z=z}"
        assert repr(Counterfactual("Y")) == "Y"

    def test_the_action_is_a_set_of_assignments(self):
        """Two counterfactuals written with their actions in a different order are the same variable."""
        assert Counterfactual("Y", {"X": "x", "Z": "z"}) == Counterfactual("Y", [("Z", "z"), ("X", "x")])
        assert hash(Counterfactual("Y", {"X": "x"})) == hash(Counterfactual("Y", [("X", "x")]))
        assert Counterfactual("Y") == Counterfactual("Y", {})
        assert Counterfactual("Y", {"X": "x"}) != Counterfactual("Y", {"X": "x'"})

    def test_event(self):
        item = CounterfactualEvent("Y", "y", intervention={"X": "x"})
        assert item.variable == "Y"
        assert item.value == "y"
        assert item.action == {"X": "x"}
        assert item.counterfactual == Counterfactual("Y", {"X": "x"})
        assert repr(item) == "Y_{X=x} = y"

    def test_events_differing_only_in_value_are_different(self):
        assert CounterfactualEvent("Y", "y") != CounterfactualEvent("Y", "y'")
        assert len({CounterfactualEvent("Y", "y"), CounterfactualEvent("Y", "y")}) == 1


class TestIDStarInputHandling:
    def test_dag_input(self):
        dag = DAG(ebunch=[("X", "M"), ("M", "Y")])
        assert IDStar().identify(dag, event=[event("Y", "y", X="x")]).to_latex() == (
            r"\sum_{m} P(M = m \mid do(X = x)) P(Y = y \mid do(M = m))"
        )

    def test_dag_latents_are_projected_to_bidirected_edges(self):
        """A DAG that names its confounder must behave like the hand-written bow arc."""
        dag = DAG(ebunch=[("U", "X"), ("U", "Y"), ("X", "Y")], latents={"U"})
        conjunction = [event("Y", "y", X="x"), event("X", "x'")]

        assert IDStar().identify(dag, event=conjunction) is False
        assert IDStar().identify(ADMG(edge_list=BOW_ARC), event=conjunction) is False

    def test_a_latent_cannot_appear_in_the_query(self):
        dag = DAG(ebunch=[("U", "X"), ("X", "Y")], latents={"U"})
        with pytest.raises(ValueError, match="cannot be both latent"):
            IDStar().identify(dag, event=[event("U", "u")])

        with pytest.raises(ValueError, match="cannot be both latent"):
            IDStar().identify(dag, event=[event("Y", "y", U="u")])

    def test_no_roles_are_required(self):
        """The query is the `event` argument, so a graph with no roles assigned is fine."""
        assert IDStar().identify(ADMG(edge_list=CHAIN), event=[event("Y", "y", X="x")]) is not False

    def test_a_single_event_need_not_be_wrapped(self):
        assert IDStar().identify(ADMG(edge_list=CHAIN), event=event("Y", "y", X="x")) == identify(
            CHAIN, [event("Y", "y", X="x")]
        )

    def test_the_query_must_be_over_nodes_of_the_graph(self):
        with pytest.raises(ValueError, match=r"\['Q'\] in `event` are not nodes"):
            identify(CHAIN, [event("Q", "q")])

        # The action is checked too, not just the variable the event is about.
        with pytest.raises(ValueError, match=r"\['Q'\] in `event` are not nodes"):
            identify(CHAIN, [event("Y", "y", Q="q")])

    def test_the_query_must_be_made_of_events(self):
        with pytest.raises(ValueError, match="CounterfactualEvent"):
            identify(CHAIN, ["Y = y"])

    def test_unsupported_graph_type(self):
        with pytest.raises(ValueError, match="must be an instance of"):
            IDStar().identify("not a graph", event=[])
