"""Tests for the ID algorithm for causal effect identification."""

from pgmpy.base import ADMG
from pgmpy.identification import ID
from pgmpy.identification.probability_expression import (
    MarginalNode,
    ProbabilityExpressionTree,
    ProbabilityNode,
    ProductNode,
)


class TestID:
    """Test suite for the ID algorithm."""

    def test_identify(self):
        """Test identify() returns correct symbolic expressions."""
        # Simple chain X -> M -> Y: P(Y|do(X)) = ∑_M P(M|X) P(Y|M,X)
        admg = ADMG(edge_list=[("X", "M", "->"), ("M", "Y", "->")], exposures={"X"}, outcomes={"Y"})
        result = ID().identify(admg)
        assert result is not False
        assert isinstance(result, ProbabilityExpressionTree)
        assert isinstance(result.root, MarginalNode)
        assert result.root.sumset == {"M"}
        assert result.to_latex() == r"\sum_{M} P(M \mid X) P(Y \mid M, X)"

        # Hedge X -> Y with X <> Y: should return False
        admg = ADMG(edge_list=[("X", "Y", "->"), ("X", "Y", "<>")], exposures={"X"}, outcomes={"Y"})
        id_algo = ID()
        result = id_algo.identify(admg)
        assert result is False
        assert id_algo.hedge_ is not None

        # Front-door X -> M -> Y with X <> Y: identifiable
        admg = ADMG(edge_list=[("X", "M", "->"), ("M", "Y", "->"), ("X", "Y", "<>")], exposures={"X"}, outcomes={"Y"})
        result = ID().identify(admg)
        assert result is not False
        assert isinstance(result, ProbabilityExpressionTree)
        # Verify it returns a marginalization over M
        assert isinstance(result.root, MarginalNode)
        assert result.root.sumset == {"M"}

        # Bow-arc X -> Z -> Y with X <> Z: non-identifiable
        admg = ADMG(edge_list=[("X", "Z", "->"), ("Z", "Y", "->"), ("X", "Z", "<>")], exposures={"X"}, outcomes={"Y"})
        id_algo = ID()
        result = id_algo.identify(admg)
        assert result is False

    def test_marginal(self):
        """Test _marginal() creates correct MarginalNode."""
        # Marginal with sumset: should create MarginalNode
        result = ID()._marginal(frozenset({"Y"}), frozenset({"X", "Y", "Z"}), None)
        assert isinstance(result, ProbabilityExpressionTree)
        assert isinstance(result.root, MarginalNode)
        assert result.root.sumset == {"X", "Z"}
        assert result.to_latex() == r"\sum_{X, Z} P(X, Y, Z)"

        # No marginalization: should return ProbabilityNode directly
        result = ID()._marginal(frozenset({"X", "Y"}), frozenset({"X", "Y"}), None)
        assert isinstance(result, ProbabilityExpressionTree)
        assert isinstance(result.root, ProbabilityNode)
        assert result.root.variables == {"X", "Y"}

    def test_get_c_components(self):
        """Test _get_c_components() correctly identifies districts."""
        # Single C-component (all connected by bidirected edges)
        admg = ADMG(edge_list=[("X", "Y", "->"), ("X", "Y", "<>"), ("Y", "Z", "<>")])
        components = ID()._get_c_components(admg)
        assert len(components) == 1
        assert components[0] == frozenset({"X", "Y", "Z"})

        # Multiple C-components
        admg = ADMG(edge_list=[("X", "Y", "->"), ("X", "Y", "<>"), ("Z", "W", "<>")])
        components = ID()._get_c_components(admg)
        assert len(components) == 2
        assert frozenset({"X", "Y"}) in components
        assert frozenset({"Z", "W"}) in components

        # Isolated nodes (no bidirected edges)
        admg = ADMG(edge_list=[("X", "Y", "->")])
        components = ID()._get_c_components(admg)
        assert len(components) == 2
        assert frozenset({"X"}) in components
        assert frozenset({"Y"}) in components

    def test_identify_recursive(self):
        """Test _identify_recursive() core algorithm logic."""
        # Base case: no intervention
        admg = ADMG(edge_list=[("X", "Y", "->")], exposures=set(), outcomes={"Y"})
        result = ID()._identify_recursive(frozenset({"Y"}), frozenset(), frozenset({"X", "Y"}), admg, None)
        assert result is not False
        assert isinstance(result, ProbabilityExpressionTree)
        # Should return marginal P(Y) = ∑_X P(X,Y)
        assert isinstance(result.root, MarginalNode)

        # Single C-component - identifiable
        admg = ADMG(edge_list=[("X", "Y", "->")], exposures={"X"}, outcomes={"Y"})
        result = ID()._identify_recursive(frozenset({"Y"}), frozenset({"X"}), frozenset({"X", "Y"}), admg, None)
        assert result is not False

        # Hedge detection
        admg = ADMG(edge_list=[("X", "Y", "->"), ("X", "Y", "<>")], exposures={"X"}, outcomes={"Y"})
        id_algo = ID()
        result = id_algo._identify_recursive(frozenset({"Y"}), frozenset({"X"}), frozenset({"X", "Y"}), admg, None)
        assert result is False

    def test_decompose_by_c_components(self):
        """Test _decompose_by_c_components() decomposition logic."""
        admg = ADMG(edge_list=[("X", "Y", "->"), ("Z", "W", "->"), ("Z", "W", "<>")], exposures={"X"}, outcomes={"Y"})
        graph_minus_x = admg.get_subgraph(frozenset({"Y", "Z", "W"}), edge_types={"->", "<>"})
        c_components = ID()._get_c_components(graph_minus_x)

        result = ID()._decompose_by_c_components(
            frozenset({"Y"}), frozenset({"X"}), frozenset({"X", "Y", "Z", "W"}), admg, c_components, None
        )
        assert result is not False
        assert isinstance(result, ProbabilityExpressionTree)

    def test_compute_expression(self):
        """Test _compute_expression() creates correct factorization."""
        # Chain X -> Y -> Z, outcomes = {Z}
        admg = ADMG(edge_list=[("X", "Y", "->"), ("Y", "Z", "->")])
        result = ID()._compute_expression(
            frozenset({"Z"}), frozenset({"X", "Y", "Z"}), frozenset({"X", "Y", "Z"}), admg, None
        )
        assert isinstance(result, ProbabilityExpressionTree)
        assert isinstance(result.root, MarginalNode)
        assert result.root.sumset == {"X", "Y"}
        # Inner should be product P(X) P(Y|X) P(Z|Y)
        assert isinstance(result.root.children[0], ProductNode)

        # S equals outcomes - no marginalization
        admg = ADMG(edge_list=[("X", "Y", "->")])
        result = ID()._compute_expression(
            frozenset({"X", "Y"}), frozenset({"X", "Y"}), frozenset({"X", "Y"}), admg, None
        )
        assert isinstance(result.root, ProductNode)
        assert len(result.root.children) == 2


class TestIDCanonicalExamples:
    """Test canonical examples from ID algorithm literature.

    These are well-known identifiable and non-identifiable structures
    that verify the correctness of the ID algorithm implementation.
    """

    def test_front_door_adjustment(self):
        """Front-door criterion: X -> M -> Y with X <> Y.

        P(Y|do(X)) = ∑_M P(M|X) ∑_{X'} P(X') P(Y|M, X')
        """
        admg = ADMG(
            edge_list=[
                ("X", "M", "->"),  # X causes M
                ("M", "Y", "->"),  # M causes Y
                ("X", "Y", "<>"),  # X confounded with Y
            ],
            exposures={"X"},
            outcomes={"Y"},
        )

        result = ID().identify(admg)
        assert result is not False
        assert isinstance(result, ProbabilityExpressionTree)
        # Should marginalize over M
        assert isinstance(result.root, MarginalNode)
        assert result.root.sumset == {"M"}

    def test_mediator_confounded_with_outcome(self):
        """Mediator confounded with outcome: X -> M -> Y with M <> Y.

        This should be identifiable since there's no confounding between X and M.
        """
        admg = ADMG(
            edge_list=[
                ("X", "M", "->"),  # X causes M
                ("M", "Y", "->"),  # M causes Y
                ("M", "Y", "<>"),  # M confounded with Y
            ],
            exposures={"X"},
            outcomes={"Y"},
        )

        result = ID().identify(admg)
        assert result is not False
        assert isinstance(result, ProbabilityExpressionTree)

    def test_direct_confounding_hedge(self):
        """Direct confounding: X -> Y with X <> Y.

        Classic non-identifiable case - the simplest hedge structure.
        """
        admg = ADMG(
            edge_list=[
                ("X", "Y", "->"),  # X causes Y
                ("X", "Y", "<>"),  # X confounded with Y
            ],
            exposures={"X"},
            outcomes={"Y"},
        )

        id_algo = ID()
        result = id_algo.identify(admg)
        assert result is False
        # Should detect hedge
        assert id_algo.hedge_ is not None

    def test_bow_arc_non_identifiable(self):
        """Bow-arc: X -> Z -> Y with X <> Z.

        Treatment confounded with its child - creates a hedge.
        """
        admg = ADMG(
            edge_list=[
                ("X", "Z", "->"),  # X causes Z
                ("Z", "Y", "->"),  # Z causes Y
                ("X", "Z", "<>"),  # X confounded with Z
            ],
            exposures={"X"},
            outcomes={"Y"},
        )

        id_algo = ID()
        result = id_algo.identify(admg)
        assert result is False
        assert id_algo.hedge_ is not None

    def test_instrumental_variable(self):
        """IV structure: Z -> X -> Y with X <> Y.

        Note: IV is NOT identifiable by ID algorithm alone.
        Requires additional assumptions (linearity, exclusion restriction, etc.).
        """
        admg = ADMG(
            edge_list=[
                ("Z", "X", "->"),  # Z is instrument
                ("X", "Y", "->"),  # X causes Y
                ("X", "Y", "<>"),  # X confounded with Y
            ],
            exposures={"X"},
            outcomes={"Y"},
        )

        id_algo = ID()
        result = id_algo.identify(admg)
        assert result is False
        # Correctly identifies that effect is not identifiable without additional assumptions

    def test_multiple_interventions(self):
        """Multiple exposures and outcomes."""
        # Multiple exposures, single outcome
        admg = ADMG(edge_list=[("X1", "Y", "->"), ("X2", "Y", "->")], exposures={"X1", "X2"}, outcomes={"Y"})
        result = ID().identify(admg)
        assert result is not False
        assert isinstance(result, ProbabilityExpressionTree)

        # Single exposure, multiple outcomes
        admg = ADMG(edge_list=[("X", "Y1", "->"), ("X", "Y2", "->")], exposures={"X"}, outcomes={"Y1", "Y2"})
        result = ID().identify(admg)
        assert result is not False
        assert isinstance(result, ProbabilityExpressionTree)
