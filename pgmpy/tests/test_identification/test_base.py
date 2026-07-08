import pytest

from pgmpy.base import ADMG, DAG, PDAG
from pgmpy.identification import BaseFormulaIdentification, BaseIdentification


@pytest.fixture
def cg():
    edges = [("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")]
    roles = {"exposures": "X", "outcomes": "Y"}
    return DAG(ebunch=edges, roles=roles)


class DummyIdentification(BaseIdentification):
    """Sorts non-exposure and non-outcome nodes in the graph and assigns the
    first or the last one as adjustment node depending on the `variant`.
    """

    def __init__(self, variant=None):
        self.variant = variant
        self.supported_graph_types = (DAG, PDAG)

    def _identify(self, causal_graph):
        if self.variant == "first":
            adjustment_node = sorted(
                set(causal_graph.nodes()) - set(causal_graph.get_role("exposures") + causal_graph.get_role("outcomes"))
            )[0]
            return causal_graph.with_role("adjustment", [adjustment_node]), True
        elif self.variant == "last":
            adjustment_node = sorted(
                set(causal_graph.nodes()) - set(causal_graph.get_role("exposures") + causal_graph.get_role("outcomes"))
            )[-1]
            return causal_graph.with_role("adjustment", [adjustment_node]), True
        else:
            return causal_graph, False


class TestBaseIdentification:
    def test_base_identification_first(self, cg):
        identifier = DummyIdentification(variant="first")
        identified_cg, is_identified = identifier(causal_graph=cg)

        assert is_identified == True
        assert identified_cg.get_role_dict() == {
            "exposures": ["X"],
            "outcomes": ["Y"],
            "adjustment": ["M"],
        }

    def test_base_identification_last(self, cg):
        identifier = DummyIdentification(variant="last")
        identified_cg, is_identified = identifier(causal_graph=cg)

        assert is_identified == True
        assert identified_cg.get_role_dict() == {
            "exposures": ["X"],
            "outcomes": ["Y"],
            "adjustment": ["U"],
        }

    def test_base_identification_gibberish(self, cg):
        identifier = DummyIdentification(variant="gibberish")
        identified_cg, is_identified = identifier(causal_graph=cg)

        assert is_identified == False
        assert identified_cg.get_role_dict() == {"exposures": ["X"], "outcomes": ["Y"]}


@pytest.fixture
def admg_bowarc():
    return ADMG(
        directed_ebunch=[("X", "Z"), ("Z", "Y")],
        bidirected_ebunch=[("X", "Z")],
        roles={"exposures": "X", "outcomes": "Y"},
    )


@pytest.fixture
def admg_no_roles():
    return ADMG(directed_ebunch=[("X", "Z"), ("Z", "Y")], bidirected_ebunch=[("X", "Z")])


@pytest.fixture
def admg_with_conditioning():
    return ADMG(
        directed_ebunch=[("X", "Z"), ("Z", "Y"), ("X", "Y")],
        bidirected_ebunch=[("X", "Y")],
        roles={"exposures": "X", "outcomes": "Y", "conditioning": "Z"},
    )


class DummyFormulaIdentification(BaseFormulaIdentification):
    """Returns a fixed string instead of a ProbabilityExpressionTree, so the
    base class contract can be tested without depending on the ID algorithm.
    """

    supported_graph_types = (ADMG, DAG)

    def _identify(self, causal_graph):
        return "identified"


class DummyConditionalFormulaIdentification(BaseFormulaIdentification):
    """Same as DummyFormulaIdentification but also requires a 'conditioning'
    role, mirroring IDC.
    """

    supported_graph_types = (ADMG, DAG)
    required_roles = ("exposures", "outcomes", "conditioning")

    def _identify(self, causal_graph):
        return "identified"


class IncompleteFormulaIdentification(BaseFormulaIdentification):
    """Does not override _identify, used to check the default raises
    NotImplementedError.
    """

    supported_graph_types = (ADMG, DAG)


class TestBaseFormulaIdentification:
    def test_identify(self, admg_bowarc):
        identifier = DummyFormulaIdentification()
        assert identifier.identify(admg_bowarc) == "identified"
        assert identifier(admg_bowarc) == "identified"

    def test_identify_wrong_graph_type(self, admg_bowarc):
        class DAGOnly(BaseFormulaIdentification):
            supported_graph_types = (DAG,)

            def _identify(self, causal_graph):
                return "ok"

        with pytest.raises(ValueError, match="must be an instance of"):
            DAGOnly().identify(admg_bowarc)

    def test_identify_missing_exposures_outcomes(self, admg_no_roles):
        with pytest.raises(ValueError, match="exposures"):
            DummyFormulaIdentification().identify(admg_no_roles)

    def test_identify_missing_extra_required_role(self, admg_bowarc):
        with pytest.raises(ValueError, match="conditioning"):
            DummyConditionalFormulaIdentification().identify(admg_bowarc)

    def test_identify_with_extra_required_role_present(self, admg_with_conditioning):
        assert DummyConditionalFormulaIdentification().identify(admg_with_conditioning) == "identified"

    def test_identify_not_implemented(self, admg_bowarc):
        with pytest.raises(NotImplementedError):
            IncompleteFormulaIdentification().identify(admg_bowarc)
