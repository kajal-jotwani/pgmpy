import pytest

from pgmpy.identification.probability_expression import (
    Division,
    Marginal,
    Prob,
    ProbabilityExpressionTree,
    Product,
    _TreeNode,
)


@pytest.fixture
def prob_y():
    return Prob(frozenset({"Y"}))


@pytest.fixture
def prob_x():
    return Prob(frozenset({"X"}))


@pytest.fixture
def prob_y_cond_x():
    return Prob(frozenset({"Y"}), cond=frozenset({"X"}))


@pytest.fixture
def frontdoor_expr():
    """
    ProbabilityExpressionTree for the frontdoor formula:
        sum_M [ P(M|X) * sum_{X'} [ P(Y|M,X') P(X') ] ]
    """
    inner_product = Product(
        [
            Prob(frozenset({"Y"}), cond=frozenset({"M", "X"})),
            Prob(frozenset({"X"})),
        ]
    )
    inner_sum = Marginal(inner_product, sumset=frozenset({"X"}))
    outer_product = Product(
        [
            Prob(frozenset({"M"}), cond=frozenset({"X"})),
            inner_sum,
        ]
    )
    return ProbabilityExpressionTree(root=Marginal(outer_product, sumset=frozenset({"M"})))


@pytest.fixture
def bowarc_expr():
    """
    ProbabilityExpressionTree for the bow-arc formula:
        sum_Z [ P(Z|X) * sum_{X'} [ P(Y|Z,X') P(X') ] ]
    """
    inner_product = Product(
        [
            Prob(frozenset({"Y"}), cond=frozenset({"Z", "X"})),
            Prob(frozenset({"X"})),
        ]
    )
    inner_sum = Marginal(inner_product, sumset=frozenset({"X"}))
    outer_product = Product(
        [
            Prob(frozenset({"Z"}), cond=frozenset({"X"})),
            inner_sum,
        ]
    )
    return ProbabilityExpressionTree(root=Marginal(outer_product, sumset=frozenset({"Z"})))


def collect_node_types(node):
    types = [node.node_type]
    for child in node.children:
        types.extend(collect_node_types(child))
    return types


def find_leaves(node):
    if node.node_type == "prob":
        return [node]
    leaves = []
    for child in node.children:
        leaves.extend(find_leaves(child))
    return leaves


class TestTreeNodeAbstract:
    def test_concrete_subclasses_are_tree_nodes(self, prob_y, prob_x):
        assert isinstance(prob_y, _TreeNode)
        assert isinstance(Marginal(prob_y, sumset=frozenset({"Y"})), _TreeNode)
        assert isinstance(Product([prob_y, prob_x]), _TreeNode)
        assert isinstance(Division(prob_y, prob_x), _TreeNode)


class TestProbInit:
    def test_init(self):
        p = Prob(
            frozenset({"Y"}),
            do=frozenset({"X"}),
            cond=frozenset({"Z"}),
            sumset=frozenset({"W"}),
        )
        assert p.node_type == "prob"
        assert p.children == []
        assert p.variables == frozenset({"Y"})
        assert p.do == frozenset({"X"})
        assert p.cond == frozenset({"Z"})
        assert p.sumset == frozenset({"W"})

        p_defaults = Prob(frozenset({"Y"}))
        assert p_defaults.do == frozenset()
        assert p_defaults.cond == frozenset()
        assert p_defaults.sumset == frozenset()

        p_coerced = Prob(["Y"], do=["X"], cond=["Z"], sumset=["W"])
        assert isinstance(p_coerced.variables, frozenset)
        assert isinstance(p_coerced.do, frozenset)
        assert isinstance(p_coerced.cond, frozenset)
        assert isinstance(p_coerced.sumset, frozenset)


class TestProbToLatex:
    def test_to_latex(self):
        assert Prob(frozenset({"Y"})).to_latex() == "P(Y)"
        assert Prob(frozenset({"Y"}), cond=frozenset({"X"})).to_latex() == r"P(Y \mid X)"
        assert Prob(frozenset({"Y"}), do=frozenset({"X"})).to_latex() == r"P(Y \mid do(X))"
        assert Prob(frozenset({"Y"}), do=frozenset({"X"}), cond=frozenset({"Z"})).to_latex() == r"P(Y \mid do(X), Z)"
        assert Prob(frozenset({"Z", "Y"})).to_latex() == "P(Y, Z)"
        assert Prob(frozenset({"Y"}), cond=frozenset({"Z", "X"})).to_latex() == r"P(Y \mid X, Z)"
        latex = Prob(frozenset({"Y", "Z"}), sumset=frozenset({"Z"})).to_latex()
        assert latex == r"\sum_{Z} P(Y, Z)"
        latex_multi = Prob(frozenset({"X", "Y", "Z"}), sumset=frozenset({"X", "Z"})).to_latex()
        assert latex_multi.startswith(r"\sum_{X, Z}")


class TestMarginalInit:
    def test_init(self, prob_y):
        m = Marginal(prob_y, sumset=frozenset({"Y"}))
        assert m.node_type == "sum"
        assert len(m.children) == 1
        assert m.children[0] is prob_y
        assert isinstance(m.sumset, frozenset)
        assert m.sumset == frozenset({"Y"})


class TestMarginalToLatex:
    def test_to_latex(self):
        inner = Prob(frozenset({"Y"}), cond=frozenset({"X"}))
        assert Marginal(inner, sumset=frozenset({"X"})).to_latex() == r"\sum_{X} P(Y \mid X)"

        inner2 = Prob(frozenset({"Y"}))
        latex = Marginal(inner2, sumset=frozenset({"Z", "X"})).to_latex()
        assert latex.startswith(r"\sum_{X, Z}")


class TestProductInit:
    def test_init(self, prob_y, prob_x, prob_y_cond_x):
        prod = Product([prob_y, prob_x])
        assert prod.node_type == "product"
        assert len(prod.children) == 2
        assert prod.children[0] is prob_y
        assert prod.children[1] is prob_x

        assert len(Product([prob_y, prob_x, prob_y_cond_x]).children) == 3

        with pytest.raises(ValueError, match="at least two factors"):
            Product([prob_y])

        with pytest.raises(ValueError, match="at least two factors"):
            Product([])


class TestProductToLatex:
    def test_to_latex(self, prob_y, prob_x):
        p1 = Prob(frozenset({"Y"}), cond=frozenset({"X"}))
        p2 = Prob(frozenset({"X"}))
        assert Product([p1, p2]).to_latex() == r"P(Y \mid X) P(X)"

        assert r"\left[" not in Product([prob_y, prob_x]).to_latex()

        inner = Marginal(Prob(frozenset({"Y"})), sumset=frozenset({"Y"}))
        latex = Product([Prob(frozenset({"X"})), inner]).to_latex()
        assert r"\left[" in latex
        assert r"\right]" in latex


class TestDivisionInit:
    def test_init(self, prob_y, prob_x):
        d = Division(prob_y, prob_x)
        assert d.node_type == "division"
        assert len(d.children) == 2
        assert d.children[0] is prob_y
        assert d.children[1] is prob_x


class TestDivisionToLatex:
    def test_to_latex(self):
        num = Prob(frozenset({"Y"}), do=frozenset({"X"}))
        den = Prob(frozenset({"Z"}), do=frozenset({"X"}))
        assert Division(num, den).to_latex() == r"\frac{P(Y \mid do(X))}{P(Z \mid do(X))}"

        num2 = Prob(frozenset({"Y", "Z"}), do=frozenset({"X"}))
        den2 = Prob(frozenset({"Z"}), do=frozenset({"X"}))
        latex = Division(num2, den2).to_latex()
        assert latex.startswith(r"\frac{")
        assert r"P(Y, Z \mid do(X))" in latex
        assert r"P(Z \mid do(X))" in latex


class TestProbabilityExpressionTreeInit:
    def test_init(self, prob_y):
        expr = ProbabilityExpressionTree(root=prob_y)
        assert expr.root is prob_y

        with pytest.raises(TypeError, match="_TreeNode"):
            ProbabilityExpressionTree(root="not a node")

        with pytest.raises(TypeError, match="_TreeNode"):
            ProbabilityExpressionTree(root=42)

        with pytest.raises(TypeError, match="_TreeNode"):
            ProbabilityExpressionTree(root=None)


class TestProbabilityExpressionTreeToLatex:
    def test_to_latex(self, prob_y_cond_x):
        expr = ProbabilityExpressionTree(root=prob_y_cond_x)
        assert expr.to_latex() == prob_y_cond_x.to_latex()
        assert repr(expr) == repr(prob_y_cond_x)


class TestTreeTraversal:
    def test_collect_node_types(self, frontdoor_expr, bowarc_expr, prob_y):
        assert collect_node_types(frontdoor_expr.root) == ["sum", "product", "prob", "sum", "product", "prob", "prob"]
        assert collect_node_types(bowarc_expr.root) == ["sum", "product", "prob", "sum", "product", "prob", "prob"]
        assert collect_node_types(prob_y) == ["prob"]

    def test_find_leaves(self, frontdoor_expr, bowarc_expr, prob_y, prob_x):
        assert all(leaf.children == [] for leaf in find_leaves(frontdoor_expr.root))
        assert len(find_leaves(frontdoor_expr.root)) == 3
        assert len(find_leaves(bowarc_expr.root)) == 3

        d = Division(prob_y, prob_x)
        assert len(d.children) == 2
        assert d.children[0] is prob_y
        assert d.children[1] is prob_x


class TestFrontdoorLatex:
    """
    Frontdoor formula: sum_M [ P(M|X) * sum_{X'} [ P(Y|M,X') P(X') ] ]

    P(Y|do(X)) in a model with mediator M and hidden confounder between X and Y.
    Reference: Pearl (2009), Causality, Example 3.3.2.
    """

    def test_to_latex(self, frontdoor_expr):
        latex = frontdoor_expr.to_latex()
        assert r"\sum_{M}" in latex
        assert r"\sum_{X}" in latex
        assert r"P(M \mid X)" in latex
        assert "P(X)" in latex
        assert r"P(Y \mid M, X)" in latex
        assert r"\left[" in latex
        assert r"\right]" in latex

    def test_tree_structure(self, frontdoor_expr):
        assert frontdoor_expr.root.node_type == "sum"
        assert frontdoor_expr.root.sumset == frozenset({"M"})


class TestBowArcLatex:
    """
    Bow-arc formula: sum_Z [ P(Z|X) * sum_{X'} [ P(Y|Z,X') P(X') ] ]

    P(Y|do(X)) for the bow-arc graph (X->Z->Y, X<->Z). Role-based methods
    cannot derive this; it requires the ID algorithm.
    Reference: Shpitser & Pearl (2006), AAAI-06.
    """

    def test_to_latex(self, bowarc_expr):
        latex = bowarc_expr.to_latex()
        assert r"\sum_{Z}" in latex
        assert r"\sum_{X}" in latex
        assert r"P(Z \mid X)" in latex
        assert "P(X)" in latex
        assert r"P(Y \mid X, Z)" in latex
        assert r"\left[" in latex
        assert r"\right]" in latex

    def test_tree_structure(self, bowarc_expr):
        assert bowarc_expr.root.node_type == "sum"
        assert bowarc_expr.root.sumset == frozenset({"Z"})
