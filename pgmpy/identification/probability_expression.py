from abc import ABC, abstractmethod


class _TreeNode(ABC):
    """
    Private abstract base class for all expression tree nodes.

    Not part of the public API. Provides the shared interface that makes
    uniform tree traversal possible: every node exposes ``node_type``
    (str), ``children`` (list of _TreeNode), ``to_latex()`` (str), and
    ``__repr__()`` (str).

    Concrete subclasses: Prob, Marginal, Product, Division.

    Do not instantiate directly.
    """

    # Each subclass overrides this with a fixed string literal.
    node_type = ""

    # Each subclass sets self.children in __init__.
    children = []

    @abstractmethod
    def to_latex(self):
        """Return a LaTeX string representation of this node.

        Returns
        -------
        str
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        """Return an unambiguous string representation of this node.

        Returns
        -------
        str
        """
        raise NotImplementedError


class Prob(_TreeNode):
    """
    Leaf node of the expression tree.

    Represents an atomic conditional probability term::

        P(variables | do(do_vars), cond)

    with an optional inline marginalisation over ``sumset``::

        sum_{sumset} P(variables | do(do_vars), cond)

    Parameters
    ----------
    variables : frozenset of str
        The random variables in the probability term.
        Example: ``frozenset({"Y"})`` renders as ``P(Y | ...)``.

    do : frozenset of str, optional
        Variables being intervened on via the do-operator.
        Example: ``frozenset({"X"})`` renders as ``do(X)`` inside
        the conditioning bar.
        Default: ``frozenset()``.

    cond : frozenset of str, optional
        Variables being passively conditioned on.
        Example: ``frozenset({"Z"})`` renders after the conditioning bar.
        Default: ``frozenset()``.

    sumset : frozenset of str, optional
        Variables summed out directly at this node (inline marginalisation).
        Example: ``frozenset({"Z"})`` prepends ``\\sum_{Z}`` to the term.
        Default: ``frozenset()``.

    Attributes
    ----------
    node_type : str
        Always ``"prob"``.

    children : list
        Always ``[]``. ``Prob`` is always a leaf node.

    Examples
    --------
    >>> Prob(frozenset({"Y"})).to_latex()
    'P(Y)'

    >>> Prob(frozenset({"Y"}), cond=frozenset({"X"})).to_latex()
    'P(Y \\\\mid X)'

    >>> Prob(frozenset({"Y"}), do=frozenset({"X"})).to_latex()
    'P(Y \\\\mid do(X))'
    """

    node_type = "prob"

    def __init__(
        self,
        variables,
        do=frozenset(),
        cond=frozenset(),
        sumset=frozenset(),
    ):
        self.variables = frozenset(variables)
        self.do = frozenset(do)
        self.cond = frozenset(cond)
        self.sumset = frozenset(sumset)
        self.children = []

    @staticmethod
    def _vars_to_latex(var_set):
        """Sort and join variable names for deterministic LaTeX output."""
        return ", ".join(sorted(var_set))

    def to_latex(self):
        """
        Return LaTeX for this atomic probability term.

        The conditioning bar is omitted when both ``do`` and ``cond`` are
        empty.  ``do(·)`` is always rendered before passive conditioning.
        The ``sumset`` prefix is prepended when non-empty.

        Returns
        -------
        str

        Examples
        --------
        >>> Prob(frozenset({"Y"}), do=frozenset({"X"}), cond=frozenset({"Z"})).to_latex()
        'P(Y \\\\mid do(X), Z)'
        """
        variables_str = self._vars_to_latex(self.variables)

        conditioning_parts = []
        if self.do:
            conditioning_parts.append("do(" + self._vars_to_latex(self.do) + ")")
        if self.cond:
            conditioning_parts.append(self._vars_to_latex(self.cond))

        if conditioning_parts:
            inner = variables_str + r" \mid " + ", ".join(conditioning_parts)
        else:
            inner = variables_str

        prob_str = "P(" + inner + ")"

        if self.sumset:
            return r"\sum_{" + self._vars_to_latex(self.sumset) + "} " + prob_str
        return prob_str

    def __repr__(self):
        parts = [f"variables={set(self.variables)!r}"]
        if self.do:
            parts.append(f"do={set(self.do)!r}")
        if self.cond:
            parts.append(f"cond={set(self.cond)!r}")
        if self.sumset:
            parts.append(f"sumset={set(self.sumset)!r}")
        return "Prob(" + ", ".join(parts) + ")"


class Marginal(_TreeNode):
    """
    Internal node representing a marginalisation over a sub-expression::

        sum_{sumset} child

    Parameters
    ----------
    child : Prob or Marginal or Product or Division
        The expression being summed over. Stored as ``children[0]``.

    sumset : frozenset of str
        The variables being summed out.

    Attributes
    ----------
    node_type : str
        Always ``"sum"``.

    children : list of length 1
        ``children[0]`` is the child expression.

    Examples
    --------
    >>> m = Marginal(
    ...     Prob(frozenset({"Y"}), cond=frozenset({"X"})),
    ...     sumset=frozenset({"X"}),
    ... )
    >>> m.to_latex()
    '\\\\sum_{X} P(Y \\\\mid X)'
    """

    node_type = "sum"

    def __init__(self, child, sumset):
        self.sumset = frozenset(sumset)
        self.children = [child]

    def to_latex(self):
        """
        Return LaTeX for ``\\sum_{sumset} child``.

        Calls ``child.to_latex()`` to render the inner expression,
        then prepends the summation symbol and sumset variables.

        Returns
        -------
        str
        """
        sumset_str = ", ".join(sorted(self.sumset))
        child_latex = self.children[0].to_latex()
        return r"\sum_{" + sumset_str + "} " + child_latex

    def __repr__(self):
        return f"Marginal(child={self.children[0]!r}, sumset={set(self.sumset)!r})"


class Product(_TreeNode):
    """
    Internal node representing a product of two or more sub-expressions::

        factors[0] * factors[1] * ... * factors[n]

    Parameters
    ----------
    factors : list of (Prob or Marginal or Product or Division)
        The expressions being multiplied. Must have at least two elements.

    Raises
    ------
    ValueError
        If ``factors`` has fewer than two elements.

    Attributes
    ----------
    node_type : str
        Always ``"product"``.

    children : list of length >= 2
        The factor nodes in the order given.

    Examples
    --------
    >>> Product([
    ...     Prob(frozenset({"Y"}), cond=frozenset({"X"})),
    ...     Prob(frozenset({"X"})),
    ... ]).to_latex()
    'P(Y \\\\mid X) P(X)'
    """

    node_type = "product"

    def __init__(self, factors):
        if len(factors) < 2:
            raise ValueError(f"Product requires at least two factors; got {len(factors)}.")
        self.children = list(factors)

    def to_latex(self):
        r"""
        Return LaTeX for the product of all children, space-separated.

        Composite children (Marginal, Product, Division) are wrapped in
        ``\left[ ... \right]`` to make nested grouping unambiguous.

        Returns
        -------
        str
        """
        parts = []
        for child in self.children:
            child_latex = child.to_latex()
            if isinstance(child, (Marginal, Product, Division)):
                parts.append(r"\left[ " + child_latex + r" \right]")
            else:
                parts.append(child_latex)
        return " ".join(parts)

    def __repr__(self):
        return "Product([" + ", ".join(repr(c) for c in self.children) + "])"


class Division(_TreeNode):
    """
    Internal node representing a ratio of two sub-expressions::

        numerator / denominator

    Parameters
    ----------
    numerator : Prob or Marginal or Product or Division
        The numerator expression. Stored as ``children[0]``.

    denominator : Prob or Marginal or Product or Division
        The denominator expression. Stored as ``children[1]``.

    Attributes
    ----------
    node_type : str
        Always ``"division"``.

    children : list of length 2
        ``children[0]`` is the numerator; ``children[1]`` is the denominator.

    Examples
    --------
    >>> Division(
    ...     Prob(frozenset({"Y"}), do=frozenset({"X"})),
    ...     Prob(frozenset({"Z"}), do=frozenset({"X"})),
    ... ).to_latex()
    '\\\\frac{P(Y \\\\mid do(X))}{P(Z \\\\mid do(X))}'
    """

    node_type = "division"

    def __init__(self, numerator, denominator):
        self.children = [numerator, denominator]

    def to_latex(self):
        r"""
        Return LaTeX for ``\frac{numerator}{denominator}``.

        Returns
        -------
        str
        """
        num_latex = self.children[0].to_latex()
        den_latex = self.children[1].to_latex()
        return r"\frac{" + num_latex + "}{" + den_latex + "}"

    def __repr__(self):
        return f"Division(numerator={self.children[0]!r}, denominator={self.children[1]!r})"


class ProbabilityExpressionTree:
    """
    Container for expression trees produced by identification algorithms.

    `ProbabilityExpressionTree` is a lightweight container that holds the root
    of a symbolic probability expression tree. It is the public object
    returned by causal identification routines and is not itself a tree
    node.

    The expression tree is composed of `Prob`, `Marginal`, `Product`, and
    `Division` nodes (all subclasses of `_TreeNode`). Inspect or traverse the
    tree via the `root` attribute and the `children` lists on nodes.

    Parameters
    ----------
    root : Prob | Marginal | Product | Division
        Root node of the expression tree. Must be an instance of `_TreeNode`.

    Raises
    ------
    TypeError
        If `root` is not an instance of `_TreeNode`.

    Attributes
    ----------
    root : Prob | Marginal | Product | Division
        The expression tree root. Sub-nodes are accessible via
        `root.children`.

    Notes
    -----
    The `to_latex()` method delegates to `self.root.to_latex()` and
    returns a complete LaTeX representation suitable for math
    environments. Rendering proceeds bottom-up: leaf `Prob` nodes render
    themselves and internal nodes combine their children's strings using
    the appropriate notation (e.g. `\\sum`, `\frac`, brackets).

    Examples
    --------
    Build the bow-arc formula
    ``\\sum_Z [ P(Z|X) \\cdot \\sum_{X'} [ P(Y|Z,X') P(X') ] ]``:

    >>> inner = Product([
    ...     Prob(frozenset({"Y"}), cond=frozenset({"Z", "X"})),
    ...     Prob(frozenset({"X"})),
    ... ])
    >>> expr = ProbabilityExpressionTree(root=Marginal(
    ...     Product([
    ...         Prob(frozenset({"Z"}), cond=frozenset({"X"})),
    ...         Marginal(inner, sumset=frozenset({"X"})),
    ...     ]),
    ...     sumset=frozenset({"Z"}),
    ... ))
    >>> expr.to_latex()
    '\\\\sum_{Z} P(Z \\\\mid X) \\\\left[ \\\\sum_{X} P(Y \\\\mid X, Z) P(X) \\\\right]'
    >>> expr.root.node_type
    'sum'
    >>> expr.root.children[0].node_type
    'product'
    """

    def __init__(self, root):
        if not isinstance(root, _TreeNode):
            raise TypeError(
                f"root must be a _TreeNode instance (one of Prob, Marginal, "
                f"Product, Division); got {type(root).__name__}."
            )
        self.root = root

    def to_latex(self):
        """
        Return a LaTeX string for the full causal expression.

        Starts the recursive ``to_latex()`` traversal from the root node.
        Each node in the tree renders itself and delegates to its children,
        so the complete formula is assembled bottom-up in a single call.

        Returns
        -------
        str
            A LaTeX string for the full expression, ready for a math
            environment.

        Examples
        --------
        >>> expr = ProbabilityExpressionTree(
        ...     root=Prob(frozenset({"Y"}), cond=frozenset({"X"}))
        ... )
        >>> expr.to_latex()
        'P(Y \\\\mid X)'
        """
        return self.root.to_latex()

    def __repr__(self):
        return repr(self.root)
