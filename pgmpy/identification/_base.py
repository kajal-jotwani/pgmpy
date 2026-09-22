class BaseGraphicalIdentification:
    """Base class for all identification methods.

    All identification methods in pgmpy must inherit `BaseGraphicalIdentification`. Inheriting methods need to define
    the `_identify` method, which implements the specific identification algorithm. The `_identify` method should take
    a causal graph as input and return a modified version of the graph with variable roles assigned, along with a
    boolean indicating whether the identification was successful.

    Examples
    --------
    >>> from pgmpy.identification import BaseGraphicalIdentification
    >>> class SimpleId(BaseGraphicalIdentification):
    ...     "A simple identification method when all variable are observed"
    ...
    ...     def _identify(self, causal_graph):
    ...         outcome_parents = causal_graph.predecessors(causal_graph.get_role("exposures"))
    ...         identified_cg = causal_graph.with_role("adjustment", outcome_parents)
    ...         return identified_cg, True
    ...
    """

    def _validate_causal_graph(self, causal_graph):
        # Check if the passed causal_graph is supported by the method.
        if not isinstance(causal_graph, self.supported_graph_types):
            raise ValueError(f"The `causal_graph` must be an instance of {self.supported_graph_types} for this method.")

        # Check if causal_graph has `exposures` and `outcomes` roles assigned.
        causal_graph.is_valid_causal_structure()

    def identify(self, causal_graph):
        """
        Run the identification algorithm on a causal graph.

        This method applies the identification procedure to the input causal graph, annotating it with variable roles
        (e.g., adjustment, IVs) while keeping the original graphical structure.

        Parameters
        ----------
        causal_graph : DAG, PDAG, ADMG, MAG, or PAG object
            The input causal graph on which to perform identification. The causal graph must have variables with
            exposures and outcomes roles defined.

        Returns
        -------
        identified_graph : DAG, PDAG, ADMG, MAG, or PAG object
            A new causal graph instance with variable roles assigned according to the identification method.

        success : bool
            True if the exposures and outcomes are successfully identified; False otherwise.
        """
        self._validate_causal_graph(causal_graph)
        return self._identify(causal_graph)

    def validate(self, causal_graph):
        """
        Validate the input causal graph for identification.

        This method checks if the variable roles assigned in the `causal_graph` are appropriate for identification. For
        example, given a causal graph with exposures, outcomes, and adjustment roles, it verifies that the adjustment
        set is valid for the given exposures and outcomes.

        Parameters
        ----------
        causal_graph : DAG, PDAG, ADMG, MAG, or PAG object
            The input causal graph to validate.

        Returns
        -------
        bool:
            True if the graph is valid for identification; False otherwise.
        """
        self._validate_causal_graph(causal_graph)
        return self._validate(causal_graph)

    def __call__(self, causal_graph):
        """Alias for the `identify` method"""
        return self.identify(causal_graph)


class BaseFormulaIdentification:
    """Base class for identification methods that return a symbolic expression.

    Subclasses should define ``supported_graph_types`` and implement ``_identify``. The ``_identify`` method must
    return a ``ProbabilityExpressionTree`` when the query is identifiable, or ``False`` otherwise. If identification
    fails, subclasses should set ``self.hedge_`` to the witness subgraph.

    A query is not always fully described by the graph. ``ID`` and ``IDC`` read theirs off the node roles
    (``exposures``, ``outcomes``, ``conditioning``), which ``required_roles`` declares; ``IDStar`` is given a
    conjunction of counterfactual events instead, which no role can carry. Whatever keyword arguments ``identify``
    is called with are therefore forwarded verbatim to ``_identify``, and ``required_roles`` may be empty.

    Parameters
    ----------
    causal_graph : ADMG or DAG
        The causal graph with the roles listed in ``required_roles`` assigned.

    **query
        The part of the query that is not carried by the graph, forwarded to ``_identify``.

    Returns
    -------
    ProbabilityExpressionTree
        The symbolic formula for the identified query.

    False
        If the query is not identifiable. The witness subgraph is stored in ``self.hedge_``.

    Examples
    --------
    >>> from pgmpy.identification import BaseFormulaIdentification
    >>> from pgmpy.identification.probability_expression import ProbabilityExpressionTree, ProbabilityNode
    >>> from pgmpy.base import DAG
    >>> class SimpleFormulaId(BaseFormulaIdentification):
    ...     supported_graph_types = (DAG,)
    ...     def _identify(self, causal_graph):
    ...         y = causal_graph.get_role("outcomes")
    ...         x = causal_graph.get_role("exposures")
    ...         return ProbabilityExpressionTree(root=ProbabilityNode(frozenset(y), cond=frozenset(x)))
    """

    supported_graph_types = ()
    required_roles = ("exposures", "outcomes")

    def _validate_causal_graph(self, causal_graph):
        """Validate the causal graph before running identification.

        Checks that:

        1. ``causal_graph`` is an instance of one of ``supported_graph_types``.
        2. Every role listed in ``required_roles`` is assigned.

        Parameters
        ----------
        causal_graph : ADMG or DAG
            The causal graph with the roles listed in `required_roles` assigned.

        Raises
        ------
        ValueError
            If the graph type is not supported, or if a required role is missing.
        """
        if not isinstance(causal_graph, self.supported_graph_types):
            raise ValueError(
                f"causal_graph must be an instance of {self.supported_graph_types} for this method. "
                f"Got {type(causal_graph).__name__}."
            )

        # The roles the subclass reads its query off: "exposures" and "outcomes" for ID, those two plus
        # "conditioning" for IDC, and none at all for the counterfactual algorithms, whose query is passed to
        # `identify` instead.
        for role in self.required_roles:
            if not causal_graph.get_role(role):
                raise ValueError(f"causal_graph must have '{role}' role assigned for {type(self).__name__}.")

    def identify(self, causal_graph, **query):
        """
        Run the identification algorithm on a causal graph.

        Validates the graph via ``_validate_causal_graph``, resets ``self.hedge_`` to ``None``, then delegates to
        ``_identify``.

        Parameters
        ----------
        causal_graph : ADMG or DAG
            The causal graph with the roles listed in `required_roles` assigned.

        **query
            The part of the query that is not carried by the graph, forwarded to ``_identify``. ``ID`` and ``IDC``
            take none; ``IDStar`` takes the conjunction of counterfactual events as ``event``.

        Returns
        -------
        ProbabilityExpressionTree
            The symbolic formula for the identified query. Access the expression tree via ``result.root``.

        False
            If the query is not identifiable. The witness subgraph is stored in ``self.hedge_``.
        """
        self._validate_causal_graph(causal_graph)
        self.hedge_ = None
        return self._identify(causal_graph, **query)

    def _identify(self, causal_graph, **query):
        """Override in subclasses to implement the identification algorithm.

        Parameters
        ----------
        causal_graph : ADMG or DAG
            The causal graph with the roles listed in `required_roles` assigned.

        **query
            The part of the query that is not carried by the graph.

        Returns
        -------
        ProbabilityExpressionTree or False
        """
        raise NotImplementedError

    def __call__(self, causal_graph, **query):
        """Alias for the ``identify`` method."""
        return self.identify(causal_graph, **query)
