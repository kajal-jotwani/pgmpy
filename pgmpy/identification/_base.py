class BaseIdentification:
    """Base class for all identification methods.

    All identification methods in pgmpy must inherit `BaseIdentification`.
    Inheriting methods need to define the `_identify` method, which implements
    the specific identification algorithm. The `_identify` method should take a
    causal graph as input and return a modified version of the graph with
    variable roles assigned, along with a boolean indicating whether the
    identification was successful.

    Examples
    --------
    >>> from pgmpy.identification import BaseIdentification
    >>> class SimpleId(BaseIdentification):
    ...     "A simple identification method when all variable are observed"
    ...
    ...     def _identify(self, causal_graph):
    ...         outcome_parents = causal_graph.predecessors(
    ...             causal_graph.get_role("exposures")
    ...         )
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

        This method applies the identification procedure to the input causal
        graph, annotating it with variable roles (e.g., adjustment, IVs) while
        keeping the original graphical structure.

        Parameters
        ----------
        causal_graph : DAG, PDAG, ADMG, MAG, or PAG object
            The input causal graph on which to perform identification. The
            causal graph must have variables with exposures and outcomes roles
            defined.

        Returns
        -------
        identified_graph : DAG, PDAG, ADMG, MAG, or PAG object
            A new causal graph instance with variable roles assigned according
            to the identification method.

        success : bool
            True if the exposures and outcomes are successfully identified; False
            otherwise.
        """
        self._validate_causal_graph(causal_graph)
        return self._identify(causal_graph)

    def validate(self, causal_graph):
        """
        Validate the input causal graph for identification.

        This method checks if the variable roles assigned in the `causal_graph`
        are appropriate for identification. For example, given a causal graph
        with exposures, outcomes, and adjustment roles, it verifies that the
        adjustment set is valid for the given exposures and outcomes.

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
    """Base class for identification methods that return formulas.

    Subclasses should define ``supported_graph_types`` and implement
    ``_identify``. The ``_identify`` method must return a
    ``ProbabilityExpressionTree`` when the effect is identifiable, or
    ``False`` otherwise. If identification fails, subclasses should set
    ``self.hedge_`` to the witness subgraph.

    Parameters
    ----------
    causal_graph : ADMG or DAG
        The causal graph with the required roles assigned. Subclasses may
        require additional roles through ``required_roles``.

    Returns
    -------
    ProbabilityExpressionTree
        The symbolic formula for the identified causal effect.

    False
        If the causal effect is not identifiable.

    Examples
    --------
    >>> from pgmpy.identification import BaseFormulaIdentification
    >>> from pgmpy.identification.probability_expression import ProbabilityExpressionTree, ProbabilityNode
    >>> class SimpleFormulaId(BaseFormulaIdentification):
    ...     def _identify(self, causal_graph):
    ...         y = causal_graph.get_role("outcomes")
    ...         x = causal_graph.get_role("exposures")
    ...         return ProbabilityExpressionTree(
    ...             root=ProbabilityNode(frozenset(y), cond=frozenset(x))
    ...         )
    """

    supported_graph_types = ()

    required_roles = ("exposures", "outcomes")

    def _validate_query(self, causal_graph):
        if not isinstance(causal_graph, self.supported_graph_types):
            raise ValueError(
                f"The `causal_graph` must be an instance of "
                f"{self.supported_graph_types} for this method. "
                f"Got {type(causal_graph).__name__}."
            )

        causal_graph.is_valid_causal_structure()

        missing_roles = [role for role in self.required_roles if not causal_graph.has_role(role)]
        if missing_roles:
            raise ValueError(f"causal_graph is missing required role(s): {missing_roles}.")

    def identify(self, causal_graph):
        """
        Run the identification algorithm on a causal graph.

        Parameters
        ----------
        causal_graph : ADMG or DAG
            The causal graph with at minimum `exposures` and `outcomes`
            roles assigned. Subclasses may require additional roles via
            `required_roles`.

        Returns
        -------
        ProbabilityExpressionTree
            The symbolic formula for the identified causal effect. Access
            the expression tree via `result.root`.

        False
            If the causal effect is not identifiable. The witness subgraph
            is stored in `self.hedge_`.
        """
        self._validate_query(causal_graph)
        return self._identify(causal_graph)

    def _identify(self, causal_graph):
        """Override in subclasses to implement the algorithm."""
        raise NotImplementedError

    def __call__(self, causal_graph):
        """Alias for the `identify` method."""
        return self.identify(causal_graph)
