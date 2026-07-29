from pgmpy.base import ADMG, DAG
from pgmpy.identification import BaseFormulaIdentification
from pgmpy.identification.probability_expression import (
    MarginalNode,
    ProbabilityExpressionTree,
    ProbabilityNode,
    ProductNode,
)


class ID(BaseFormulaIdentification):
    """
    Given a causal graph, identifies the causal effect using the ID algorithm.

    The class implements the recursive identification procedure of
    :footcite:t:`shpitser_2006` for ADMGs with exposure and outcome roles.
    When the effect is identifiable, it returns a symbolic probability
    expression for the interventional distribution. When the effect is not
    identifiable, it returns ``False`` and stores the witness hedge in
    ``self.hedge_``.

    Parameters
    ----------
    causal_graph : ADMG
        An ADMG with exposures and outcomes roles assigned.

    Returns
    -------
    ProbabilityExpressionTree
        The symbolic formula for P(Y|do(X)) when identifiable.
    False
        When the effect is not identifiable. The witness hedge is stored
        in ``self.hedge_``.

    Examples
    --------
    >>> from pgmpy.base import ADMG
    >>> from pgmpy.identification import ID
    >>> admg = ADMG(
    ...     edge_list=[("X", "M", "->"), ("M", "Y", "->"), ("X", "Y", "<>")],
    ...     exposures={"X"},
    ...     outcomes={"Y"},
    ... )
    >>> id_algo = ID()
    >>> result = id_algo.identify(admg)
    """

    supported_graph_types = (ADMG, DAG)

    def _identify(self, causal_graph):
        """Run the ID algorithm.

        Parameters
        ----------
        causal_graph : ADMG or DAG
            The causal graph with exposures and outcomes roles assigned.

        Returns
        -------
        ProbabilityExpressionTree or False
            The identified formula, or False if not identifiable.
        """
        # TODO: Remove this conversion once DAG inherits _CoreGraph.
        if isinstance(causal_graph, DAG):
            causal_graph = ADMG(
                edge_list=[(u, v, "->") for u, v in causal_graph.edges()],
                roles=causal_graph.get_role_dict(),
            )

        exposures = frozenset(causal_graph.get_role("exposures"))
        outcomes = frozenset(causal_graph.get_role("outcomes"))
        variables = frozenset(causal_graph.nodes())

        # One topological ordering, fixed over the ORIGINAL graph and threaded
        # unchanged through every recursive call.
        ordering = causal_graph.get_topological_order()

        # The estimand is the observational joint P(V) and is transformed as we recurse.
        estimand = ProbabilityNode(variables)

        result = self._identify_recursive(outcomes, exposures, variables, causal_graph, estimand, ordering)
        if result is False:
            return False
        return ProbabilityExpressionTree(root=result)

    def _marginalize(node, sumset):
        """Return ``Σ_{sumset} node``, collapsing to ``node`` if sumset is empty."""
        if sumset:
            return MarginalNode(node, sumset=frozenset(sumset))
        return node

    def _product(factors):
        """Build a product of ``factors``, collapsing a single factor to itself."""
        factors = list(factors)
        if len(factors) == 1:
            return factors[0]
        return ProductNode(factors)

    def _ordered_districts(graph):
        """Return the C-components (districts) of ``graph`` as a deterministically
        ordered ``list`` of frozensets.
        """
        order = {n: i for i, n in enumerate(graph.nodes())}
        return sorted(graph.get_district(), key=lambda c: min(order[n] for n in c))

    def _district_product(district, variables, ordering):
        r"""Return ``∏_{V_i ∈ district} P(V_i | v_pi^{(i-1)})``.

        The product runs over the variables of ``district`` in the fixed global
        ``ordering``. Each factor conditions on every variable that precedes
        ``V_i`` in that ordering *among the current variable set* ``variables``.
        This realises the products that appear in lines 6 and 7 of the paper.

        Parameters
        ----------
        district : frozenset
            The C-component whose chain factorisation is wanted.
        variables : frozenset
            The current variable set ``V`` of the subproblem (predecessors are
            restricted to it, preserving the global order).
        ordering : list
            The fixed topological ordering of the original graph.
        """
        order_in_vars = [v for v in ordering if v in variables]
        factors = []
        for v_i in order_in_vars:
            if v_i in district:
                predecessors = frozenset(order_in_vars[: order_in_vars.index(v_i)])
                factors.append(ProbabilityNode(frozenset({v_i}), cond=predecessors))
        return ID._product(factors)

    def _identify_recursive(self, outcomes, exposures, variables, causal_graph, estimand, ordering):
        """Recursive implementation of the ID algorithm.

        Parameters
        ----------
        outcomes : frozenset
            Outcome variables (what we want to estimate).
        exposures : frozenset
            Intervention variables (what we intervene on).
        variables : frozenset
            All variables in the current subproblem.
        causal_graph : ADMG or DAG
            Current causal graph (may be a subgraph of the original).
        estimand : _TreeNode
            The carried probability expression ``P``. Begins as ``P(V)`` and is
            marginalised (line 2) or replaced by a district product ``Q_{S'}``
            (line 7) as the recursion proceeds.
        ordering : list
            The fixed topological ordering of the original graph.

        Returns
        -------
        ProbabilityExpressionTree or False
            The identified formula, or False if not identifiable.
        """
        # Line 1: no intervention -> the effect on Y is the marginal of the
        # carried estimand on Y, i.e. Σ_{V\Y} P.
        if not exposures:
            return self._marginalize(estimand, variables - outcomes)

        # Line 2: Ancestral restriction
        # If not all variables are ancestors of outcomes, restrict to An(outcomes)
        ancestors = causal_graph.get_ancestors(outcomes)
        if variables != ancestors:
            marg_estimand = self._marginalize(estimand, variables - ancestors)
            ancestral_graph = causal_graph.get_ancestral_graph(ancestors)
            return self._identify_recursive(
                outcomes,
                exposures & ancestors,
                ancestors,
                ancestral_graph,
                marg_estimand,
                ordering,
            )

        # Line 3: Add interventions for non-ancestors
        # Variables not in An(outcomes)_{G_exposures} can be added to interventions
        intervention_graph = causal_graph.do(exposures)
        ancestors_in_intervention = intervention_graph.get_ancestors(outcomes)
        w = (variables - exposures) - ancestors_in_intervention
        if w:
            return self._identify_recursive(outcomes, exposures | w, variables, causal_graph, estimand, ordering)

        # Lines 4-7: C-component based decomposition
        graph_minus_exposures = causal_graph.get_subgraph(variables - exposures, edge_types={"->", "<>"})
        c_components = self._ordered_districts(graph_minus_exposures)

        # Line 4: Multiple C-components - decompose
        if len(c_components) > 1:
            return self._decompose_by_c_components(
                outcomes, exposures, variables, causal_graph, c_components, estimand, ordering
            )

        # G\X has a single C-component S.
        S = c_components[0] if c_components else frozenset()

        # Line 5: Hedge detection
        c_components_graph = causal_graph.get_district()
        if len(c_components_graph) == 1:
            self.hedge_ = causal_graph.get_subgraph(S, edge_types={"->", "<>"})
            return False

        # Line 6: S is itself a C-component of G -> return
        #   Σ_{S\Y} ∏_{V_i ∈ S} P(V_i | v_pi^{(i-1)}).
        for S_prime in c_components_graph:
            product = self._district_product(S, variables, ordering)
            return self._marginalize(product, S - outcomes)

        # Line 7: S is a proper subset of some C-component S' of G -> replace the
        # estimand by the district product Q_{S'} = ∏_{V_i ∈ S'} P(V_i | v_pi^{(i-1)})
        # and recurse on G_{S'} with the reduced action set x ∩ S'.
        for S_prime in c_components_graph:
            if S < S_prime:
                q_sprime = self._district_product(S_prime, variables, ordering)
                subgraph = causal_graph.get_subgraph(S_prime, edge_types={"->", "<>"})
                return self._identify_recursive(
                    outcomes,
                    exposures & S_prime,
                    S_prime,
                    subgraph,
                    q_sprime,
                    ordering,
                )

        raise RuntimeError("ID algorithm reached an unexpected state")

    def _decompose_by_c_components(
        self, outcomes, exposures, variables, causal_graph, c_components, estimand, ordering
    ):
        r"""Line 4: decompose by C-components.

        When ``G\exposures`` has multiple C-components ``{S1, ..., Sk}``,

        .. math::

            P_\mathbf{x}(\mathbf y) = \sum_{\mathbf v \setminus (\mathbf y \cup \mathbf x)}
                \prod_i \mathbf{ID}(\mathbf s_i, \mathbf v \setminus \mathbf s_i, P, G).
        """
        subexprs = []
        for S in c_components:
            sub = self._identify_recursive(S, variables - S, variables, causal_graph, estimand, ordering)
            if sub is False:
                return False
            subexprs.append(sub)
        product = self._product(subexprs)
        return self._marginalize(product, variables - (outcomes | exposures))
