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
        # TODO: Remove this conversion once DAG inherits _CoreGraph. The ID
        # algorithm is built on _CoreGraph methods (get_district, get_ancestral_graph,
        # get_subgraph, do, ...) which DAG does not yet expose, so DAG inputs currently
        # cannot run the algorithm directly. As a temporary workaround we convert the
        # DAG to an equivalent ADMG (all directed edges, no bidirected edges) carrying
        # over the node roles, and run the algorithm on that.
        if isinstance(causal_graph, DAG):
            causal_graph = ADMG(
                edge_list=[(u, v, "->") for u, v in causal_graph.edges()],
                roles=causal_graph.get_role_dict(),
            )

        exposures = frozenset(causal_graph.get_role("exposures"))
        outcomes = frozenset(causal_graph.get_role("outcomes"))
        variables = frozenset(causal_graph.nodes())

        # Base case: if no intervention, return marginal P(outcomes)
        if not exposures:
            return self._marginal(outcomes, variables, None)

        return self._identify_recursive(outcomes, exposures, variables, causal_graph, None)

    def _marginal(self, outcomes, variables, base_expr):
        r"""Return P(outcomes) = Σ_{variables\outcomes} P(variables).

        Line 1 of the ID algorithm: when there's no intervention,
        the causal effect is just the marginal distribution.

        Parameters
        ----------
        outcomes : frozenset
            Variables to marginalize to.
        variables : frozenset
            All variables in the current scope.
        base_expr : ProbabilityNode or None
            The current probability expression. If None, use P(variables).
        """
        sumset = variables - outcomes

        if base_expr is None:
            # Use the joint P(variables)
            prob = ProbabilityNode(variables)
        else:
            # Use the provided base expression
            prob = base_expr

        if sumset:
            return ProbabilityExpressionTree(root=MarginalNode(prob, sumset=sumset))
        return ProbabilityExpressionTree(root=prob)

    def _identify_recursive(self, outcomes, exposures, variables, causal_graph, base_expr):
        """Recursive implementation of the ID algorithm (Lines 2-7).

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
        base_expr : ProbabilityNode or None
            The current probability expression. If None, use P(variables).

        Returns
        -------
        ProbabilityExpressionTree or False
            The identified formula, or False if not identifiable.
        """
        # Restrict to nodes in the graph
        graph_nodes = frozenset(causal_graph.nodes())
        outcomes = outcomes & graph_nodes
        exposures = exposures & graph_nodes
        variables = variables & graph_nodes

        # Base case: no intervention
        if not exposures:
            return self._marginal(outcomes, variables, base_expr)

        # Line 2: Ancestral restriction
        # If not all variables are ancestors of outcomes, restrict to An(outcomes)
        ancestors = causal_graph.get_ancestors(outcomes)
        if variables != ancestors:
            exposures_restricted = exposures & ancestors
            ancestral_graph = causal_graph.get_ancestral_graph(ancestors)
            return self._identify_recursive(outcomes, exposures_restricted, ancestors, ancestral_graph, base_expr)

        # Line 3: Add interventions for non-ancestors
        # Variables not in An(outcomes)_{G_exposures} can be added to interventions
        exposures_in_graph = exposures & graph_nodes
        if exposures_in_graph:
            intervention_graph = causal_graph.do(exposures_in_graph)
            ancestors_in_intervention = intervention_graph.get_ancestors(outcomes & graph_nodes)
            non_ancestors = (variables - exposures_in_graph) - ancestors_in_intervention
            if non_ancestors:
                return self._identify_recursive(
                    outcomes, exposures_in_graph | non_ancestors, variables, causal_graph, base_expr
                )

        # Lines 4-7: C-component based decomposition
        graph_minus_exposures = causal_graph.get_subgraph(variables - exposures, edge_types={"->", "<>"})
        c_components = graph_minus_exposures.get_district()

        # Line 4: Multiple C-components - decompose
        if len(c_components) > 1:
            return self._decompose_by_c_components(
                outcomes, exposures, variables, causal_graph, c_components, base_expr
            )

        # Single C-component in G\exposures
        S = next(iter(c_components)) if c_components else frozenset()

        # Line 5: Hedge detection
        c_components_graph = causal_graph.get_district()
        if len(c_components_graph) == 1:
            self.hedge_ = (causal_graph, S)
            return False

        # Line 6: S is a C-component of causal_graph
        for S_prime in c_components_graph:
            if S == S_prime:
                return self._compute_expression(outcomes, S, variables, causal_graph, base_expr)

        # Line 7: Find enclosing C-component
        for S_prime in c_components_graph:
            if S < S_prime:  # S is proper subset of S'
                # Transform P(V) into P_S'(V_S') = ∏ P(Vi | predecessors)
                transformed_expr = self._compute_expression(S_prime, S_prime, variables, causal_graph, base_expr)
                subgraph = causal_graph.get_subgraph(S_prime, edge_types={"->", "<>"})
                # Recurse with the transformed probability expression
                # Pass the root of the expression tree as the new base
                return self._identify_recursive(outcomes, exposures & S_prime, S_prime, subgraph, transformed_expr.root)

        raise RuntimeError("ID algorithm reached unexpected state")

    def _decompose_by_c_components(self, outcomes, exposures, variables, causal_graph, c_components, base_expr):
        r"""Line 4: Decompose by C-components.

        When G\exposures has multiple C-components {S1, ..., Sk},
        the effect decomposes as:
            P_exposures(outcomes) = Σ_{variables\(outcomes∪exposures)} ∏_i P_{variables\si}(si)
        """
        sumset = variables - (outcomes | exposures)
        subexprs = []

        for S in c_components:
            subexpr = self._identify_recursive(S, variables - S, variables, causal_graph, base_expr)
            if subexpr is False:
                return False
            subexprs.append(subexpr.root if isinstance(subexpr, ProbabilityExpressionTree) else subexpr)

        result_root = subexprs[0] if len(subexprs) == 1 else ProductNode(subexprs)

        if sumset:
            return ProbabilityExpressionTree(root=MarginalNode(result_root, sumset=sumset))
        return ProbabilityExpressionTree(root=result_root)

    def _compute_expression(self, outcomes, S, variables, causal_graph, base_expr):
        r"""Line 6: Compute P_x(y) = Σ_{S\outcomes} ∏ P(Vi | predecessors).

        When S (the C-component from G\\exposures) is exactly a C-component of G,
        we can compute:
            P_exposures(outcomes) = Σ_{S\outcomes} ∏_{Vi ∈ S} P(Vi | predecessors_in_topological_order)

        Parameters
        ----------
        outcomes : frozenset
            Outcome variables.
        S : frozenset
            The C-component containing all variables in current scope.
        variables : frozenset
            All variables in the original graph (for topological ordering).
        causal_graph : ADMG
            Current causal graph (subgraph).
        base_expr : ProbabilityNode or None
            The current probability expression. If None, use P(variables).
        """
        # Get topological order of the causal graph and restrict to nodes in S
        topo_order = causal_graph.get_topological_order()
        topo_in_S = [v for v in topo_order if v in S]

        factors = []
        for Vi in topo_in_S:
            vi_index = topo_order.index(Vi)
            predecessors = frozenset(topo_order[:vi_index])
            factors.append(ProbabilityNode(frozenset({Vi}), cond=predecessors))

        product = factors[0] if len(factors) == 1 else ProductNode(factors)
        sumset = S - outcomes

        result = MarginalNode(product, sumset=sumset) if sumset else product
        return ProbabilityExpressionTree(root=result)
