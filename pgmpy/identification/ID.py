from pgmpy.base import ADMG, DAG
from pgmpy.identification import BaseFormulaIdentification
from pgmpy.identification.probability_expression import (
    DivisionNode,
    MarginalNode,
    ProbabilityExpressionTree,
    ProbabilityNode,
    ProductNode,
)


class ID(BaseFormulaIdentification):
    """
    Given a causal graph, identifies the causal effect using the ID algorithm.

    The class implements the recursive identification procedure of :footcite:t:`shpitser_2006` for ADMGs and DAGs with
    exposure and outcome roles. When the effect is identifiable, it returns a symbolic probability expression for the
    interventional distribution. When the effect is not identifiable, it returns ``False`` and stores the witness hedge
    in ``self.hedge_``.

    Parameters
    ----------
    causal_graph : ADMG, DAG
        An ADMG or DAG with exposures and outcomes roles assigned. The two roles must be disjoint.

    Returns
    -------
    ProbabilityExpressionTree
        The symbolic formula for P(Y|do(X)) when identifiable.
    False
        When the effect is not identifiable. The witness hedge is stored in ``self.hedge_`` as the pair of graphs
        ``(G, G_S)`` thrown by line 5 of the algorithm; the C-forests forming the hedge are edge subgraphs of that pair
        (Theorem 6 of the paper).

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
    >>> id_algo.identify(admg).to_latex()
    '\\\\sum_{M} P(M \\\\mid X) \\\\left[ \\\\sum_{X} P(X) P(Y \\\\mid M, X) \\\\right]'
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
            admg = ADMG(edge_list=[(u, v, "->") for u, v in causal_graph.edges()])
            # `edge_list` alone silently drops nodes that have no edges.
            admg.add_nodes_from(causal_graph.nodes())
            for role, variables in causal_graph.get_role_dict().items():
                admg.with_role(role=role, variables=variables, inplace=True)
            causal_graph = admg

        exposures = frozenset(causal_graph.get_role("exposures"))
        outcomes = frozenset(causal_graph.get_role("outcomes"))
        variables = frozenset(causal_graph.nodes())

        # Definition 2 of the paper defines P_x(Y) only for Y ∩ X = ∅.
        if overlap := (exposures & outcomes):
            raise ValueError(f"'exposures' and 'outcomes' must be disjoint, but {sorted(overlap)} is in both.")

        # π, a single topological ordering fixed over the ORIGINAL graph and threaded unchanged through every recursive
        # call. Every graph the recursion constructs is an induced subgraph of G, and restricting a topological order
        # to a subset of nodes is still a topological order of the induced subgraph.
        ordering = causal_graph.get_topological_order()

        # P, the estimand. Starts as the observational joint P(v) and is rewritten by lines 2 and 7 as the recursion
        # proceeds.
        estimand = ProbabilityNode(variables)

        result = self._identify_recursive(outcomes, exposures, variables, causal_graph, estimand, ordering)
        if result is False:
            return False
        return ProbabilityExpressionTree(root=result)

    def _marginalize(self, node, sumset):
        r"""Return :math:`\sum_{sumset} node`.

        Three exact simplifications keep the expression readable: an empty ``sumset`` is the identity, summing
        variables out of a plain joint ``P(A)`` gives the smaller joint ``P(A \ sumset)``, and nested sums collapse
        into one. The second is not merely cosmetic -- it is what keeps the estimand a plain joint for as long as the
        algorithm has not genuinely left it, which in turn lets ``_district_product`` emit atomic conditionals instead
        of ratios.
        """
        sumset = frozenset(sumset)
        if not sumset:
            return node
        if isinstance(node, ProbabilityNode) and not node.do and not node.cond:
            return ProbabilityNode(node.variables - sumset)
        if isinstance(node, MarginalNode):
            return MarginalNode(node.children[0], sumset=sumset | node.sumset)
        return MarginalNode(node, sumset=sumset)

    def _district_product(self, estimand, district, ordered):
        r"""Return :math:`\prod_{V_i \in district} P(v_i \mid v_\pi^{(i-1)})`.

        This is the product shared by lines 6 and 7 of the paper. What the paper's notation hides is that ``P`` here is
        *the estimand carried into the current call*, not the original observational joint: line 7 replaces ``P`` by
        ``Q[S']`` before recursing, so a line 6 (or a further line 7) reached below it must factorise that ``Q[S']``
        instead. A conditional of an arbitrary estimand is recovered by marginalisation and division,

        .. math::

            P(v_i \mid v_\pi^{(i-1)}) =
                \frac{\sum_{v_{i+1}, \ldots, v_n} P}{\sum_{v_i, \ldots, v_n} P}

        which is where the division in e.g. the napkin graph comes from. When the estimand is still a plain joint both
        sums collapse to plain joints and the ratio :math:`P(A) / P(A \setminus \{V_i\})` is written back as the atomic
        term :math:`P(v_i \mid a \setminus \{v_i\})`.

        Line 7 writes the factor as ``P(V_i | V_π^{(i-1)} ∩ S', v_π^{(i-1)} \ S')``, distinguishing preceding variables
        inside ``S'`` from those outside it that are held at a fixed value. Both sit behind the conditioning bar, so as
        a probability expression this is the same product as line 6's, only ranging over ``S'`` instead of ``S``.

        Parameters
        ----------
        estimand : _TreeNode
            The ``P`` argument of the current call.
        district : frozenset
            The C-component to factorise: ``S`` on line 6, ``S'`` on line 7.
        ordered : list
            The current variable set ``V`` in topological order, i.e. ``π`` restricted to this subproblem.
        """
        factors = []
        for i, v_i in enumerate(ordered):
            if v_i not in district:
                continue

            numerator = self._marginalize(estimand, ordered[i + 1 :])
            if i == 0:
                # v_π^{(0)} is empty, so the denominator would be Σ_v P = 1.
                factors.append(numerator)
                continue

            denominator = self._marginalize(estimand, ordered[i:])
            if isinstance(numerator, ProbabilityNode) and isinstance(denominator, ProbabilityNode):
                # Both collapsed, so P(A) / P(A \ {V_i}) is the atomic P(v_i | a \ {v_i}).
                factors.append(ProbabilityNode(frozenset({v_i}), cond=denominator.variables))
            else:
                factors.append(DivisionNode(numerator, denominator))
        # A district may be a singleton, and ProductNode needs two factors.
        return factors[0] if len(factors) == 1 else ProductNode(factors)

    def _identify_recursive(self, outcomes, exposures, variables, causal_graph, estimand, ordering):
        """Recursive implementation of the ID algorithm, following Fig. 3 of the paper.

        Parameters
        ----------
        outcomes : frozenset
            ``y``, the outcome variables.
        exposures : frozenset
            ``x``, the variables being intervened on.
        variables : frozenset
            ``V``, all variables of the current subproblem. Always equal to the node set of ``causal_graph``.
        causal_graph : ADMG
            ``G``, the current graph; an induced subgraph of the original.
        estimand : _TreeNode
            ``P``, the carried probability expression. Begins as ``P(v)`` and is marginalised (line 2) or replaced by
            ``Q[S']`` (line 7).
        ordering : list
            ``π``, the topological ordering fixed over the original graph.

        Returns
        -------
        _TreeNode or False
            The expression for ``P_x(y)``, or False if a hedge was found.
        """
        # Line 1: x = ∅. The effect is a marginal of the carried estimand.
        if not exposures:
            return self._marginalize(estimand, variables - outcomes)

        # Line 2: V ≠ An(Y)_G. Restrict the whole problem to An(Y).
        ancestors = frozenset(causal_graph.get_ancestors(outcomes))
        if variables != ancestors:
            return self._identify_recursive(
                outcomes,
                exposures & ancestors,
                ancestors,
                causal_graph.get_ancestral_graph(ancestors),
                self._marginalize(estimand, variables - ancestors),
                ordering,
            )

        # Line 3: W = (V \ X) \ An(Y)_{G_x̄}. Intervening on W is free (Lemma 6).
        w = (variables - exposures) - causal_graph.do(exposures).get_ancestors(outcomes)
        if w:
            return self._identify_recursive(outcomes, exposures | w, variables, causal_graph, estimand, ordering)

        # C(G \ X), ordered by π so the emitted product is deterministic.
        position = {node: index for index, node in enumerate(ordering)}
        c_components = sorted(
            causal_graph.get_subgraph(variables - exposures).get_district(),
            key=lambda component: min(position[node] for node in component),
        )

        # Line 4: C(G \ X) = {S1, ..., Sk}. Decompose into k independent subproblems.
        if len(c_components) > 1:
            factors = []
            for component in c_components:
                factor = self._identify_recursive(
                    component, variables - component, variables, causal_graph, estimand, ordering
                )
                if factor is False:
                    # FAIL propagates through recursive calls like an exception.
                    return False
                factors.append(factor)
            # k > 1 here, so there are always at least two factors.
            return self._marginalize(ProductNode(factors), variables - (outcomes | exposures))

        # C(G \ X) = {S}. S is bidirected-connected in G \ X and G \ X keeps every bidirected edge of G between its
        # nodes, so S is bidirected-connected in G too and therefore lies inside exactly one C-component of G.
        S = c_components[0]
        S_prime = frozenset(causal_graph.get_district(next(iter(S))))

        # Line 5: C(G) = {G}, i.e. the whole of G is that one C-component.
        if S_prime == variables:
            self.hedge_ = (causal_graph, causal_graph.get_subgraph(S))
            return False

        ordered = [node for node in ordering if node in variables]

        # Line 6: S ∈ C(G). Return Σ_{s\y} Π_{Vi∈S} P(v_i | v_π^{(i-1)}).
        if S_prime == S:
            return self._marginalize(self._district_product(estimand, S, ordered), S - outcomes)

        # Line 7: S ⊂ S' ∈ C(G). Recurse into G_{S'} with the estimand replaced by Q[S'].
        return self._identify_recursive(
            outcomes,
            exposures & S_prime,
            S_prime,
            causal_graph.get_subgraph(S_prime),
            self._district_product(estimand, S_prime, ordered),
            ordering,
        )
