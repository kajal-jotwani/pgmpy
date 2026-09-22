from itertools import combinations, product

from pgmpy.base import ADMG, DAG
from pgmpy.identification import BaseFormulaIdentification
from pgmpy.identification.probability_expression import (
    ConstantNode,
    MarginalNode,
    ProbabilityExpressionTree,
    ProbabilityNode,
    ProductNode,
)

# What make-cg returns in place of the relabelled query when two nodes it merged carry different values: the events
# of the query then contradict each other, so the query is consistent with no world at all and P(gamma) = 0.
_INCONSISTENT = object()

# Stands in for "this node has not been given a value", which is not the same as being given the value ``None``.
_UNSET = object()


class Counterfactual:
    r"""
    A counterfactual variable :math:`Y_x`: the variable :math:`Y` in the submodel :math:`M_x`.

    The subscript ``x`` is the action the submodel is obtained by, that is, the set of variables the submodel
    holds fixed together with the values it holds them at. An empty subscript leaves the original model, so
    ``Counterfactual("Y")`` is the observable variable Y itself.

    Instances are immutable and hashable, and are used as the nodes of the parallel worlds and counterfactual
    graphs that :class:`IDStar` builds.

    Parameters
    ----------
    variable : hashable
        The variable of the original causal model the counterfactual is derived from.

    intervention : mapping or iterable of (variable, value) pairs, optional
        The action, as the value each fixed variable is held at. Default: the empty action.

    Attributes
    ----------
    variable : hashable
        The variable the counterfactual is derived from.

    intervention : tuple of (variable, value) pairs
        The action, sorted so that two counterfactuals built from the same action compare and hash alike.

    Examples
    --------
    >>> from pgmpy.identification import Counterfactual
    >>> Counterfactual("Y", {"X": "x0"})
    Y_{X=x0}
    >>> Counterfactual("Y", {"X": "x0"}).action
    {'X': 'x0'}
    >>> Counterfactual("Y") == Counterfactual("Y", {})
    True
    """

    def __init__(self, variable, intervention=None):
        self.variable = variable
        self.intervention = tuple(sorted(dict(intervention or {}).items(), key=lambda item: str(item[0])))

    @property
    def action(self):
        """The action as a mapping from each fixed variable to the value it is held at.

        Returns
        -------
        action: dict
        """
        return dict(self.intervention)

    def __repr__(self):
        if not self.intervention:
            return str(self.variable)
        subscript = ", ".join(f"{variable}={value}" for variable, value in self.intervention)
        return f"{self.variable}_{{{subscript}}}"

    def __eq__(self, other):
        if not isinstance(other, Counterfactual):
            return NotImplemented
        return (self.variable, self.intervention) == (other.variable, other.intervention)

    def __hash__(self):
        return hash((self.variable, self.intervention))


class CounterfactualEvent:
    r"""
    A counterfactual event :math:`Y_x = y`: the counterfactual variable :math:`Y_x` attaining the value ``y``.

    A conjunction of such events is the query :class:`IDStar` identifies. Events may disagree about the world
    they speak of -- that is what makes the query counterfactual rather than interventional. For instance,
    ``P(y_x, x')`` asks for the probability that Y would have been y had X been x, in a population where X was
    in fact observed to be x'.

    Parameters
    ----------
    variable : hashable
        The variable of the original causal model the event is about.

    value : hashable
        The value the counterfactual variable attains.

    intervention : mapping or iterable of (variable, value) pairs, optional
        The action the counterfactual variable is taken under. Default: the empty action, which makes the event
        a plain observation of ``variable``.

    Attributes
    ----------
    counterfactual : Counterfactual
        The counterfactual variable the event assigns a value to.

    value : hashable
        The assigned value.

    Examples
    --------
    >>> from pgmpy.identification import CounterfactualEvent
    >>> CounterfactualEvent("Y", "y0", intervention={"X": "x0"})
    Y_{X=x0} = y0
    >>> CounterfactualEvent("X", "x1")
    X = x1
    """

    def __init__(self, variable, value, intervention=None):
        self.counterfactual = Counterfactual(variable, intervention)
        self.value = value

    @property
    def variable(self):
        """The variable of the original causal model the event is about."""
        return self.counterfactual.variable

    @property
    def intervention(self):
        """The action, as a sorted tuple of (variable, value) pairs."""
        return self.counterfactual.intervention

    @property
    def action(self):
        """The action as a mapping from each fixed variable to the value it is held at."""
        return self.counterfactual.action

    def __repr__(self):
        return f"{self.counterfactual!r} = {self.value}"

    def __eq__(self, other):
        if not isinstance(other, CounterfactualEvent):
            return NotImplemented
        return (self.counterfactual, self.value) == (other.counterfactual, other.value)

    def __hash__(self):
        return hash((self.counterfactual, self.value))


class IDStar(BaseFormulaIdentification):
    r"""
    Given a causal graph, identifies a counterfactual query from experimental distributions.

    The class implements the ID* algorithm of :footcite:t:`shpitser_2008` (Fig. 12) together with the
    counterfactual graph construction make-cg (Fig. 10) that it rests on. The query is a conjunction
    :math:`\gamma` of counterfactual events, and the expression returned is written in terms of the
    interventional distributions :math:`P_x`, that is, in terms of experimental studies. ID* is sound and
    complete (Theorems 26 and 31): whenever it fails, no method can identify the query.

    Unlike ``ID``, the query is not read off node roles -- a counterfactual event carries an action and a value,
    which a role cannot express -- so it is passed to ``identify`` as the ``event`` argument.

    Parameters
    ----------
    causal_graph: ADMG | DAG
        An ADMG, or a DAG whose unobserved variables carry the ``latents`` role. No roles are required.

    event: CounterfactualEvent | iterable of CounterfactualEvent
        The conjunction :math:`\gamma` whose probability is sought. Every variable it mentions, in an event or
        in an action, must be an observed node of the graph.

    Returns
    -------
    expression: ProbabilityExpressionTree | False
        The symbolic formula for :math:`P(\gamma)`, or False if the query is not identifiable. An inconsistent
        query is identifiable and its expression is the constant 0; an empty one has probability 1. When False,
        the counterfactual graph of the failing call is stored in ``self.hedge_``; by Theorem 32 it contains a
        C-component with the conflicting value assignment that witnesses non-identifiability.

    Examples
    --------
    The effect of treatment on the treated, as a joint query: how likely is it that Y would have been ``y0``
    under treatment ``x0``, jointly with X in fact being ``x1``?

    >>> from pgmpy.base import ADMG
    >>> from pgmpy.identification import CounterfactualEvent, IDStar
    >>> admg = ADMG(edge_list=[("X", "M", "->"), ("M", "Y", "->"), ("X", "Y", "<>")])
    >>> event = [
    ...     CounterfactualEvent("Y", "y0", intervention={"X": "x0"}),
    ...     CounterfactualEvent("X", "x1"),
    ... ]
    >>> IDStar().identify(admg, event=event).to_latex()
    '\\sum_{m} P(M = m \\mid do(X = x0)) P(X = x1, Y = y0 \\mid do(M = m))'

    The probability of necessity and sufficiency is not identifiable in any graph where X is a parent of Y
    (Lemma 27), because :math:`Y_{x_0}` and :math:`Y_{x_1}` share an unobserved parent.

    >>> event = [
    ...     CounterfactualEvent("Y", "y0", intervention={"X": "x0"}),
    ...     CounterfactualEvent("Y", "y1", intervention={"X": "x1"}),
    ... ]
    >>> IDStar().identify(ADMG(edge_list=[("X", "Y", "->")]), event=event)
    False

    References
    ----------
    - :footcite:t:`shpitser_2008`
    """

    supported_graph_types = (ADMG, DAG)
    # The query is the `event` argument of `identify`, not a set of roles on the graph.
    required_roles = ()

    def _identify(self, causal_graph, event):
        r"""Run the ID* algorithm.

        Parameters
        ----------
        causal_graph: ADMG | DAG
            The causal graph.

        event: CounterfactualEvent | iterable of CounterfactualEvent
            The conjunction :math:`\gamma`.

        Returns
        -------
        expression: ProbabilityExpressionTree | False
            The identified formula, or False if the query is not identifiable.
        """
        events = self._validate_event(causal_graph, event)

        # ID* is stated for semi-Markovian models, so a DAG is first replaced by its latent projection. The
        # projection keeps the unobserved parents shared by two variables, as bidirected edges; the unobserved
        # parent every variable has of its own is what the parallel worlds graph adds back, since that is the
        # parent the copies of a variable in different worlds share.
        if isinstance(causal_graph, DAG):
            causal_graph = causal_graph.to_admg()

        result = self._identify_counterfactual(causal_graph, events)
        if result is False:
            return False
        return ProbabilityExpressionTree(root=result)

    @staticmethod
    def _mentioned_variables(events):
        """Return every variable the conjunction mentions, whether in an event or in an action."""
        return {event.variable for event in events} | {variable for event in events for variable in event.action}

    def _validate_event(self, causal_graph, event):
        """Check the query and return it as a frozen set of events.

        Parameters
        ----------
        causal_graph: ADMG | DAG
            The causal graph.

        event: CounterfactualEvent | iterable of CounterfactualEvent
            The query as passed to ``identify``. A lone event may be passed unwrapped.

        Returns
        -------
        events: frozenset of CounterfactualEvent

        Raises
        ------
        ValueError
            If the query is not made of ``CounterfactualEvent`` instances, or mentions a variable that is not an
            observed node of the graph.
        """
        events = frozenset([event] if isinstance(event, CounterfactualEvent) else event)

        if wrong_type := [item for item in events if not isinstance(item, CounterfactualEvent)]:
            raise ValueError(f"`event` must consist of CounterfactualEvent instances. Got {wrong_type[0]!r}.")

        mentioned = self._mentioned_variables(events)
        if missing := (mentioned - set(causal_graph.nodes())):
            raise ValueError(f"{sorted(missing, key=str)} in `event` are not nodes of the causal graph.")

        # A latent has no interventional quantity of its own, so no event can be about one. The latent projection
        # `_identify` takes next drops the latents, so the query has to be checked against them here, while they
        # are still nodes of the graph.
        if latent_variables := (set(causal_graph.latents) & mentioned):
            raise ValueError(f"{sorted(latent_variables, key=str)} cannot be both latent and in 'event'.")

        return events

    def _identify_counterfactual(self, causal_graph, events):
        r"""Recursive implementation of ID*, following Fig. 12 of the paper.

        Parameters
        ----------
        causal_graph: ADMG
            ``G``, the causal graph. It is the same in every call: unlike ID, ID* never recurses into a subgraph.
            What the recursion varies is the conjunction, and with it the counterfactual graph Line 4 builds.

        events: frozenset of CounterfactualEvent
            :math:`\gamma`, the conjunction whose probability is sought.

        Returns
        -------
        expression: _TreeNode | False
            The expression for :math:`P(\gamma)`, or False if Line 8 threw FAIL.
        """
        # step 1: the probability of an empty conjunction is 1 by convention.
        if not events:
            return ConstantNode(1)

        # step 2: an event x_{x'..} violates the Axiom of Effectiveness -- X cannot attain x in a world where it
        # has been forced to x' -- so the conjunction is inconsistent and its probability is 0.
        for event in events:
            if event.action.get(event.variable, event.value) != event.value:
                return ConstantNode(0)

        # step 3: an event x_{x..} is tautological -- X attains x in a world where it has been forced to x -- so
        # dropping it leaves the probability unchanged.
        for event in events:
            if event.variable in event.action:
                return self._identify_counterfactual(causal_graph, events - {event})

        # step 4: build the counterfactual graph and the relabelled conjunction gamma'.
        graph, observations = self._make_counterfactual_graph(causal_graph, events)

        # step 5: make-cg found two events speaking of what turned out to be the same random variable, but
        # assigning it different values.
        if observations is _INCONSISTENT:
            return ConstantNode(0)

        # C(G'), the maximal C-components of the counterfactual graph. Nodes fixed by an intervention count as
        # part of no C-component, and V(G') is the set of observable nodes that are not fixed.
        observable = {node for node in graph.nodes() if not self._is_fixed(node)}
        districts = graph.get_subgraph(observable).get_district()

        # step 6: more than one C-component, so the problem decomposes into one subproblem per C-component.
        if len(districts) > 1:
            return self._c_component_factorization(causal_graph, graph, observable, districts, observations)

        # Steps 7-9: the base case, a counterfactual graph with a single C-component.
        return self._single_c_component(graph, observable, observations)

    def _c_component_factorization(self, causal_graph, graph, observable, districts, observations):
        r"""Line 6: decompose by C-component factorization.

        The term of a C-component :math:`S_i` is the conjunction of its nodes taken under the action that fixes
        every other observable node of the counterfactual graph at its value, :math:`s^i_{v(G') \setminus s_i}`,
        and the whole product is summed over the observable nodes gamma' does not mention.

        Parameters
        ----------
        causal_graph: ADMG
            ``G``, the causal graph the recursive calls are made on.

        graph: ADMG
            ``G'``, the counterfactual graph.

        observable: set of Counterfactual
            ``V(G')``, the nodes of the counterfactual graph not fixed by an intervention.

        districts: set of frozenset of Counterfactual
            ``C(G')``, the maximal C-components of the counterfactual graph.

        observations: dict
            :math:`\gamma'`, mapping each node the conjunction mentions to the value it assigns.

        Returns
        -------
        expression: _TreeNode | False
            The sum of the product of the per-C-component terms, or False if one of them threw FAIL.
        """
        # The nodes of V(G') that gamma' does not mention are the ones the outer summation runs over. Each is
        # given a value symbol of its own: two nodes derived from the same variable are different random
        # variables and are summed over separately, so naming the summation after the variable would not do.
        used_values = set(observations.values()) | {
            self._fixed_value(node) for node in graph.nodes() if self._is_fixed(node)
        }
        free_values = self._free_values(observable - set(observations), used_values)
        values = {**observations, **free_values}

        factors = []
        # C(G') is a set, so the districts are put in a fixed order to keep the emitted product deterministic.
        for district in sorted(districts, key=lambda component: sorted(map(str, component))):
            outside = observable - district
            # The term of this C-component fixes every node outside it, so no node outside it is left listening
            # to its own parents. `_district_term` reads off that graph which of them a given node still hears.
            cut = graph.do(outside)
            conjunction = frozenset(self._district_term(cut, node, values, outside) for node in district)

            factor = self._identify_counterfactual(causal_graph, conjunction)
            if factor is False:
                # FAIL propagates through the recursive calls like an exception.
                return False
            if factor == ConstantNode(0):
                # An impossible factor makes every summand of the product zero.
                return ConstantNode(0)
            factors.append(factor)

        # There is more than one C-component here, so the product always has at least two factors.
        factorization = ProductNode(factors)
        sumset = set(free_values.values())
        return MarginalNode(factorization, sumset=sumset) if sumset else factorization

    def _district_term(self, cut, node, values, outside):
        r"""Return the event of ``node`` in the term of its C-component on Line 6.

        The term holds every node outside the C-component fixed at its value, so what the event of ``node`` is
        taken under is whatever ``node`` still listens to once that has happened: the outside nodes it is still a
        descendant of, and the nodes already fixed by an intervention that it is still a descendant of. A node
        held fixed passes nothing on from its own causes, so anything reaching ``node`` only through another
        outside node cannot change it, and naming it in the subscript would be redundant.

        This is the "removing redundant subscripts" step the paper performs on the term of its worked example
        (p. 1963), where :math:`Y_{x,z}` in the C-component :math:`\{Y_{x,z}, X\}` is written :math:`y_{z,w}`:
        fixing W cuts the only path by which the action on X reached Y. It is also what keeps the subscript a
        consistent action -- several nodes derived from one variable, with different values, may well sit outside
        the C-component at once, and at most one of them can still be heard. Should a node the C-component is
        already taken under be derived from that same variable too, it is the one that is kept, since the action
        that reaches ``node`` through its own world is the one its value was defined by.

        Parameters
        ----------
        cut: ADMG
            ``G'`` with the incoming edges of every node in `outside` removed, which is the graph the term's
            action leaves behind.

        node: Counterfactual
            The node of the C-component the event is about.

        values: dict
            The value of every node of ``V(G')``, observed or summed over.

        outside: set of Counterfactual
            :math:`v(G') \setminus s_i`, the observable nodes outside the C-component of ``node``.

        Returns
        -------
        event: CounterfactualEvent
        """
        ancestors = cut.get_ancestors(node)
        action = dict(self._subscript(cut, node))

        for other in outside:
            if other in ancestors and other.variable not in action:
                action[other.variable] = values[other]

        return CounterfactualEvent(node.variable, values[node], action)

    def _single_c_component(self, graph, district, observations):
        r"""Lines 7-9: the base case, where the counterfactual graph is a single C-component.

        Parameters
        ----------
        graph: ADMG
            ``G'``, the counterfactual graph.

        district: set of Counterfactual
            ``S``, the only C-component, which is the whole of ``V(G')``.

        observations: dict
            :math:`\gamma'`, mapping each node the conjunction mentions to the value it assigns.

        Returns
        -------
        expression: _TreeNode | False
            The interventional term of Line 9, or False if Line 8 threw FAIL.
        """
        # Line 8: fail on a conflict, that is, on a variable that one subscript sets to x while another subscript
        # or an observed event gives it a different value x'. By Theorem 32 that is the graphical condition under
        # which P(gamma) cannot be identified, so failing here is what makes ID* complete.
        #
        # The check below is the one the soundness argument of Theorem 26 asks for: that S hold "no conflicting
        # value assignments to any variable, obtained either by observation or intervention". The pseudocode of
        # Line 8 states the narrower condition that one of the two conflicting values be a subscript, which lets
        # through a single C-component holding two nodes derived from one variable that gamma' observes to take
        # different values -- a conjunction of the P(y_x, y') form that Lemma 27 rules non-identifiable, and one
        # that a single term of P could not express in any case.
        assignment = {}
        for node in district:
            for variable, value in self._subscript(graph, node):
                if assignment.setdefault(variable, value) != value:
                    self.hedge_ = graph
                    return False

        # x, the union of the subscripts of the counterfactual variables in S, recorded before the observed
        # values are folded in: it is what the returned term intervenes on.
        action = dict(assignment)

        for node, value in observations.items():
            if assignment.setdefault(node.variable, value) != value:
                self.hedge_ = graph
                return False

        # Line 9: with no conflict it is safe to take the union of all subscripts, and the answer is the effect of
        # those subscripts on the variables of gamma'. The nodes of S that gamma' does not mention are ancestors
        # the query says nothing about, and summing them out is exactly what leaving them out of the term does.
        variables = {node.variable for node in observations}
        return ProbabilityNode(variables, do=set(action), values=assignment)

    def _make_counterfactual_graph(self, causal_graph, events):
        r"""Build the counterfactual graph of a conjunction, following make-cg in Fig. 10 of the paper.

        The parallel worlds graph holds one copy of ``causal_graph`` per action the conjunction mentions, and the
        copies share their unobserved variables. Distinct nodes of that graph can be the very same random
        variable, which makes d-separation in it misleading, so Lemmas 24 and 25 are applied in topological order
        to merge every such pair. The result is restricted to the ancestors of the conjunction.

        Parameters
        ----------
        causal_graph: ADMG
            ``G``, the causal graph.

        events: frozenset of CounterfactualEvent
            :math:`\gamma`, the conjunction.

        Returns
        -------
        graph: ADMG
            :math:`G_\gamma`, the counterfactual graph. Its nodes are ``Counterfactual`` instances named after the
            world they were created in; the counterfactual variable a node denotes once the merging is done is the
            one with the subscript ``_subscript`` reads off the graph.

        observations: dict or _INCONSISTENT
            :math:`\gamma'`, mapping each node of the counterfactual graph the relabelled conjunction mentions to
            the value it assigns, or ``_INCONSISTENT`` if two events merged into one node disagree on the value.
        """
        # First step of Fig. 10: one submodel graph per action mentioned in gamma, sharing their U nodes. The
        # actions are ordered by size so that the copy in the world with the fewest interventions is the one a
        # merge keeps, which is what gives merged nodes their shortest subscript.
        worlds = sorted({event.intervention for event in events}, key=lambda world: (len(world), str(world)))
        graph = self._parallel_worlds_graph(causal_graph, worlds)

        # gamma', which starts as gamma and is relabelled by the merges below. Two events about the same
        # counterfactual variable that assign it different values are already inconsistent.
        observations = {}
        for event in events:
            if observations.setdefault(event.counterfactual, event.value) != event.value:
                return graph, _INCONSISTENT

        # Second and third steps of Fig. 10: apply Lemmas 24 and 25 to each pair of observable nodes derived from
        # the same variable, in topological order. A topological order of the parallel worlds graph runs through
        # the copies of one variable of G before those of any of its children, so ordering the variables of G is
        # enough, and by the time a variable is reached all of its parents have been merged as far as they will
        # be.
        representative = {}
        for variable in causal_graph.get_topological_order():
            survivors = []
            for world in worlds:
                node = Counterfactual(variable, world)
                # A node fixed by an intervention is a constant rather than a copy of the mechanism of its
                # variable, so Lemma 24, which asks for two equal mechanisms, never applies to it.
                if self._is_fixed(node):
                    continue

                for kept in survivors:
                    if not self._is_same_variable(causal_graph, node, kept, representative, observations):
                        continue
                    # The two nodes are the same random variable, so gamma cannot assign them different values.
                    # Only an event on each of them can disagree; an event on one alone carries over to the merge.
                    node_value = observations.get(node, _UNSET)
                    kept_value = observations.get(kept, _UNSET)
                    if node_value is not _UNSET and kept_value is not _UNSET and node_value != kept_value:
                        return graph, _INCONSISTENT
                    self._merge(graph, kept, node)
                    representative[node] = kept
                    if node in observations:
                        observations.setdefault(kept, observations.pop(node))
                    break
                else:
                    survivors.append(node)

        # Fourth step of Fig. 10: restrict to the ancestors of gamma'. The justification is the one that Step 2 of
        # ID rests on -- what is not an ancestor of the query cannot contribute to it.
        return graph.get_ancestral_graph(set(observations)), observations

    def _parallel_worlds_graph(self, causal_graph, worlds):
        """Build the parallel worlds graph: one copy of ``causal_graph`` per world, sharing the unobserved parents.

        Every observable variable of a causal model is a function of its parents and of an unobserved parent of
        its own, and all submodels of a model share their unobserved variables. The copies of a variable in two
        worlds are therefore always confounded, which is what makes counterfactuals more than a product of
        one-world quantities; in a semi-Markovian graph that shared parent shows up as a bidirected edge between
        the two copies. An unobserved variable shared by two observable ones -- the bidirected edges of
        ``causal_graph`` -- is shared across the worlds in the same way, so it confounds every copy of the first
        with every copy of the second.

        Parameters
        ----------
        causal_graph: ADMG
            ``G``, the causal graph.

        worlds: list of tuple
            The actions mentioned in the query, each a sorted tuple of (variable, value) pairs.

        Returns
        -------
        graph: ADMG
            The parallel worlds graph over ``Counterfactual`` nodes.
        """
        graph = ADMG()
        variables = list(causal_graph.nodes())

        # `edge_list` alone would silently drop the variables of a world that have no edges.
        for world in worlds:
            graph.add_nodes_from(Counterfactual(variable, world) for variable in variables)

        for world in worlds:
            for variable in variables:
                # do(x) cuts every arrow into the variables it fixes, the directed ones and the ones standing for
                # unobserved parents alike.
                if self._is_fixed(Counterfactual(variable, world)):
                    continue
                for parent in causal_graph.get_parents(variable):
                    graph.add_edge(Counterfactual(parent, world), Counterfactual(variable, world), "->")

        # The unobserved parent every variable has of its own, shared by the copies of that variable.
        for variable in variables:
            for world, other_world in combinations(worlds, 2):
                self._add_shared_parent(graph, Counterfactual(variable, world), Counterfactual(variable, other_world))

        # The unobserved parents two variables share, which the copies of those variables all share in turn.
        confounded = [(u, v) for u, v, edge_type in causal_graph.get_edges(data=True) if edge_type == "<>"]
        for (u, v), (world, other_world) in product(confounded, product(worlds, repeat=2)):
            self._add_shared_parent(graph, Counterfactual(u, world), Counterfactual(v, other_world))

        return graph

    @staticmethod
    def _add_shared_parent(graph, first, second):
        """Record that two nodes share an unobserved parent, unless one of them is fixed or they are the same."""
        if first == second or IDStar._is_fixed(first) or IDStar._is_fixed(second):
            return
        if not graph.has_edge(first, second, "<>"):
            graph.add_edge(first, second, "<>")

    def _is_same_variable(self, causal_graph, alpha, beta, representative, observations):
        """Check whether two nodes of the parallel worlds graph are the same random variable, by Lemma 24.

        Both nodes are derived from the same variable of ``causal_graph``, so they share its domain and its
        mechanism, and their parents correspond variable by variable. The lemma then asks that each pair of
        corresponding parents be either the same node or two nodes known to attain the same value, whether that
        value was set by an intervention or observed. Unobserved parents are shared by every world by
        construction, so only the observed ones are left to check.

        Parameters
        ----------
        causal_graph: ADMG
            ``G``, the causal graph.

        alpha, beta: Counterfactual
            The two nodes, derived from the same variable and neither fixed by an intervention.

        representative: dict
            Maps each node merged so far to the node it was merged into.

        observations: dict
            The values the query assigns, as relabelled by the merges made so far.

        Returns
        -------
        same: bool
            True if the two nodes are the same random variable.
        """
        for parent in causal_graph.get_parents(alpha.variable):
            first = self._find(representative, Counterfactual(parent, alpha.intervention))
            second = self._find(representative, Counterfactual(parent, beta.intervention))
            if first == second:
                continue

            first_value = self._value_of(first, observations)
            if first_value is _UNSET or first_value != self._value_of(second, observations):
                return False

        return True

    def _merge(self, graph, keep, drop):
        """Merge two nodes established to be the same random variable into one, by Lemma 25.

        The merged node inherits the mechanism, and with it the parents, of one of the two; the children of both
        become its children. Which of the two is kept is an arbitrary choice the paper leaves open, and the graphs
        the choices lead to are all acceptable counterfactual graphs.

        Parameters
        ----------
        graph: ADMG
            The graph being merged, modified in place.

        keep, drop: Counterfactual
            The node that survives the merge and the node that is absorbed into it.

        Returns
        -------
        None
        """
        for child in graph.get_children(drop):
            if child != keep and not graph.has_edge(keep, child, "->"):
                graph.add_edge(keep, child, "->")

        # A shared unobserved parent of the two merged nodes would leave a bidirected self-loop behind, which is
        # what `_add_shared_parent` skips as well: a variable is not confounded with itself.
        for spouse in graph.get_spouses(drop):
            if spouse != keep and not graph.has_edge(keep, spouse, "<>"):
                graph.add_edge(keep, spouse, "<>")

        # Removing the absorbed node takes its incoming edges with it, leaving the parents of `keep` alone.
        graph.remove_node(drop)

    def _subscript(self, graph, node):
        r"""Return sub(node): the action the counterfactual variable a node denotes is taken under.

        A merge can leave a node with a name that no longer describes it, since the name comes from the world the
        node was created in while the merged node stands for a variable in several of them. What the node is
        taken under is read off the graph instead: the fixed nodes it descends from, which is the
        :math:`W = An(\omega)_{G'} \cap sub(\gamma)` of the paper.

        Parameters
        ----------
        graph: ADMG
            The counterfactual graph.

        node: Counterfactual
            The node whose subscript is sought.

        Returns
        -------
        subscript: set of (variable, value) pairs
        """
        return {
            (ancestor.variable, self._fixed_value(ancestor))
            for ancestor in graph.get_ancestors(node)
            if self._is_fixed(ancestor)
        }

    @staticmethod
    def _free_values(free_nodes, used_values):
        """Name the value of each node the query does not mention, so that Line 6 can sum over it.

        Parameters
        ----------
        free_nodes: set of Counterfactual
            The nodes to name a value for.

        used_values: set
            The values already spoken for, which the new names must stay clear of.

        Returns
        -------
        values: dict
            Maps each node to its value name, the lower-cased name of its variable with as many primes as it
            takes to make it new.
        """
        values = {}
        taken = set(used_values)

        for node in sorted(free_nodes, key=str):
            value = str(node.variable).lower()
            while value in taken:
                value += "'"
            taken.add(value)
            values[node] = value

        return values

    @staticmethod
    def _find(representative, node):
        """Return the node that ``node`` has been merged into, or ``node`` itself if it was never merged."""
        while node in representative:
            node = representative[node]
        return node

    @staticmethod
    def _value_of(node, observations):
        """Return the value a node is known to attain -- set by an intervention or observed -- or ``_UNSET``."""
        if IDStar._is_fixed(node):
            return IDStar._fixed_value(node)
        return observations.get(node, _UNSET)

    @staticmethod
    def _is_fixed(node):
        """Whether the action a node is taken under fixes the very variable the node is derived from."""
        return node.variable in node.action

    @staticmethod
    def _fixed_value(node):
        """The value a fixed node is held at by its own action."""
        return node.action[node.variable]
