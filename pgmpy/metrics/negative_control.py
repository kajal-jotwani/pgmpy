import numpy as np
import pandas as pd

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import SHD, _BaseSupervisedMetric, _BaseUnsupervisedMetric


class NegativeControl(_BaseUnsupervisedMetric):
    """
        Simulation-based negative control for evaluating causal discovery
        algorithms against random guessing (Section 6, Petersen 2025).

        Core question the metric answers: Is your algorithm doing better than random guessing?
    :
          1. Run your algorithm on real data X → get Ĝ_algo and its edge
             count m_est. Score it: f(G_true, Ĝ_algo).
          2. Generate b random DAGs (or CPDAGs) with m_est edges each.
          3. Score each random graph directly: f(G_true, Ĝ_NC_i).
          4. p = fraction of random scores that are at least as good as
             the algorithm's score.

        Parameters
        ----------
        causal_discovery_algo : class
            A pgmpy estimator *class* (not an instance), e.g. ``PC``.
            Must accept ``data`` as first arg and expose ``.estimate()``.

        algo_kwargs : dict, optional
            Passed to ``causal_discovery_algo.__init__``.

        estimate_kwargs : dict, optional
            Passed to ``.estimate()``.

        base_metric : _BaseSupervisedMetric, optional
            Metric f(G_true, G_est). Defaults to ``SHD()``.

        n_simulations : int, default 1000
            Number of negative control random graphs (b in the paper).

        seed : int, optional
            Random seed for reproducibility.

        show_progress : bool, default True
            Show tqdm progress bar.

        Returns
        -------
        dict
            ``observed_score``  – f(G_true, Ĝ_algo)
            ``nc_scores``       – np.ndarray of f(G_true, Ĝ_NC_i) values
            ``nc_mean``         – mean of nc_scores
            ``nc_ci``           – (2.5th, 97.5th) percentile tuple
            ``p_value``         – fraction of NC scores ≥ algo score
                                  (or ≤ for lower-is-better).
                                  p < 0.05 → algo beats random guessing.

        Examples
        --------
        >>> from pgmpy.metrics.negative_control import NegativeControl
        >>> from pgmpy.metrics import SHD
        >>> from pgmpy.estimators import PC
        >>> nc = NegativeControl(
        ...     causal_discovery_algo=PC,
        ...     base_metric=SHD(),
        ...     n_simulations=1000,
        ...     seed=42,
        ... )
        >>> result = nc.evaluate(X=data, causal_graph=true_dag)
        >>> result["p_value"]   # < 0.05 means algo beats random guessing

        References
        ----------
        .. [1] Petersen, A. H. (2025). Are you doing better than random
               guessing? A call for using negative controls when evaluating
               causal discovery algorithms. UAI 2025.
               https://arxiv.org/abs/2412.10039
    """

    _tags = {
        "name": "negative_control",
        "requires_true_graph": False,
        "requires_data": True,
        "lower_is_better": False,  # small p_value is GOOD
        "supported_graph_types": (DAG, PDAG),
        "is_default": False,
    }

    def __init__(
        self,
        causal_discovery_algo,
        algo_kwargs: dict | None = None,
        estimate_kwargs: dict | None = None,
        base_metric: _BaseSupervisedMetric | None = None,
        n_simulations: int = 1000,
        seed: int | None = None,
        show_progress: bool = True,
    ):
        self.causal_discovery_algo = causal_discovery_algo
        self.algo_kwargs = algo_kwargs or {}
        self.estimate_kwargs = estimate_kwargs or {}
        self.base_metric = base_metric if base_metric is not None else SHD()

        self.n_simulations = n_simulations
        self.seed = seed
        self.show_progress = show_progress
        super().__init__()

    def _run_discovery(self, data: pd.DataFrame):
        """Instantiate algo with data and call .estimate()."""
        return self.causal_discovery_algo(data, **self.algo_kwargs).estimate(**self.estimate_kwargs)

    def _to_scalar(self, score) -> float:
        """Extract a float from a metric return value (float or dict)."""
        if isinstance(score, dict):
            return float(next(iter(score.values())))
        return float(score)

    def _evaluate(self, X: pd.DataFrame, causal_graph) -> dict:
        """
        Parameters
        ----------
        X : pd.DataFrame
            Observed data. Used to run the discovery algorithm (Step 1).

        causal_graph : DAG or PDAG
            The ground truth graph G_true.
            Used only for scoring — not for algorithm training.
            This is why the class is _BaseUnsupervisedMetric.
        """
        rng = np.random.default_rng(self.seed)
        nodes = list(causal_graph.nodes())
        d = len(nodes)

        # STEP 1: Run algo on real data to get Ĝ_algo and its edge count m_est.
        est_graph = self._run_discovery(X)
        m_est = est_graph.number_of_edges()

        # Detect whether the algorithm returned a PDAG/CPDAG.
        is_pdag = isinstance(est_graph, PDAG)

        # Score: f(G_true, Ĝ_algo)
        observed_score = self._to_scalar(
            self.base_metric(
                true_causal_graph=causal_graph,
                est_causal_graph=est_graph,
            )
        )
        # Step 2 — generate random DAG with m_est edges.
        # Step 3 — score it DIRECTLY against G_true: f(G_true, Ĝ_NC_i).

        # Pre-generate all seeds for reproducibility
        sim_seeds = rng.integers(0, 2**32 - 1, size=self.n_simulations)

        nc_scores = []

        if self.show_progress:
            try:
                from tqdm import tqdm

                iter_range = tqdm(
                    range(self.n_simulations),
                    desc="Negative control simulations",
                    unit="sim",
                )
            except ImportError:
                iter_range = range(self.n_simulations)
        else:
            iter_range = range(self.n_simulations)

        for i in iter_range:
            # Step 2: Generate random Erdős-Rényi DAG with exactly m_est edges.
            # DAG.get_random(n_edges=k) generates a random DAG with exactly k edges.
            nc_graph = DAG.get_random(
                n_nodes=d,
                n_edges=m_est,
                node_names=nodes,
                seed=int(sim_seeds[i]),
            )

            if is_pdag:
                nc_graph = nc_graph.to_pdag()

            # Step 3: Score f(G_true, Ĝ_NC_i) — DIRECT comparison.
            # No simulate, no re-run.
            nc_scores.append(
                self._to_scalar(
                    self.base_metric(
                        true_causal_graph=causal_graph,
                        est_causal_graph=nc_graph,
                    )
                )
            )

        nc_scores = np.array(nc_scores, dtype=float)

        # STEP 4: Empirical p-value calculation:
        # p = (1/b) * Σ 1[f(G_true, Ĝ_algo) ≤ f(G_true, Ĝ_NC_i)]
        # for lower-is-better metrics (e.g. SHD).
        # Reverse inequality for higher-is-better (e.g. precision).

        lower_is_better = self.base_metric._tags.get("lower_is_better", True)

        if len(nc_scores) == 0:
            p_value = np.nan
        elif lower_is_better:
            # Fraction of NC scores ≤ algo score
            # (random guessing matched or beat the algo)
            p_value = float(np.mean(nc_scores <= observed_score))
        else:
            # Fraction of NC scores ≥ algo score
            p_value = float(np.mean(nc_scores >= observed_score))

        return {
            "observed_score": observed_score,
            "nc_scores": nc_scores,
            "nc_mean": float(np.mean(nc_scores)),
            "nc_ci": (
                float(np.percentile(nc_scores, 2.5)),
                float(np.percentile(nc_scores, 97.5)),
            ),
            "p_value": p_value,
        }
