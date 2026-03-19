from pgmpy.independencies import IndependenceAssertion

from ._base import _BaseCITest


class IndependenceMatch(_BaseCITest):
    """
    Check if `X ⊥⊥ Y | Z` is in `independences`.

    This method is implemented to have a uniform API when the independences
    are provided explicitly instead of being inferred from data.

    Parameters
    ----------
    independencies : pgmpy.independencies.Independencies
        The object containing the known independencies.
    """

    _tags = {
        "name": "independence_match",
        "data_types": ("discrete", "continuous", "mixed"),
        "default_for": None,
        "requires_data": False,
    }

    def __init__(self, independencies=None):
        self.independencies = independencies
        super().__init__()

    def is_independent(
        self,
        X: str,
        Y: str,
        Z: list | tuple = (),
        significance_level: float = 0.05,
    ) -> bool:
        """
        Test independence assertion.

        Returns
        -------
        bool
            True if the independence assertion is present in `independencies`, else False.
        """
        self._validate_inputs(X, Y, Z)

        if self.independencies is None:
            raise ValueError("independencies must be provided in __init__.")

        return IndependenceAssertion(X, Y, list(Z)) in self.independencies
