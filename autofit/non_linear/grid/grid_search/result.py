from typing import List

import numpy as np

from autofit import exc
from autofit.mapper import model_mapper as mm
from autofit.mapper.prior.prior import Prior
from autofit.non_linear.result import Result

LimitLists = List[List[float]]


class GridSearchResult:

    def __init__(
            self,
            results: List[Result],
            lower_limits_lists: LimitLists,
            grid_priors: List[Prior]
    ):
        """
        The result of a grid search.

        Parameters
        ----------
        results
            The results of the non linear optimizations performed at each grid step
        lower_limits_lists
            A list of lists of values representing the lower bounds of the grid searched values at each step
        """
        self.lower_limits_lists = lower_limits_lists
        self.results = results
        self.no_dimensions = len(self.lower_limits_lists[0])
        self.no_steps = len(self.lower_limits_lists)
        self.side_length = int(self.no_steps ** (1 / self.no_dimensions))
        self.step_size = 1 / self.side_length
        self.grid_priors = grid_priors

    @property
    def physical_lower_limits_lists(self) -> LimitLists:
        """
        The lower physical values for each grid square
        """
        return self._physical_values_for(
            self.lower_limits_lists
        )

    @property
    def physical_centres_lists(self) -> LimitLists:
        """
        The middle physical values for each grid square
        """
        return self._physical_values_for(
            self.centres_lists
        )

    @property
    def physical_upper_limits_lists(self) -> LimitLists:
        """
        The upper physical values for each grid square
        """
        return self._physical_values_for(
            self.upper_limits_lists
        )

    @property
    def upper_limits_lists(self) -> LimitLists:
        """
        The upper values for each grid square
        """
        return [
            [
                limit + self.step_size
                for limit in limits
            ]
            for limits in self.lower_limits_lists
        ]

    @property
    def centres_lists(self) -> LimitLists:
        """
        The centre values for each grid square
        """
        return [
            [
                (upper + lower) / 2
                for upper, lower
                in zip(upper_limits, lower_limits)
            ]
            for upper_limits, lower_limits in zip(
                self.lower_limits_lists,
                self.upper_limits_lists
            )
        ]

    def _physical_values_for(
            self,
            unit_lists: LimitLists
    ) -> LimitLists:
        """
        Compute physical values for lists of lists of unit hypercube
        values.

        Parameters
        ----------
        unit_lists
            A list of lists of hypercube values

        Returns
        -------
        A list of lists of physical values
        """
        return [
            [
                prior.value_for(
                    limit
                )
                for prior, limit in
                zip(
                    self.grid_priors,
                    limits
                )
            ]
            for limits in unit_lists
        ]

    def __setstate__(self, state):
        return self.__dict__.update(state)

    def __getstate__(self):
        return self.__dict__

    def __getattr__(self, item: str) -> object:
        """
        We default to getting attributes from the best result. This allows promises to reference best results.
        """
        return getattr(self.best_result, item)

    @property
    def shape(self):
        return tuple([
            self.side_length
            for _ in range(
                self.no_dimensions
            )
        ])

    @property
    def best_result(self):
        """
        The best result of the grid search. That is, the result output by the non linear search that had the highest
        maximum figure of merit.

        Returns
        -------
        best_result: Result
        """
        best_result = None
        for result in self.results:
            if (
                    best_result is None
                    or result.log_likelihood > best_result.log_likelihood
            ):
                best_result = result
        return best_result

    @property
    def best_model(self):
        """
        Returns
        -------
        best_model: mm.ModelMapper
            The model mapper instance associated with the highest figure of merit from the grid search
        """
        return self.best_result.model

    @property
    def all_models(self):
        """
        Returns
        -------
        all_models: [mm.ModelMapper]
            All model mapper instances used in the grid search
        """
        return [result.model for result in self.results]

    @property
    def physical_step_sizes(self):

        physical_step_sizes = []

        # TODO : Make this work for all dimensions in a less ugly way.

        for dim in range(self.no_dimensions):

            values = [value[dim] for value in self.physical_lower_limits_lists]
            diff = [abs(values[n] - values[n - 1]) for n in range(1, len(values))]

            if dim == 0:
                physical_step_sizes.append(np.max(diff))
            elif dim == 1:
                physical_step_sizes.append(np.min(diff))
            else:
                raise exc.GridSearchException(
                    "This feature does not support > 2 dimensions"
                )

        return tuple(physical_step_sizes)

    @property
    def results_reshaped(self):
        """
        Returns
        -------
        likelihood_merit_array: np.ndarray
            An arrays of figures of merit. This arrays has the same dimensionality as the grid search, with the value in
            each entry being the figure of merit taken from the optimization performed at that point.
        """
        return np.reshape(
            np.array([result for result in self.results]),
            tuple(self.side_length for _ in range(self.no_dimensions)),
        )

    @property
    def max_log_likelihood_values(self):
        """
        Returns
        -------
        likelihood_merit_array: np.ndarray
            An arrays of figures of merit. This arrays has the same dimensionality as the grid search, with the value in
            each entry being the figure of merit taken from the optimization performed at that point.
        """
        return np.reshape(
            np.array([result.log_likelihood for result in self.results]),
            tuple(self.side_length for _ in range(self.no_dimensions)),
        )

    @property
    def log_evidence_values(self):
        """
        Returns
        -------
        likelihood_merit_array: np.ndarray
            An arrays of figures of merit. This arrays has the same dimensionality as the grid search, with the value in
            each entry being the figure of merit taken from the optimization performed at that point.
        """
        return np.reshape(
            np.array([result.samples.log_evidence for result in self.results]),
            tuple(self.side_length for _ in range(self.no_dimensions)),
        )
