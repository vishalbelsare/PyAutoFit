import logging
from abc import ABC

from autoconf import conf
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.non_linear.analysis.multiprocessing import AnalysisPool
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.result import Result
from autofit.non_linear.samples import OptimizerSamples

logger = logging.getLogger(
    __name__
)


class Analysis(ABC):
    """
    Protocol for an analysis. Defines methods that can or
    must be implemented to define a class that compute the
    likelihood that some instance fits some data.
    """

    def log_likelihood_function(self, instance):
        raise NotImplementedError()

    def visualize(self, paths: AbstractPaths, instance, during_analysis):
        pass

    def save_attributes_for_aggregator(self, paths: AbstractPaths):
        pass

    def save_results_for_aggregator(self, paths: AbstractPaths, model: CollectionPriorModel,
                                    samples: OptimizerSamples):
        pass

    def modify_before_fit(self, paths: AbstractPaths, model: AbstractPriorModel):
        """
        Overwrite this method to modify the attributes of the `Analysis` class before the non-linear search begins.

        An example use-case is using properties of the model to alter the `Analysis` class in ways that can speed up
        the fitting performed in the `log_likelihood_function`.
        """
        return self

    def modify_after_fit(self, paths: AbstractPaths, model: AbstractPriorModel, result: Result):
        """
        Overwrite this method to modify the attributes of the `Analysis` class before the non-linear search begins.

        An example use-case is using properties of the model to alter the `Analysis` class in ways that can speed up
        the fitting performed in the `log_likelihood_function`.
        """
        return self

    def make_result(self, samples, model, search):
        return Result(samples=samples, model=model, search=search)

    def profile_log_likelihood_function(self, paths: AbstractPaths, instance):
        """
        Overwrite this function for profiling of the log likelihood function to be performed every update of a
        non-linear search.

        This behaves analogously to overwriting the `visualize` function of the `Analysis` class, whereby the user
        fills in the project-specific behaviour of the profiling.

        Parameters
        ----------
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        instance
            The maximum likliehood instance of the model so far in the non-linear search.
        """
        pass

    def __add__(
            self,
            other: "Analysis"
    ) -> "CombinedAnalysis":
        """
        Analyses can be added together. The resultant
        log likelihood function returns the sum of the
        underlying log likelihood functions.

        Parameters
        ----------
        other
            Another analysis class

        Returns
        -------
        A class that computes log likelihood based on both analyses
        """
        if isinstance(
                other,
                CombinedAnalysis
        ):
            return other + self
        return CombinedAnalysis(
            self, other
        )


class CombinedAnalysis(Analysis):
    def __init__(self, *analyses: Analysis):
        """
        Computes the summed log likelihood of multiple analyses
        applied to a single model.

        Either analyses are performed sequentially and summed,
        or they are mapped out to processes.

        If the number of cores is greater than one then the
        analyses are distributed across a number of processes
        equal to the number of cores.

        Parameters
        ----------
        analyses
        """
        self.analyses = analyses

        n_cores = conf.instance[
            "general"
        ][
            "analysis"
        ][
            "n_cores"
        ]

        if n_cores > 1:
            self.log_likelihood_function = AnalysisPool(
                analyses,
                n_cores
            )
        else:
            self.log_likelihood_function = lambda instance: sum(
                analysis.log_likelihood_function(
                    instance
                )
                for analysis in analyses
            )

    def _for_each_analysis(self, func, paths):
        """
        Convenience function to call an underlying function for each
        analysis with a paths object with an integer attached to the
        end.

        Parameters
        ----------
        func
            Some function of the analysis class
        paths
            An object describing how data should be saved
        """
        for i, analysis in enumerate(self.analyses):
            child_paths = paths.create_child(
                name=f"{paths.name}_{i}"
            )
            func(child_paths, analysis)

    def save_attributes_for_aggregator(self, paths: AbstractPaths):
        def func(child_paths, analysis):
            analysis.save_attributes_for_aggregator(
                child_paths,
            )

        self._for_each_analysis(
            func,
            paths
        )

    def save_results_for_aggregator(
            self,
            paths: AbstractPaths,
            model: CollectionPriorModel,
            samples: OptimizerSamples
    ):
        def func(child_paths, analysis):
            analysis.save_results_for_aggregator(
                child_paths,
                model,
                samples
            )

        self._for_each_analysis(
            func,
            paths
        )

    def visualize(
            self,
            paths: AbstractPaths,
            instance,
            during_analysis
    ):
        """
        Visualise the instance according to each analysis.

        Visualisation output is distinguished by using an integer suffix
        for each analysis path.

        Parameters
        ----------
        paths
            Paths object for overall fit
        instance
            An instance of the model
        during_analysis
            Is this visualisation during analysis?
        """

        def func(child_paths, analysis):
            analysis.visualize(
                child_paths,
                instance,
                during_analysis
            )

        self._for_each_analysis(
            func,
            paths
        )

    def make_result(
        self, samples, model, search
    ):
        return [analysis.make_result(samples, model, search) for analysis in self.analyses]


    def profile_log_likelihood_function(
            self,
            paths: AbstractPaths,
            instance,
    ):
        """
        Profile the log likliehood function of the maximum likelihood model instance using each analysis.
        Profiling output is distinguished by using an integer suffix for each analysis path.
        Parameters
        ----------
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        instance
            The maximum likliehood instance of the model so far in the non-linear search.
        """

        def func(child_paths, analysis):
            analysis.profile_log_likelihood_function(
                child_paths,
                instance,
            )

        self._for_each_analysis(
            func,
            paths
        )

    def __len__(self):
        return len(self.analyses)

    def __add__(self, other: Analysis):
        """
        Adding anything to a CombinedAnalysis results in another
        analysis containing all underlying analyses (no combined
        analysis children)

        Parameters
        ----------
        other
            Some analysis

        Returns
        -------
        An overarching analysis
        """
        if isinstance(
                other,
                CombinedAnalysis
        ):
            return CombinedAnalysis(
                *self.analyses,
                *other.analyses
            )
        return CombinedAnalysis(
            *self.analyses,
            other
        )

    def log_likelihood_function(
            self,
            instance
    ) -> float:
        """
        The implementation of this function is decided in the constructor
        based on the number of cores available

        Parameters
        ----------
        instance
            An instance of a model

        Returns
        -------
        The likelihood that model corresponds to the data encapsulated
        by the child analyses
        """
