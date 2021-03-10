import matplotlib.pyplot as plt
import numpy as np
import pytest

import autofit as af
from autofit import graphical as g
from autofit.mock.mock import Gaussian
from test_autofit.graphical.gaussian.model import Analysis

x = np.arange(100)
n = 2

should_plot = True


@pytest.fixture(
    name="centre_model"
)
def make_centre_model():
    return af.PriorModel(
        af.GaussianPrior,
        mean=af.GaussianPrior(
            mean=50,
            sigma=10
        ),
        sigma=af.GaussianPrior(
            mean=20,
            sigma=5
        )
    )


def test_embedded_priors(
        centre_model
):
    assert isinstance(
        centre_model.random_instance().value_for(0.5),
        float
    )


def test_hierarchical_factor(
        centre_model
):
    factor = g.HierarchicalFactor(
        centre_model,
        af.GaussianPrior(50, 10)
    )

    assert len(factor.priors) == 3

    laplace = g.LaplaceFactorOptimiser()

    gaussian = factor.optimise(laplace).collection
    assert gaussian.instance_from_prior_medians().mean == pytest.approx(50, abs=0.1)


@pytest.fixture(
    name="centres"
)
def make_centre(
        centre_model
):
    centres = list()
    for _ in range(n):
        centres.append(
            centre_model.random_instance().value_for(0.5)
        )
    return centres


@pytest.fixture(
    name="data"
)
def generate_data(
        centres
):
    data = []
    for centre in centres:
        gaussian = Gaussian(
            centre=centre,
            intensity=20,
            sigma=10,
        )

        data.append(
            gaussian(x)
        )
    return data


def test_generate_data(
        data
):
    if should_plot:
        for gaussian in data:
            plt.plot(x, gaussian)
        plt.show()


def test_model_factor(
        data,
        centres
):
    centre_argument = af.GaussianPrior(
        mean=50,
        sigma=2
    )
    prior_model = af.PriorModel(
        Gaussian,
        centre=centre_argument,
        intensity=20,
        sigma=2
    )
    factor = g.AnalysisFactor(
        prior_model,
        analysis=Analysis(
            centre=centres[0]
        )
    )

    xs = np.linspace(-100, 100)
    loglikes = [factor({centre_argument: x}) for x in xs]
    plt.plot(xs, loglikes)
    plt.show()

    laplace = g.LaplaceFactorOptimiser()

    gaussian = factor.optimise(laplace).collection
    assert gaussian.centre.mean == pytest.approx(centres[0], abs=0.5)


def test_full_fit(centre_model, data, centres):
    graph = g.FactorGraphModel()
    for centre in centres:
        centre_argument = af.GaussianPrior(
            mean=50,
            sigma=2
        )
        prior_model = af.PriorModel(
            Gaussian,
            centre=centre_argument,
            intensity=af.GaussianPrior(
                mean=20,
                sigma=2
            ),
            sigma=2
        )
        graph.add(
            g.AnalysisFactor(
                prior_model,
                analysis=Analysis(
                    centre
                )
            )
        )
        graph.add(
            g.HierarchicalFactor(
                centre_model,
                centre_argument
            )
        )

    laplace = g.LaplaceFactorOptimiser()

    history = g.expectation_propagation.EPHistory()

    collection = graph.optimise(
        laplace,
        max_steps=10,
        callback=history
    ).collection

    # plt.plot(
    #     [
    #         meanfield.log_evidence
    #         for meanfield in history.history.values()
    #     ]
    # )
    i = np.arange(len(history.history))

    f, (ax1, ax2) = plt.subplots(2)
    ax1.plot([meanfield.log_evidence for meanfield in history.history.values()])

    for variable in set(graph.prior_model.priors):
        ax2.errorbar(
            i,
            [meanfield.mean_field[variable].mean for meanfield in history.history.values()],
            [meanfield.mean_field[variable].sigma for meanfield in history.history.values()],
            label=".".join(
                graph.prior_model.path_for_prior(
                    variable
                )
            )
        )

    ax2.legend()
    plt.show()

    for gaussian, centre in zip(
            collection.with_prefix(
                "AnalysisFactor"
            ),
            centres
    ):
        assert gaussian.instance_from_prior_medians().centre == pytest.approx(
            centre,
            abs=0.5
        )
