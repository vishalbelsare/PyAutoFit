import pytest

import autofit as af


@pytest.fixture(
    name="uniform_dict"
)
def make_uniform_dict():
    return {
        "type": "Uniform",
        "lower_limit": 2,
        "upper_limit": 3
    }


@pytest.fixture(
    name="uniform_prior"
)
def make_uniform_prior(
        uniform_dict
):
    return af.Prior.from_dict(
        uniform_dict
    )


@pytest.fixture(
    name="log_uniform_dict"
)
def make_log_uniform_dict():
    return {
        "type": "LogUniform",
        "lower_limit": 0.2,
        "upper_limit": 0.3
    }


@pytest.fixture(
    name="log_uniform_prior"
)
def make_log_uniform_prior(
        log_uniform_dict
):
    return af.Prior.from_dict(
        log_uniform_dict
    )


@pytest.fixture(
    name="gaussian_dict"
)
def make_gaussian_dict():
    return {
        "type": "Gaussian",
        "lower_limit": -10,
        "upper_limit": 10,
        "mean": 3,
        "sigma": 4
    }


@pytest.fixture(
    name="gaussian_prior"
)
def make_gaussian_prior(
        gaussian_dict
):
    return af.Prior.from_dict(
        gaussian_dict
    )


class TestDict:
    pass


class TestFromDict:
    def test_uniform(self, uniform_prior):
        assert isinstance(uniform_prior, af.UniformPrior)
        assert uniform_prior.lower_limit == 2
        assert uniform_prior.upper_limit == 3

    def test_log_uniform(self, log_uniform_prior):
        assert isinstance(log_uniform_prior, af.LogUniformPrior)
        assert log_uniform_prior.lower_limit == 0.2
        assert log_uniform_prior.upper_limit == 0.3

    def test_gaussian(self, gaussian_prior):
        assert isinstance(gaussian_prior, af.GaussianPrior)
        assert gaussian_prior.lower_limit == -10
        assert gaussian_prior.upper_limit == 10
        assert gaussian_prior.mean == 3
        assert gaussian_prior.sigma == 4
