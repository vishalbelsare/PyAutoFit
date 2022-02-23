import pytest

import autofit as af
from autofit.non_linear.grid import sensitivity as s


def test_lists(sensitivity):
    assert len(list(sensitivity._perturbation_instances)) == 8


def test_tuple_step_size(sensitivity):
    sensitivity.number_of_steps = (2, 2, 4)

    assert len(sensitivity._lists) == 16


def test_labels(sensitivity):
    labels = list(sensitivity._labels)
    assert labels == [
        "centre_0.25_normalization_0.25_sigma_0.25",
        "centre_0.25_normalization_0.25_sigma_0.75",
        "centre_0.25_normalization_0.75_sigma_0.25",
        "centre_0.25_normalization_0.75_sigma_0.75",
        "centre_0.75_normalization_0.25_sigma_0.25",
        "centre_0.75_normalization_0.25_sigma_0.75",
        "centre_0.75_normalization_0.75_sigma_0.25",
        "centre_0.75_normalization_0.75_sigma_0.75",
    ]


def test_searches(sensitivity):
    assert len(list(sensitivity._searches)) == 8


def test_perform_job(job):
    result = job.perform()
    assert isinstance(result, s.JobResult)
    assert isinstance(result.perturbed_result, af.Result)
    assert isinstance(result.result, af.Result)


def test_job_paths(
        job,
        search
):
    output_path = search.paths.output_path
    assert job.perturbed_search.paths.output_path == f"{output_path}/[perturbed]"
    assert job.search.paths.output_path == f"{output_path}/[base]"


class TestPerturbationModels:
    @pytest.mark.parametrize(
        "limit_scale, fl, fu, sl, su",
        [
            (1.0, 0.0, 0.5, 0.5, 1.0,),
            (2.0, -0.25, 0.75, 0.25, 1.25,),
            (4.0, -0.75, 1.25, -0.25, 1.75,),
        ]
    )
    def test_perturbation_models(
            self,
            sensitivity,
            limit_scale,
            fl, fu, sl, su
    ):
        sensitivity.limit_scale = limit_scale
        jobs = sensitivity._make_jobs()
        models = [
            job.perturbation_model
            for job in jobs
        ]

        first, second, *_ = models

        assert first is not second

        assert first.sigma.lower_limit == fl
        assert first.sigma.upper_limit == fu
        assert second.sigma.lower_limit == sl
        assert second.sigma.upper_limit == su

    def test_model_with_limits(self):
        model = af.Model(af.Gaussian)

        with_limits = model.with_limits([
            (3, 5),
            (3, 5),
            (3, 5),
        ])
        assert with_limits.centre.lower_limit == 3
        assert with_limits.centre.upper_limit == 5

    def test_prior_with_limits(self):
        prior = af.UniformPrior().with_limits(3, 5)
        assert prior.lower_limit == 3
        assert prior.upper_limit == 5
