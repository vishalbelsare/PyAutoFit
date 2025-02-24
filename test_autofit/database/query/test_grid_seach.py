import pytest

import autofit as af
from autofit import database as db
from autofit.mock.mock import Gaussian


def _make_children(
        grid_id
):
    return [
        db.Fit(
            id=f"child_{grid_id}_{i}",
            instance=Gaussian(
                centre=i
            ),
            model=af.Model(
                Gaussian,
                centre=float(-i)
            ),
            max_log_likelihood=grid_id + i
        )
        for i in range(10)
    ]


@pytest.fixture(
    name="children"
)
def make_children():
    return _make_children(1)


@pytest.fixture(
    name="grid_fit"
)
def make_grid_fit(children):
    return db.Fit(
        id="grid",
        is_grid_search=True,
        children=children,
        instance=Gaussian(
            centre=1
        )
    )


@pytest.fixture(
    autouse=True
)
def add_to_session(
        grid_fit,
        session
):
    session.add(
        grid_fit
    )
    session.flush()


def test_convert_prior():
    assert float(
        af.UniformPrior(
            lower_limit=0,
            upper_limit=1
        )
    ) == 0.5


def test_cell_aggregator(
        aggregator
):
    assert aggregator.grid_searches().cell_number(
        2
    ).values(
        "id"
    ) == [
               "child_1_3"
           ]


def test_model_order_no():
    model_1 = af.Model(
        Gaussian,
        centre=1.0
    )
    model_2 = af.Model(
        Gaussian,
        centre=2.0
    )

    assert model_1.order_no < model_2.order_no


def test_negative():
    model_1 = af.Model(
        Gaussian,
        centre=-3.0
    )
    model_2 = af.Model(
        Gaussian,
        centre=2.0
    )
    assert model_1.order_no < model_2.order_no


def test_model_order_no_complicated():
    model_1 = af.Model(
        Gaussian,
        centre=1.0,
        normalization=af.UniformPrior(0.0, 1.0)
    )
    model_2 = af.Model(
        Gaussian,
        centre=2.0,
        normalization=af.UniformPrior(0.0, 0.5)
    )
    model_3 = af.Model(
        Gaussian,
        centre=2.0,
        normalization=af.UniformPrior(0.0, 1.0)
    )

    assert model_1.order_no < model_2.order_no < model_3.order_no


def test_grid_search_best_fits(
        aggregator
):
    best_fit = aggregator.grid_searches().best_fits()
    assert isinstance(
        best_fit,
        db.GridSearchAggregator
    )
    assert best_fit[0].max_log_likelihood == 10


def test_multiple_best_fits(
        aggregator,
        session
):
    session.add(
        db.Fit(
            id="grid_2",
            is_grid_search=True,
            children=_make_children(2),
            instance=Gaussian(
                centre=1
            )
        )
    )
    session.commit()
    best_fit = aggregator.grid_searches().best_fits()
    assert best_fit.values(
        "max_log_likelihood"
    ) == [10, 11]


def test_grid_search(
        aggregator,
        grid_fit
):
    result, = aggregator.query(
        aggregator.search.is_grid_search
    ).fits

    assert result is grid_fit


def test_grid_searches(
        aggregator
):
    aggregator = aggregator.grid_searches()
    assert isinstance(
        aggregator,
        db.GridSearchAggregator
    )


class TestChildren:
    def test_simple(
            self,
            aggregator,
            children
    ):
        assert aggregator.grid_searches().children().fits == children

    def test_query_after(
            self,
            aggregator
    ):
        results = aggregator.grid_searches().children().query(
            aggregator.centre <= 5
        ).fits
        assert len(results) == 6

    def test_query_before(
            self,
            aggregator,
            grid_fit,
            session
    ):
        session.add(
            db.Fit(
                id="grid2",
                is_grid_search=True,
                instance=Gaussian(
                    centre=2
                )
            )
        )
        session.flush()

        parent_aggregator = aggregator.query(
            aggregator.search.is_grid_search & (aggregator.centre == 1)
        )

        result, = parent_aggregator.fits

        assert result is grid_fit

        child_aggregator = parent_aggregator.grid_searches().children()

        results = child_aggregator.fits
        assert len(results) == 10

        results = aggregator.query(
            aggregator.search.is_grid_search & (aggregator.centre == 2)
        ).grid_searches().children().fits
        assert len(results) == 0
