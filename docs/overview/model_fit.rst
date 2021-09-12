.. _model_fit:

Fitting a Model
===============

To illustrate **PyAutoFit** we'll use the example modeling problem of fitting a 1D Gaussian profile to noisy data.

To begin, lets import ``autofit`` (and ``numpy``) using the convention below:

.. code-block:: bash

    import autofit as af
    import numpy as np

Data
----

The example ``data`` with errors (black) and the model-fit (red), are shown below:

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/master/docs/images/toy_model_fit.png
  :width: 600
  :alt: Alternative text

Model
-----

We now need to define a 1D Gaussian profile as a **PyAutoFit** *model-component*, where a *model component* is a
component of the model we fit to the ``data``. It has associated with it a set of *parameters* which are varied for
during a *model-fit*, which is performed using a *non-linear search*.

*Model components* are defined using Python classes using the format below, where the class name is
the *model component* name and the constructor arguments are its *parameters*.

.. code-block:: bash

    class Gaussian:

        def __init__(
            self,
            centre=0.0,     # <- PyAutoFit recognises these
            normalization=0.1,  # <- constructor arguments are
            sigma=0.01,     # <- the Gaussian's parameters.
        ):

            self.centre = centre
            self.normalization = normalization
            self.sigma = sigma

The code above defines a **PyAutoFit** *model component* called a ``Gaussian``. When used for *model-fitting* it has
three parameters: ``centre``, ``normalization`` and ``sigma``.

When we fit the model to ``data`` and compute a likelihood an instance of the class above is accessible, with specific
values of ``centre``, ``normalization`` and ``sigma`` chosen by the non-linear search algorithm that fits the model to
the data.

This means that the class's functions are available to compute the likelihood, so lets add a ``profile_from_xvalues``
function that generates the 1D profile from the ``Gaussian``.

.. code-block:: bash

    class Gaussian:
        def __init__(
            self,
            centre=0.0,     # <- PyAutoFit recognises these
            normalization=0.1,  # <- constructor arguments are
            sigma=0.01,     # <- the Gaussian's parameters.
        ):

            self.centre = centre
            self.normalization = normalization
            self.sigma = sigma

        def profile_from_xvalues(self, xvalues):

            transformed_xvalues = xvalues - self.centre

            return np.multiply(
                np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
                np.exp(-0.5 * np.square(np.divide(transformed_xvalues, self.sigma))),
            )

We use the ``Model`` object to compose the model, which in this case is a single ``Gaussian``.  The model is
defined with 3 free parameters, thus the dimensionality of non-linear parameter space is 3.

.. code-block:: bash

    model = af.Model(Gaussian)

Complex high dimensional models can be built from these individual model components, as described in
the `model composition overview page <https://pyautofit.readthedocs.io/en/latest/overview/model_complex.html>`_

Analysis
--------

Now we've defined our model, we need to tell **PyAutoFit** how to fit the model to data. This requires us to
define a **PyAutoFit** ``Analysis`` class:

.. code-block:: bash

    class Analysis(af.Analysis):

        def __init__(self, data, noise_map):

            super().__init__()

            self.data = data
            self.noise_map = noise_map

        def log_likelihood_function(self, instance):

            """
            The 'instance' that comes into this method is an instance of the Gaussian
            class, whose parameters were chosen by our non-linear search.

            The the print statements below will illustrate this when a model-fit is performed!
            """

            print("Gaussian Instance:")
            print("Centre = ", instance.centre)
            print("normalization = ", instance.normalization)
            print("Sigma = ", instance.sigma)

            """
            Get the range of x-values the data is defined on, to evaluate the model
            of the Gaussian.
            """

            xvalues = np.arange(self.data.shape[0])

            """
            Use these xvalues to create model_data of our Gaussian.
            """

            model_data = instance.profile_from_xvalues(xvalues=xvalues)

            """
            Fit the model gaussian to the data, computing the residuals, chi-squareds
            and returning the log likelihood value to the non-linear search.
            """

            residual_map = self.data - model_data
            chi_squared_map = (residual_map / self.noise_map) ** 2.0
            log_likelihood = -0.5 * sum(chi_squared_map)

            return log_likelihood

Lets consider exactly what is happening in the ``Analysis`` class above.

- The ``data`` is passed into the constructor of the ``Analysis`` class. Above, only ``data`` and a ``noise_map`` are
  input, but the constructor can be easily extended to add other parts of the dataset.

- The ``log_likelihood_function`` receives an ``instance`` of the model, which in this example is an ``instance`` of the
  ``Gaussian`` class. This ``instance`` has values for its *parameters* (``centre``, ``normalization`` and ``sigma``) which
  are chosen by the non-linear search used to fit the model, as discussed next.

- The ``log_likelihood_function`` returns a log likelihood value, which the non-linear search uses evaluate the
  goodness-of-fit of a model to the data when sampling parameter space.

Non-Linear Search
-----------------

Next, we *compose* our model, set up our ``Analysis`` and fit the model to the ``data`` using a non-linear search:

.. code-block:: bash

    model = af.Model(Gaussian)
    analysis = Analysis(data=data, noise_map=noise_map)

    emcee = af.Emcee(name="example_search")

    result = emcee.fit(model=model, analysis=analysis)

We perform the fit using the non-linear search algorithm `emcee <https://github.com/dfm/emcee>`_. We cover
non-linear search's in more detail in the `non-linear search overview page <https://pyautofit.readthedocs.io/en/latest/overview/non_linear_search.html>`_.

Result
------

By running the code above **PyAutoFit** performs the model-fit, outputting all results into structured paths on you
hard-disk. It also returns a ``Result`` object in Python, which includes lists containing the non-linear search's
parameter samples, the maximum likelihood model, marginalized parameters estimates, errors are so on:

.. code-block:: bash

    print(result.samples.parameter_lists)
    print(result.samples.max_log_likelihood_vector)
    print(result.samples.median_pdf_vector)
    print(result.samples.error_vector_at_sigma)

It can even return *instances* of the ``Gaussian`` class using the values of the model results:

.. code-block:: bash

    instance = result.max_log_likelihood_instance

    print("Maximum Likelihood Gaussian Instance:")
    print("Centre = ", instance.centre)
    print("normalization = ", instance.normalization)
    print("Sigma = ", instance.sigma)

This can be used to straight forwardly plot the model fit to the data:

.. code-block:: bash

    instance = result.max_log_likelihood_instance

    model_data = instance.profile_from_xvalues(xvalues=np.arange(data.shape[0]))

    plt.plot(range(data.shape[0]), data)
    plt.plot(range(data.shape[0]), model_data)

Results are covered in more detail in the `result overview page <https://pyautofit.readthedocs.io/en/latest/overview/result.html>`_.

Wrap-Up
-------

This completes our introduction to the **PyAutoFit** API. Next, we'll cover how to *compose* and *fit*
models using multiple *model components* and *customize* the model parameterization.

If you'd like to perform the fit shown in this script, checkout the
`simple examples <https://github.com/Jammy2211/autofit_workspace/tree/master/notebooks/overview/simplee>`_ on the
``autofit_workspace``. We detail how **PyAutoFit** works in the first 3 tutorials of
the `HowToFit lecture series <https://pyautofit.readthedocs.io/en/latest/howtofit/howtofit.html>`_.