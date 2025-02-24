from autoconf.exc import ConfigException, PriorException


class FitException(Exception):
    """
    An exception to be thrown if the non linear search must resample; equivalent to returning an infinitely bad fit
    """

    pass


class PriorLimitException(FitException, PriorException):
    pass


class PipelineException(Exception):
    pass


class DeferredInstanceException(Exception):
    """
    Exception raised when an attempt is made to access an attribute or function of a
    deferred instance prior to instantiation
    """

    pass


class AggregatorException(Exception):
    pass


class GridSearchException(Exception):
    pass
