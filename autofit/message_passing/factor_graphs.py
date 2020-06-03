from collections import \
    (
    defaultdict, ChainMap, Counter
)
from itertools import chain, count, repeat
from typing import (
    NamedTuple, Callable, Tuple, Dict, Set, Union,
    Collection, Optional
)

import numpy as np

from autofit.message_passing.utils import add_arrays

_plate_ids = count()


class Plate:
    def __init__(self, index=slice(None), id_: Optional[int] = None,
                 name: Optional[str] = None):
        self.index = index
        self.id = next(_plate_ids) if id_ is None else id_
        self.name = f"plate_{self.id}" if name is None else name

    def __getitem__(self, args):
        return self(args)

    def set_name(self, name: str) -> "Plate":
        self.name = name
        return self

    def __call__(self, args):
        return type(self)(args)

    def __repr__(self):
        return f"{type(self).__name__}({self.index}, name={self.name})"

    def __eq__(self, other):
        if type(self) is type(other):
            return self.id == other.id
        else:
            return False

    def __hash__(self):
        return self.id


plate = Plate()


class Variable:
    __slot__ = ("name")

    def __init__(self, name: str, *plates):
        self.name = name
        self.plates = plates

    def __repr__(self):
        args = ", ".join(chain([self.name], map(repr, self.plates)))
        return f"{self.__class__.__name__}({args})"

    def __hash__(self):
        return hash((self.name, type(self)))

    @property
    def ndim(self):
        return len(self.plates)


class Factor(NamedTuple):
    factor: Callable
    name: str
    vectorised: bool = True

    def call_factor(self, *args, **kwargs):
        return self.factor(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return FactorNode(self, *args, **kwargs)

    def __hash__(self):
        return hash((self.name, self.factor))


# couldn't redefine __new__ for Factor
def factor(factor: Callable, name: Optional[str] = None,
           _vectorised: bool = True) -> Factor:
    if name is None:
        name = factor.__name__
    return Factor(factor, name, _vectorised)


class JacobianValue(NamedTuple):
    log_value: Dict[str, np.ndarray]
    deterministic_values: Dict[Tuple[str, str], np.ndarray]


def numerical_jacobian(factor: "FactorNode",
                       *args: Tuple[str, ...],
                       _eps: float = 1e-6,
                       _calc_deterministic: bool = True,
                       **kwargs: Dict[str, np.array]
                       ) -> JacobianValue:
    """Calculates the numerical Jacobian of the passed factor

    the arguments passed correspond to the variable that we want
    to take the derivatives for

    the values must be passed as keywords

    _eps specifies the numerical accuracy of the numerical derivative

    returns a jac = JacobianValue namedtuple

    jac.log_value stores the Jacobian of the factor output as a dictionary
    where the keys are the names of the variables

    jac.determinisic_values store the Jacobian of the deterministic variables
    where the keys are a Tuple[str, str] pair of the variable and deterministic
    variables

    Example
    -------
    >>> import numpy as np
    >>> y_ = Variable('y')
    >>> A = np.arange(4).reshape(2, 2)
    >>> dot = factor(lambda x: A.dot(x))(x_) == y_
    >>> dot.jacobian('x', x=[1, 2])
    JacobianValue(
        log_value={'x': array([[0.], [0.]])},
        deterministic_values={('x', 'y'): array([[0., 2.], [1., 3.]])})
    """
    # copy the input array
    p0 = {v: np.array(x, dtype=float) for v, x in kwargs.items()}
    f0 = factor(**p0)
    log_f0 = f0.log_value
    det_vars0 = f0.deterministic_values

    jac_f = {
        v: np.empty(np.shape(kwargs[v]) + np.shape(log_f0))
        for v in args}
    if _calc_deterministic:
        jac_det = {
            (det, v): np.empty(
                np.shape(val) + np.shape(kwargs[v]))
            for v in args
            for det, val in det_vars0.items()}
        det_slices = {
            v: (slice(None),) * np.ndim(a) for v, a in kwargs.items()}
    else:
        jac_det = {}

    for v in args:
        x0 = p0[v]
        if x0.shape:
            inds = tuple(a.ravel() for a in np.indices(x0.shape))
            for ind in zip(*inds):
                x0[ind] += _eps
                p0[v] = x0
                f = factor(**p0)
                x0[ind] -= _eps

                jac_f[v][ind] = (f.log_value - log_f0) / _eps
                if _calc_deterministic:
                    det_vars = f.deterministic_values
                    for det, val in det_vars.items():
                        jac_det[det, v][det_slices[v] + ind] = \
                            (val - det_vars0[det]) / _eps
        else:
            p0[v] += _eps
            f = factor(**p0)
            p0[v] -= _eps

            jac_f[v] = (f.log_value - log_f0) / _eps
            if _calc_deterministic:
                det_vars = f.deterministic_values
                for det, val in det_vars.items():
                    jac_det[det, v] = (val - det_vars0[det]) / _eps

    return JacobianValue(jac_f, jac_det)


def numerical_hessdiag(factor: "FactorNode",
                       *args: Tuple[str, ...],
                       _eps: float = 1e-6,
                       _calc_deterministic: bool = True,
                       **kwargs: Dict[str, np.array]
                       ) -> JacobianValue:
    # copy the input array
    p0 = {v: np.array(x, dtype=float) for v, x in kwargs.items()}
    f0 = factor(**p0)
    log_f0 = f0.log_value
    det_vars0 = f0.deterministic_values

    hess_f = {
        v: np.empty(np.shape(kwargs[v]) + np.shape(log_f0))
        for v in args}
    if _calc_deterministic:
        hess_det = {
            (v, det): np.empty(
                np.shape(kwargs[v]) + np.shape(val))
            for v in args
            for det, val in det_vars0.items()}
    else:
        hess_det = {}

    for v in args:
        x0 = p0[v]
        if x0.shape:
            inds = tuple(a.ravel() for a in np.indices(x0.shape))
            for ind in zip(*inds):
                x0[ind] += _eps
                p0[v] = x0
                f_p = factor(**p0)
                x0[ind] -= 2 * _eps
                f_m = factor(**p0)

                log_f = f0.log_value
                hess_f[v][ind] = (
                                         f_p.log_value + f_m.log_value - 2 * log_f0) / _eps ** 2

                if _calc_deterministic:
                    for det, val in det_vars0.items():
                        hess_det[v, det][ind] = (
                                                        f_p.deterministic_values[det]
                                                        + f_m.deterministic_values[det]
                                                        - 2 * val) / _eps ** 2
        else:
            x0 += _eps
            p0[v] = x0
            f_p = factor(**p0)
            x0 -= 2 * _eps
            f_m = factor(**p0)

            hess_f[v] = (
                                f_p.log_value + f_m.log_value - 2 * log_f0) / _eps
            if _calc_deterministic:
                for det, val in det_vars0.items():
                    hess_det[v, det] = (
                                               f_p.deterministic_values[det]
                                               + f_m.deterministic_values[det]
                                               - 2 * val) / _eps ** 2

    return hess_f, hess_det


class FactorValue(NamedTuple):
    log_value: np.ndarray
    deterministic_values: Dict[str, np.ndarray]


class FactorNode:
    _deterministic_variables: Dict[str, Variable] = {}
    is_composite: bool = property(lambda self: False)
    is_deterministic: bool = property(lambda self: False)

    def __init__(self, factor: Factor,
                 *args: Tuple[Variable, ...],
                 **kwargs: Dict[str, Variable]):
        self._input_args = args
        self._input_kwargs = kwargs
        self._factor = factor
        self._variables = {v.name: v for v in args}
        self._variables.update((v.name, v) for v in kwargs.values())

        self._args = tuple(v.name for v in args)
        self._kwargs = {n: v.name for n, v in kwargs.items()}

        self._initialise()
        self._hash = hash(
            (self._factor, self._args,
             frozenset(self._kwargs.items()),
             frozenset(self.variables.items()),
             frozenset(self.deterministic_variables.items())))

    jacobian = numerical_jacobian
    hessdiag = numerical_hessdiag

    def _initialise(self):
        self.n_variables = len(self._variables)
        self.n_deterministic = len(self._deterministic_variables)
        self._all_variables = ChainMap(
            self._variables,
            self._deterministic_variables)
        self._args_dims = tuple(
            len(self.all_variables[v].plates) for v in self._args)
        self._kwargs_dims = {
            k: len(self.all_variables[v].plates) for k, v in self._kwargs.items()}
        self._plates = tuple(set(
            plate for v in self.all_variables.values() for plate in v.plates))
        self._variable_plates = {
            n: self._match_plates(v.plates)
            for n, v in self.all_variables.items()}

    def __hash__(self) -> int:
        return self._hash

    def _resolve_args(self, *args: Tuple[np.ndarray, ...],
                      **kwargs: Dict[str, np.ndarray]
                      ) -> Tuple[Tuple[np.ndarray, ...], Dict[str, np.ndarray]]:
        """Transforms in the input arguments to match the arguments
        specified for the factor"""
        n_args = len(args)
        args = args + tuple(kwargs[v] for v in self._args[n_args:])
        kws = {n: kwargs[v] for n, v in self._kwargs}

        variables = {v: x for v, x in zip(self._args, args)}
        variables.update(
            (self._kwargs[k], x) for k, x in kws.items())
        return args, kws, self._function_shape(variables)

    def _function_shape(self, variables: Dict[str, np.ndarray]
                        ) -> Tuple[int, ...]:
        """Calculates the expected function shape based on the variables
        """
        var_shapes = {v: np.shape(x) for v, x in variables.items()}
        var_dims_diffs = {
            v: len(s) - self.all_variables[v].ndim
            for v, s in var_shapes.items()}
        var_dims_diffs = {
            v: len(s) - self.all_variables[v].ndim  #
            for v, s in var_shapes.items()}
        """
        If all the passed variables have an extra dimension then 
        we assume we're evaluating multiple instances of the function at the 
        same time

        otherwise an error is raised
        """
        if set(var_dims_diffs.values()) == {1}:
            # Check if we're passing multiple values e.g. for sampling
            shift = 1
        elif set(var_dims_diffs.values()) == {0}:
            shift = 0
        else:
            raise ValueError("dimensions of passed inputs do not match")

        """
        Updating shape of output array to match input arrays

        singleton dimensions are always assumed to match as in
        standard array broadcasting

        e.g. (1, 2, 3) == (3, 2, 1)
        """
        shape = np.ones(self.ndim + shift, dtype=int)
        for v, vs in var_shapes.items():
            ind = self._variable_plates[v] + shift
            vshape = vs[shift:]
            if shift:
                ind = np.r_[0, ind]
                vshape = (vs[0],) + vshape

            if shape.size:
                assert (
                        np.equal(shape[ind], 1) |
                        np.equal(shape[ind], vshape) |
                        np.equal(vshape, 1)).all()
                shape[ind] = np.maximum(shape[ind], vshape)

        return tuple(shape)

    def _variables_difference(self, *args: Tuple[np.ndarray, ...],
                              **kwargs: Dict[str, np.ndarray]
                              ) -> Set[str]:
        args = self._args[:len(args)]
        return (self._variables.keys() - args).difference(kwargs)

    def _call_factor(self, *args: Tuple[np.ndarray, ...],
                     **kwargs: Dict[str, np.ndarray]
                     ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        args, kws, shape = self._resolve_args(*args, **kwargs)
        if self._factor.vectorised:
            return self._factor.call_factor(*args, **kws), shape
        else:
            return self._py_vec_call(*args, **kws), shape

    def _py_vec_call(self, *args: Tuple[np.ndarray, ...],
                     **kwargs: Dict[str, np.ndarray]) -> np.ndarray:
        """Some factors may not be vectorised to broadcast over
        multiple inputs

        this method checks whether multiple input values have been
        passed, and if so automatically loops over the inputs.
        If any of the inputs have initial dimension one, it repeats
        that value to match the length of the other inputs

        If the other inputs do not match then it raises ValueError
        """
        arg_dims = tuple(map(np.ndim, args))
        kwargs_dims = {k: np.ndim(a) for k, a in kwargs.items()}
        # Check dimensions of inputs directly match plates
        direct_call = (
                self._args_dims == arg_dims and
                all(dim == kwargs_dims[k] for k, dim in self._kwargs_dims.items()))
        if direct_call:
            return self._factor.call_factor(*args, **kwargs)
        else:
            # Check dimensions of inputs match plates + 1
            vectorised = (
                    (tuple(d + 1 for d in self._args_dims) == arg_dims) and
                    all(dim + 1 == kwargs_dims[k]
                        for k, dim in self._kwargs_dims.items()))

            if not vectorised:
                raise ValueError(
                    "input dimensions do not match required dims"
                    f"input: *args={arg_dims}, **kwargs={kwargs_dims}"
                    f"required: *args={self._args_dims}, "
                    f"**kwargs={self._kwargs_dims}")

            lens = [len(a) for a in args]
            kw_lens = {k: len(a) for k, a in kwargs.items()}

            # checking 1st dimensions match
            sizes = set(chain(lens, kw_lens.values()))
            dim0 = max(sizes)
            if sizes.difference({1, dim0}):
                raise ValueError(
                    f"size mismatch first dimensions passed: {sizes}")

            # teeing up iterators to generate arguments to factor calls
            zip_args = zip(*(
                a if l == dim0 else repeat(a[0])
                for a, l in zip(args, lens)))
            iter_kws = {
                k: iter(a) if kw_lens[k] == dim0 else iter(repeat(a[0]))
                for k, a in kwargs.items()}

            # iterator to generate keyword arguments
            def gen_kwargs():
                for i in range(dim0):
                    yield {
                        k: next(a) for k, a in iter_kws.items()}

            # TODO this loop can also be paralleised for increased performance
            res = np.array([
                self._factor.call_factor(*args, **kws)
                for args, kws in zip(zip_args, gen_kwargs())])

            return res

        raise ValueError(
            "input dimensions do not match required dims"
            f"input: *args={arg_dims}, **kwargs={kwargs_dims}"
            f"required: *args={self._args_dims}, "
            f"**kwargs={self._kwargs_dims}")

    def __call__(self, *args: Tuple[np.ndarray, ...],
                 **kwargs: Dict[str, np.ndarray]) -> FactorValue:
        val, shape = self._call_factor(*args, **kwargs)
        return FactorValue(val.reshape(shape), {})

    def _match_plates(self, plates: Collection[Plate]) -> np.ndarray:
        return np.array([self._plates.index(p) for p in plates], dtype=int)

    def _broadcast(self, plate_inds: Collection[int], value: np.ndarray) -> np.ndarray:
        shape = np.shape(value)
        plate_inds = np.asanyarray(plate_inds)
        shift = len(shape) - plate_inds.size

        assert shift in {0, 1}
        newshape = np.ones(self.ndim + shift, dtype=int)
        newshape[:shift] = shape[:shift]
        newshape[shift + plate_inds] = shape[shift:]

        return np.reshape(value, newshape)

    def broadcast_plates(self, plates: Collection[Plate], value: np.ndarray) -> np.ndarray:
        return self._broadcast(self._match_plates(plates), value)

    def broadcast_variable(self, variable: str, value: np.ndarray) -> np.ndarray:
        """
        broad casts the value of a variable to match the specific shape
        of the factor

        if the number of dimensions passed of the variable is 1
        greater than the dimensions of the variable then it's assumed
        that that dimension corresponds to multiple samples of that variable
        """
        return self._broadcast(self._variable_plates[variable], value)

    def collapse(self, variable: str, value: np.ndarray, agg_func=np.sum) -> np.ndarray:
        """
        broad casts the value of a variable to match the specific shape
        of the factor

        if the number of dimensions passed of the variable is 1
        greater than the dimensions of the variable then it's assumed
        that that dimension corresponds to multiple samples of that variable
        """
        ndim = np.ndim(value)
        shift = ndim - self.ndim
        assert shift in {0, 1}
        inds = self._variable_plates[variable] + shift
        dropaxes = tuple(np.setdiff1d(
            np.arange(shift, ndim), inds))

        # to ensured axes of returned array is in the correct order
        moved = np.moveaxis(value, inds, np.sort(inds))
        return agg_func(moved, axis=dropaxes)

    def __eq__(self, other: Union["FactorNode", Variable]
               ) -> Union[bool, "DeterministicFactorNode"]:
        if isinstance(other, FactorNode):
            if isinstance(other, type(self)):
                return (
                        (self._factor == other._factor)
                        and (self._args == other._args)
                        and (frozenset(self._kwargs.items())
                             == frozenset(other._kwargs.items()))
                        and (frozenset(self.variables.items())
                             == frozenset(other.variables.items()))
                        and (frozenset(self.deterministic_variables.items())
                             == frozenset(self.deterministic_variables.items())))
            else:
                return False

        elif isinstance(other, Variable):
            other = [other]

        return DeterministicFactorNode(
            self._factor, other,
            *(self._variables[name] for name in self._args),
            **{n: self._variables[name] for n, name in self._kwargs.items()})

    def __mul__(self, other) -> "FactorGraph":
        return FactorGraph([self]) * other

    def __repr__(self) -> str:
        args = ", ".join(chain(
            self._args,
            map("{0[0]}={0[1]}".format, self._kwargs.items())))
        return f"Factor({self._factor.name})({args})"

    @property
    def variables(self) -> Dict[str, Variable]:
        return self._variables

    @property
    def deterministic_variables(self) -> Dict[str, Variable]:
        return self._deterministic_variables

    @property
    def all_variables(self) -> Dict[str, Variable]:
        return self._all_variables

    @property
    def plates(self):
        return self._plates

    @property
    def ndim(self):
        return len(self.plates)

    @property
    def name(self):
        return self._factor.name

    @property
    def call_signature(self):
        args = ", ".join(self._args)
        kws = ", ".join(map("{0[0]}={0[1]}".format, self._kwargs.items()))
        call_strs = []
        if args:
            call_strs.append(args)
        if kws:
            call_strs.extend(['*', kws])
        call_str = ", ".join(call_strs)
        call_sig = f"{self.name}({call_str})"
        return call_sig


class DeterministicFactorNode(FactorNode):
    is_deterministic: bool = property(lambda self: True)

    def __init__(self, factor: Factor,
                 deterministic_variables: Tuple[Variable, ...] = (),
                 *args: Tuple[Variable, ...],
                 **kwargs: Dict[str, Variable]):
        self._deterministic_variables = {v.name: v for v in deterministic_variables}
        super().__init__(factor, *args, **kwargs)

    def __call__(self, *args: Tuple[np.ndarray, ...],
                 **kwargs: Dict[str, np.ndarray]) -> FactorValue:
        res, shape = self._call_factor(*args, **kwargs)
        shift = len(shape) - self.ndim
        plate_dim = dict(zip(self.plates, shape[shift:]))

        det_shapes = {
            v: shape[:shift] + tuple(
                plate_dim[p] for v in self.deterministic_variables.values()
                for p in v.plates)
            for v in self.deterministic_variables}

        if not (isinstance(res, tuple) or self.n_deterministic > 1):
            res = res,

        log_val = 0. if shape == () else np.zeros(np.ones_like(shape))
        det_vals = {
            k: np.reshape(val, det_shapes[k]) if det_shapes[k] else val
            for k, val in zip(self._deterministic_variables, res)}
        return FactorValue(log_val, det_vals)

    def __repr__(self) -> str:
        factor_str = super().__repr__()
        var_str = ", ".join(self._deterministic_variables)
        return f"({factor_str} == ({var_str}))"


class FactorGraph(DeterministicFactorNode):
    is_composite: bool = property(lambda self: True)

    def __init__(self, factors: Collection[FactorNode], name=None):
        self._factors = tuple(factors)
        self._name = ".".join(f.name for f in factors) if name is None else name

        self._variables = ChainMap(*(
            f.variables for f in self._factors))
        self._deterministic_variables = ChainMap(*(
            f.deterministic_variables for f in self._factors))
        self._all_variables = ChainMap(*(
            f.all_variables for f in self._factors))

        self._factor_variables = {
            f: f.variables for f in self._factors}
        self._factor_det_variables = {
            f: f.deterministic_variables for f in self._factors}
        self._factor_all_variables = {
            f: f.all_variables for f in self._factors}

        self._validate()
        self._initialise()
        self._hash = hash(frozenset(self.factors))

    @property
    def name(self):
        return self._name

    def _validate(self) -> None:
        det_var_counts = ", ".join(
            v for v, c in Counter(
                v for f in self.factors
                for v in f.deterministic_variables).items()
            if c > 1)
        if det_var_counts:
            raise ValueError(
                "Improper FactorGraph, "
                f"Deterministic variables {det_var_counts} appear in "
                "multiple factors")

        self._call_sequence, variables = self._get_call_sequence()
        self._all_factors = tuple(sum(self._call_sequence, []))

        diff = variables.keys() ^ self._variables.keys()
        if diff:
            raise ValueError(
                "Improper FactorGraph? unused variables: "
                + ", ".join(diff))

    def _get_call_sequence(self) -> None:
        """Calculates an appropriate call sequence for the factor graph

        each set of calls can be evaluated independently in parallel
        """
        variables = {
            v: self._variables[v] for v in
            (self._variables.keys() - self._deterministic_variables.keys())}

        factor_args = [factor._args for factor in self.factors]
        max_len = min(map(len, factor_args))
        self._args = tuple(
            factor_args[0][i] for i in range(max_len)
            if len(set(arg[i] for arg in factor_args)) == 1)
        self._kwargs = {k: k for k in variables.keys() - self._args}

        call_sets = defaultdict(list)
        for factor in self.factors:
            missing_vars = frozenset(factor._variables_difference(**variables))
            call_sets[missing_vars].append(factor)

        call_sequence = []
        while call_sets:
            # the factors that can be evaluated have no missing variables
            factors = call_sets.pop(frozenset(()))
            # if there's a KeyError then the FactorGraph is improper
            calls = []
            new_variables = {}
            for factor in factors:
                if factor.is_deterministic:
                    det_vars = factor._deterministic_variables
                else:
                    det_vars = {}

                calls.append(factor)
                new_variables.update(det_vars)

            call_sequence.append(calls)

            # update to include newly calculated factors
            for missing in call_sets:
                if missing.intersection(new_variables):
                    factors = call_sets.pop(missing)
                    call_sets[missing.difference(new_variables)].extend(factors)

            variables.update(new_variables)
        return call_sequence, variables

    def __call__(self, *args: Tuple[np.ndarray, ...],
                 **kwargs: Dict[str, np.ndarray]) -> FactorValue:
        # generate set of factors to call, these are indexed by the
        # missing deterministic variables that need to be calculated
        log_value = 0.
        det_values = {}
        variables = kwargs

        n_args = len(args)
        if n_args > len(self._args):
            kws_str = ", ".join(self._kwargs)
            raise TypeError(
                f"too many arguments passed, must pass {len(self._args)} arguments, "
                f"factor graph call signature: {self.call_signature}")

        missing = self._kwargs.keys() - variables.keys() - set(self._args[:n_args])
        if missing:
            n_miss = len(missing)
            missing_str = ", ".join(missing)
            raise TypeError(f"{self} missing {n_miss} arguments: {missing_str}"
                            f"factor graph call signature: {self.call_signature}")

        for calls in self._call_sequence:
            # TODO parallelise this part?
            for factor in calls:
                ret = factor(*args, **variables)
                ret_value = self.broadcast_plates(factor.plates, ret.log_value)
                log_value = add_arrays(log_value, ret_value)
                det_values.update(ret.deterministic_values)
                variables.update(ret.deterministic_values)

        return FactorValue(log_value, det_values)

    def __mul__(self, other: FactorNode) -> "FactorGraph":
        factors = self.factors

        if isinstance(other, FactorGraph):
            factors += other.factors
        elif isinstance(other, FactorNode):
            factors += (other,)
        else:
            raise TypeError(
                f"type of passed element {(type(other))} "
                "does not match required types, (`FactorGraph`, `FactorNode`)")

        return type(self)(factors)

    def __repr__(self) -> str:
        factors_str = " * ".join(map(repr, self.factors))
        return f"({factors_str})"

    @property
    def factors(self) -> Tuple[FactorNode, ...]:
        return self._factors

    @property
    def factor_variables(self) -> Dict[FactorNode, str]:
        return self._factor_all_variables

    @property
    def factor_deterministic_variables(self) -> Dict[FactorNode, str]:
        return self._factor_det_variables

    @property
    def factor_all_variables(self) -> Dict[FactorNode, str]:
        return self._factor_all_variables
