from typing import Optional, Dict, Tuple, Any, Callable

import numpy as np

from autofit.graphical.factor_graphs.abstract import FactorValue
from autofit.mapper.variable_operator import VariableData

from autofit.graphical.laplace.line_search import line_search, OptimisationState


## get ascent direction


def gradient_ascent(state: OptimisationState) -> VariableData:
    return state.gradient


def newton_direction(state: OptimisationState) -> VariableData:
    return state.hessian.ldiv(state.gradient)


## Quasi-newton approximations


def sr1_update(
    state1: OptimisationState, state: OptimisationState, mintol=1e-8, **kwargs
) -> OptimisationState:
    yk = VariableData.sub(state1.gradient, state.gradient)
    dk = VariableData.sub(state1.parameters, state.parameters)
    Bk = state.hessian
    zk = yk + Bk * dk
    zkdk = -zk.dot(dk)

    tol = mintol * dk.norm() * zk.norm()
    if zkdk > tol:
        vk = zk / np.sqrt(zkdk)
        Bk1 = Bk.lowrankupdate(vk)
    elif zkdk < -tol:
        vk = zk / np.sqrt(-zkdk)
        Bk1 = Bk.lowrankdowndate(vk)
    else:
        Bk1 = Bk

    state1.hessian = Bk1
    return state1


def diag_sr1_update(
    state1: OptimisationState, state: OptimisationState, **kwargs
) -> OptimisationState:
    yk = VariableData.sub(state1.gradient, state.gradient)
    dk = VariableData.sub(state1.parameters, state.parameters)
    Bk = state.hessian
    zk = yk + Bk * dk
    dzk = dk * zk
    alpha = -zk.dot(dk) / dzk.dot(dzk)
    state1.hessian = Bk.diagonalupdate(alpha * (zk ** 2))
    return state1


def bfgs1_update(
    state1: OptimisationState,
    state: OptimisationState,
    **kwargs,
) -> OptimisationState:
    """
    y_k = g_{k+1} - g{k}
    d_k = x_{k+1} - x{k}
    B_{k+1} = B_{k}
    + \frac
        {y_{k}y_{k}^T}
        {y_{k}^T d_{k}}}
    - \frac
        {B_{k} d_{k} (B_{k} d_{k})^T}
        {d_{k}^T B_{k} d_{k}}}}
    """
    yk = VariableData.sub(state.gradient, state1.gradient)
    dk = VariableData.sub(state1.parameters, state.parameters)
    Bk = state.hessian

    ykTdk = yk.dot(dk)
    Bdk = Bk.dot(dk)
    dkTBdk = -VariableData.dot(Bdk, dk)

    state1.hessian = Bk.update(
        (yk, VariableData(yk).div(ykTdk)), (Bdk, VariableData(Bdk).div(dkTBdk))
    )
    return state1


def bfgs_update(
    state1: OptimisationState,
    state: OptimisationState,
    **kwargs,
) -> OptimisationState:
    yk = VariableData.sub(state1.gradient, state.gradient)
    dk = VariableData.sub(state1.parameters, state.parameters)
    Bk = state.hessian

    ykTdk = -yk.dot(dk)
    Bdk = Bk.dot(dk)
    dkTBdk = VariableData.dot(Bdk, dk)

    state1.hessian = Bk.update(
        (yk, VariableData(yk).div(ykTdk)), (Bdk, VariableData(Bdk).div(dkTBdk))
    )
    return state1


def quasi_deterministic_update(
    state1: OptimisationState,
    state: OptimisationState,
    **kwargs,
) -> OptimisationState:
    dk = VariableData.sub(state1.parameters, state.parameters)
    zk = VariableData.sub(
        state1.value.deterministic_values, state.value.deterministic_values
    )
    Bxk, Bzk = state1.hessian, state.det_hessian
    zkTzk2 = zk.dot(zk) ** 2
    alpha = (dk.dot(Bxk.dot(dk)) - zk.dot(Bzk.dot(zk))) / zkTzk2
    if alpha >= 0:
        Bzk1 = Bzk.lowrankupdate(np.sqrt(alpha) * (zk))
    else:
        Bzk1 = Bzk.lowrankdowndate(np.sqrt(-alpha) * (zk))

    state1.det_hessian = Bzk1
    return state1


def diag_quasi_deterministic_update(
    state1: OptimisationState,
    state: OptimisationState,
    **kwargs,
) -> OptimisationState:
    dk = VariableData.sub(state1.parameters, state.parameters)
    zk = VariableData.sub(
        state1.value.deterministic_values, state.value.deterministic_values
    )
    Bxk, Bzk = state1.hessian, state.det_hessian
    zk2 = zk ** 2
    zk4 = (zk2 ** 2).sum()
    alpha = (dk.dot(Bxk.dot(dk)) - zk.dot(Bzk.dot(zk))) / zk4
    state1.det_hessian = Bzk.diagonalupdate(alpha * zk2)

    return state1


## Newton step


def take_step(
    state: OptimisationState,
    old_state: Optional[OptimisationState] = None,
    *,
    search_direction=newton_direction,
    calc_line_search=line_search,
    search_direction_kws: Optional[Dict[str, Any]] = None,
    line_search_kws: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], OptimisationState]:
    state.search_direction = search_direction(state, **(search_direction_kws or {}))
    return calc_line_search(state, old_state, **(line_search_kws or {}))


def take_quasi_newton_step(
    state: OptimisationState,
    old_state: Optional[OptimisationState] = None,
    *,
    search_direction=newton_direction,
    calc_line_search=line_search,
    quasi_newton_update=bfgs_update,
    search_direction_kws: Optional[Dict[str, Any]] = None,
    line_search_kws: Optional[Dict[str, Any]] = None,
    quasi_newton_kws: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], OptimisationState]:
    """ """
    state.search_direction = search_direction(state, **(search_direction_kws or {}))
    if state.search_direction.vecnorm(np.Inf) == 0:
        # if gradient is zero then at maximum already
        return 0.0, state

    stepsize, state1 = calc_line_search(state, old_state, **(line_search_kws or {}))
    if stepsize:
        # Only update estimate if a step has been taken
        state1 = quasi_newton_update(state1, state, **(quasi_newton_kws or {}))
        if state.det_hessian:
            state1 = quasi_deterministic_update(
                state1, state, **(quasi_newton_kws or {})
            )

    return stepsize, state1


def xtol_condition(state, old_state, xtol=1e-6, ord=None, **kwargs):
    dx = VariableData.sub(state.parameters, old_state.parameters).vecnorm(ord=ord)
    if dx < xtol:
        return True, f"Minimum parameter change tolerance achieved, {dx} < {xtol}"


def grad_condition(state, old_state, gtol=1e-5, ord=None, **kwargs):
    dg = VariableData.vecnorm(state.gradient, ord=ord)
    if dg < gtol:
        return True, f"Gradient tolerance achieved, {dg} < {gtol}"


def ftol_condition(state, old_state, ftol=1e-6, monotone=True, **kwargs):
    df = state.value - old_state.value
    if 0 < df < ftol:
        return True, f"Minimum function change tolerance achieved, {df} < {ftol}"
    elif monotone and df < 0:
        return False, f"factor failed to increase on next step, {df}"


def nfev_condition(state, old_state, maxfev=10000, **kwargs):
    if state.f_count > maxfev:
        return (
            True,
            f"Maximum number of function evaluations (maxfev={maxfev}) has been exceeded.",
        )


def ngev_condition(state, old_state, maxgev=10000, **kwargs):
    if state.g_count > maxgev:
        return (
            True,
            f"Maximum number of gradient evaluations (maxgev={maxgev}) has been exceeded.",
        )


stop_conditions = (
    xtol_condition,
    ftol_condition,
    grad_condition,
)

_OPT_CALLBACK = Callable[[OptimisationState, OptimisationState], None]


def optimise_quasi_newton(
    state: OptimisationState,
    old_state: Optional[OptimisationState] = None,
    *,
    max_iter=100,
    search_direction=newton_direction,
    calc_line_search=line_search,
    quasi_newton_update=bfgs_update,
    stop_conditions=stop_conditions,
    search_direction_kws: Optional[Dict[str, Any]] = None,
    line_search_kws: Optional[Dict[str, Any]] = None,
    quasi_newton_kws: Optional[Dict[str, Any]] = None,
    stop_kws: Optional[Dict[str, Any]] = None,
    callback: Optional[_OPT_CALLBACK] = None,
) -> Tuple[bool, OptimisationState, str]:

    success = False
    message = "max iterations reached"
    for i in range(max_iter):
        stepsize, state1 = take_quasi_newton_step(
            state,
            old_state,
            search_direction=search_direction,
            calc_line_search=calc_line_search,
            quasi_newton_update=quasi_newton_update,
            search_direction_kws=search_direction_kws,
            line_search_kws=line_search_kws,
            quasi_newton_kws=quasi_newton_kws,
        )
        state, old_state = state1, state

        if callback:
            callback(state, old_state)

        if stepsize is None:
            message = "abnormal termination of line search"
            break
        elif not np.isfinite(state.value):
            message = "function is no longer finite"
            break
        else:
            for stop_condition in stop_conditions:
                stop = stop_condition(state, old_state, **(stop_kws or {}))
                if stop:
                    success, message = stop
                    break
            else:
                continue
            break

    message += f", iter={i+1}"
    return success, state, message
