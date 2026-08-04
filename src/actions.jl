"""
    set_param!(integ, i, v)

Set parameter `i` to `v` — piecewise-constant control (thrust level, inflow
rate, harvest quota). Marks the integrator state modified so the first-stage
derivative cache is refreshed with the new parameters.
"""
set_param!(integ, i::Integer, v) = (integ.p[i] = v; u_modified!(integ, true); nothing)

"""
    impulse!(integ, i, dv)

Add `dv` to state component `i` — impulsive control (drug dose, injection,
deposit). Marks the integrator state modified.
"""
impulse!(integ, i::Integer, dv) = (integ.u[i] += dv; u_modified!(integ, true); nothing)

"""
    set_state!(integ, i, v)

Overwrite state component `i` with `v` — resets and replacements (component
swap, tank drain). Marks the integrator state modified.
"""
set_state!(integ, i::Integer, v) = (integ.u[i] = v; u_modified!(integ, true); nothing)
