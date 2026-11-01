TITLE Damped Hill-type muscle fibre and series elastic tendon

COMMENT
The representative electrical fibre has a fixed cable geometry. Its mechanical
fibre length evolves independently. Positive velocity denotes lengthening.
length is the musculotendon path length supplied by the joint (mm).
activation is dimensionless, supplied explicitly by CaSP.
The massless fibre/tendon force balance includes a small parallel damper.
xm and xce remain as legacy offset coordinates for existing analysis scripts.
This fixed-step model is intended for the h.cvode_active(0) simulation driver.
ENDCOMMENT

NEURON {
    SUFFIX fHill
    POINTER activation
    RANGE length, fiber_length0, tendon_slack_length, fiber_length, velocity
    RANGE p0, tendon_strain_at_fmax, active_width, passive_width
    RANGE max_shortening_rate, fv_shape, fv_max, damping
    RANGE min_fiber_ratio, max_fiber_ratio
    RANGE A, F, Fc, Fpassive, xm, xce, xm_init, xce_init
}

PARAMETER {
    length = 5                 : mm, externally supplied musculotendon length
    fiber_length0 = 3          : mm, optimal fibre length
    tendon_slack_length = 2    : mm
    p0 = 23                    : N, force scale of a representative muscle
    tendon_strain_at_fmax = 0.05
    active_width = 0.45
    passive_width = 0.6
    max_shortening_rate = 10   : optimal fibre lengths / second
    fv_shape = 0.25
    fv_max = 1.5
    damping = 0.05             : normalized force / normalized velocity
    min_fiber_ratio = 0.4
    max_fiber_ratio = 1.6
    xm_init = -8               : legacy coordinate origins, not physical lengths
    xce_init = -8
}

STATE { fiber_length }

ASSIGNED {
    activation
    A
    F
    Fc
    Fpassive
    velocity
    xm
    xce
}

INITIAL {
    fiber_length = fiber_length0
    evaluate()
}

BREAKPOINT {
    SOLVE contraction METHOD euler
    if (fiber_length < min_fiber_ratio*fiber_length0) {
        fiber_length = min_fiber_ratio*fiber_length0
    }
    if (fiber_length > max_fiber_ratio*fiber_length0) {
        fiber_length = max_fiber_ratio*fiber_length0
    }
    evaluate()
}

DERIVATIVE contraction {
    evaluate()
    fiber_length' = velocity
}

PROCEDURE evaluate() { LOCAL strain, ratio, active, passive, u
    A = activation
    if (A < 0) { A = 0 }
    if (A > 1) { A = 1 }
    ratio = fiber_length/fiber_length0
    active = A*exp(-((ratio-1)/active_width)^2)
    passive = 0
    if (ratio > 1) { passive = ((ratio-1)/passive_width)^2 }
    strain = (length-fiber_length-tendon_slack_length)/tendon_slack_length
    F = 0
    if (strain > 0) { F = p0*strain/tendon_strain_at_fmax }
    Fc = p0*active
    Fpassive = p0*passive
    u = balance_velocity(F/p0, active, passive)
    velocity = u*max_shortening_rate*fiber_length0/1000 : mm/ms
    if (fiber_length <= min_fiber_ratio*fiber_length0 && velocity < 0) { velocity = 0 }
    if (fiber_length >= max_fiber_ratio*fiber_length0 && velocity > 0) { velocity = 0 }
    xm = xm_init + length - (fiber_length0+tendon_slack_length)
    xce = xce_init + fiber_length - fiber_length0
}

FUNCTION force_velocity(u) {
    if (u < 0) {
        if (u <= -1) { force_velocity = 0 }
        else { force_velocity = (1+u)/(1-u/fv_shape) }
    } else {
        force_velocity = 1+(fv_max-1)*u/(u+fv_shape)
    }
}

FUNCTION balance_velocity(target, active, passive) { LOCAL lo, hi, mid, residual, i
    : Solve target = active*f_v(u) + passive + damping*u.
    : Damping makes the equation monotone, including at zero activation.
    : The finite velocity range also bounds pathological transient stretching.
    if (fabs(target-active-passive) < 1e-12) {
        balance_velocity = 0
    } else {
        lo = -3
        hi = 3
        FROM i = 0 TO 23 {
            mid = (lo+hi)/2
            residual = active*force_velocity(mid)+passive+damping*mid-target
            if (residual > 0) { hi = mid }
            else { lo = mid }
        }
        balance_velocity = (lo+hi)/2
    }
}
