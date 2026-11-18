TITLE Muscle spindle Ia rate sampled independently of output spikes

NEURON {
    ARTIFICIAL_CELL IaGenerator
    POINTER xm
    RANGE start, dur, noise, update_ms
    RANGE rate0, k_len, k_vel, rmin, rmax, xm_rest
    RANGE rate, vel, y
}

PARAMETER {
    start = 5 (ms)       : preserve the initial spike at start
    dur = -1 (ms)        : <= 0 means no end time
    noise = 0 <0,1>
    update_ms = 1 (ms)
    rate0 = 50           : Hz
    k_len = 25           : Hz/mm
    k_vel = 8            : Hz per mm/s
    rmin = 2             : Hz
    rmax = 200           : Hz
    xm_rest = 3 (mm)
}

ASSIGNED {
    xm
    y
    rate
    vel
    end (ms)
    on
    xm_prev
    t_prev (ms)
    phase
    phase_limit
}

INITIAL {
    on = 0
    y = 0
    phase = 0
    rate = rate0
    vel = 0
    xm_prev = xm
    t_prev = 0
    if (noise < 0) { noise = 0 }
    if (noise > 1) { noise = 1 }
    if (update_ms <= 0) { update_ms = 1 }
    : At most one deterministic spike per update, even at rmax.
    if (rmax > 0 && update_ms > 1000/rmax) { update_ms = 1000/rmax }
    if (start >= 0) { net_send(start, 3) }
}

PROCEDURE update_rate() {
    rate = rate0 + k_len * (xm - xm_rest) + k_vel * vel
    if (rate < rmin) { rate = rmin }
    if (rate > rmax) { rate = rmax }
}

FUNCTION cycle_size() {
    if (noise == 0) {
        cycle_size = 1
    } else {
        : At constant rate this has the same mean/noise mixture as NetStim.
        cycle_size = (1-noise) + noise*exprand(1)
        if (cycle_size < 1e-9) { cycle_size = 1e-9 }
    }
}

FUNCTION next_tick() (ms) {
    next_tick = update_ms
    if (t + next_tick > end) { next_tick = end - t }
}

NET_RECEIVE (w) { LOCAL dt_ms, old_rate
    if (flag == 3) {
        on = 1
        end = 1e9
        if (dur > 0) { end = t + dur }
        xm_prev = xm
        t_prev = t
        phase = 0
        phase_limit = cycle_size()
        vel = 0
        update_rate()
        y = 2
        net_event(t)
        net_send(.1, 2)
    }
    if (flag == 1 && on == 1) {
        dt_ms = t - t_prev
        if (dt_ms > 0) {
            vel = 1000 * (xm - xm_prev) / dt_ms
            old_rate = rate
            update_rate()
            : Trapezoidal integration keeps the clock responsive at low rates.
            phase = phase + (old_rate + rate)*dt_ms/2000
            xm_prev = xm
            t_prev = t
            while (phase >= phase_limit) {
                phase = phase - phase_limit
                phase_limit = cycle_size()
                y = 2
                net_event(t)
                net_send(.1, 2)
            }
        }
    }
    if (flag == 2) { y = 0 }
    if ((flag == 1 || flag == 3) && on == 1) {
        if (t < end) {
            net_send(next_tick(), 1)
        } else {
            on = 0
        }
    }
}
