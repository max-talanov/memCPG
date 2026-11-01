NEURON {
	ARTIFICIAL_CELL IaGenerator
	POINTER xm
	RANGE start, dur, noise
	RANGE rate0, k_len, k_vel, rmin, rmax, xm_rest
	RANGE rate, vel, y
}

PARAMETER {
	start	= 5	(ms)	: first spike
	dur		= -1 (ms)	: burst duration; <= 0 = until the end of the simulation
	noise	= 0 <0,1>	: 0 = regular, 1 = Poisson
	rate0	= 50		: (Hz) tonic rate at resting length and zero velocity
	k_len	= 5			: (Hz/mm) static sensitivity to stretch
	k_vel	= 2			: (Hz per mm/s) dynamic sensitivity to stretch velocity
	rmin	= 2			: (Hz) lower bound of the firing rate
	rmax	= 200		: (Hz) upper bound of the firing rate
	xm_rest	= 3 (mm)	: positive resting fibre length; set from fHill in leg.py
}

ASSIGNED {
	xm
	y
	rate
	vel
	event (ms)
	end (ms)
	on
	xm_prev
	t_prev (ms)
}

INITIAL {
	on = 0
	y = 0
	rate = rate0
	vel = 0
	xm_prev = xm
	t_prev = start
	if (noise < 0) { noise = 0 }
	if (noise > 1) { noise = 1 }
	if (start >= 0) {
		on = 1
		event = start
		net_send(event, 3)
	}
}

PROCEDURE init_sequence (t (ms)) {
	on = 1
	event = t
	if (dur > 0) {
		end = t + 1e-6 + dur
	} else {
		end = 1e9
	}
}

FUNCTION invl (t (ms)) (ms) { LOCAL mean, dt_ms
	dt_ms = t - t_prev
	if (dt_ms <= 0) {
		vel = 0
	} else {
		vel = 1000 * (xm - xm_prev) / dt_ms	: mm per second
	}
	rate = rate0 + k_len * (xm - xm_rest) + k_vel * vel
	if (rate < rmin) { rate = rmin }
	if (rate > rmax) { rate = rmax }
	xm_prev = xm
	t_prev = t
	mean = 1000 / rate
	if (noise == 0) {
		invl = mean
	} else {
		invl = (1 - noise) * mean + noise * mean * exprand(1)
	}
}

NET_RECEIVE (w) {
	if (flag == 3) {		: from INITIAL
		if (on == 1) {
			init_sequence(t)
			net_send(0, 1)
		}
	}
	if (flag == 1 && on == 1) {
		y = 2
		net_event(t)
		event = event + invl(t)
		if (event > end) {
			on = 0
		}
		if (on == 1) {
			net_send(event - t, 1)
		}
		net_send(.1, 2)
	}
}
