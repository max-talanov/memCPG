TITLE Calcium dynamics and cross-bridge formation

UNITS { }

NEURON {
	SUFFIX CaSP

	::module 1::
	RANGE k1, k2, k3, k4, k5, k6, k, k5i, k6i
	RANGE Umax, Rmax, t1, t2, R, vth, U
	RANGE phi1, phi2, phi3, phi4, phi

	::module 2::
	RANGE c1, c2, c3, c4, c5
	RANGE AMinf, AMtau, SF_AM
	RANGE acm, alpha, alpha1, alpha2, alpha3, beta, gamma

	::simulation::
	RANGE spk_index, t_axon, R
	POINTER fiber_length
	RANGE activation, fiber_length0, length_reference
}

PARAMETER {
	::module 1::
	k1 = 3000		: M-1*ms-1
	k2 = 3			: ms-1
	k3 = 400		: M-1*ms-1
	k4 = 1			: ms-1
	k5i = 4e5		: M-1*ms-1
	k6i = 150		: ms-1
	k = 850			: M-1
	SF_AM = 5
	Rmax = 10		: ms-1
	Umax = 2000		: M-1*ms-1
	t1 = 3			: ms
	t2 = 25			: ms
	phi1 = 0.03
	phi2 = 1.23
	phi3 = 0.01
	phi4 = 1.08
	CS0 = 0.03     	:[M]
	B0 = 0.00043	:[M]
	T0 = 0.00007 	:[M]

	::module 2::
	c1 = 0.128
	c2 = 0.093
	c3 = 61.206
	c4 = -13.116
	c5 = 5.095
	alpha = 2
	alpha1 = 4.77
	alpha2 = 400
	alpha3 = 160
	beta = 0.47
	gamma = 0.001

	::simulation::
	vth = -40
	spk_index = 0
	t_axon = 0.01
	fiber_length0 = 3 : mm, reference physical fibre length
	length_reference = -8 : preserve the original phi() coordinate convention
}

STATE {
	CaSR
	CaSRCS
	Ca
	CaB
	CaT
	AM
}

ASSIGNED {
	v 	(mV)
	R
	R_On
	Spike_On
	k5
	k6
	AMinf
	AMtau
	fiber_length
	last_spk	(ms)
	acm
	activation
}

BREAKPOINT {

	SPK_DETECT (v, t)
	CaR (CaSR, t)

	SOLVE state METHOD cnexp

	::isometric and isokinetic condition::
	set_activation()
}

DERIVATIVE state { LOCAL Uval
	rate (length_reference + fiber_length - fiber_length0, CaT, AM, t)

	Uval = U(Ca)
	CaSR' = -k1*CS0*CaSR + (k1*CaSR+k2)*CaSRCS - R + Uval
	CaSRCS' = k1*CS0*CaSR - (k1*CaSR+k2)*CaSRCS

	Ca' = - k5*T0*Ca + (k5*Ca+k6)*CaT - k3*B0*Ca + (k3*Ca+k4)*CaB + R - Uval
	CaB' = k3*B0*Ca - (k3*Ca+k4)*CaB
	CaT' = k5*T0*Ca - (k5*Ca+k6)*CaT

	AM' = (AMinf -AM)/AMtau
}

PROCEDURE SPK_DETECT (v (mv), t (ms)) {
	if (Spike_On == 0 && v > vth) {
	Spike_On = 1
	last_spk = t + t_axon
	spk_index = spk_index + 1
	R_On = 1
	} else if (v < vth) {
	Spike_On = 0
	}
}

FUNCTION U (x) { LOCAL xk, xk2
	if (x >= 0) {
		xk = x*k
		xk2 = xk*xk
		U = Umax*(xk2/(1+xk+xk2))^2
	}
	else {U = 0}
}

FUNCTION phi (x) {
	if (x <= -8) {phi = phi1*x + phi2}
	else {phi = phi3*x + phi4}
}

PROCEDURE CaR (CaSR (M), t (ms)) { LOCAL tempR, t_minus_last  ::Ca_Release::
	if (R_On == 1) {
    if (last_spk > -1e9){
      t_minus_last = t - last_spk
      tempR = CaSR*Rmax*(1-exp(-t_minus_last/t1))*exp(-t_minus_last/t2)
    } else { tempR = 0 }
	if (t < last_spk) { tempR = 0 }
	R = tempR
	}
	else {R = 0}
}

PROCEDURE rate (length_coordinate, CaT (M), AM (M), t(ms)) { LOCAL x
	k5 = phi(length_coordinate)*k5i
	k6 = k6i/(1 + SF_AM*AM)
	x = CaT/T0
	AMinf = 0.5*(1+tanh((x-c1)/c2))
	AMtau = c3/(cosh((x-c4)/(2*c5)))
}

PROCEDURE set_activation() {
	activation = AM^alpha
}

INITIAL {
	CaSR = 0.0025  		:[M]
	CaSRCS = 0			:[M]
	Ca = 1e-10			:[M]
	CaB = 0				:[M]
	CaT = 0				:[M]
	AM = 0				:[M]
	set_activation()
	last_spk = -1e9
	spk_index = 0
	R_On = 0
	Spike_On = 0
}
