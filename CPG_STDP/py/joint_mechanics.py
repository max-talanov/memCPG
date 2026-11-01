"""One flexion/extension degree of freedom per leg.

The mechanical solver uses SI units and has no NEURON dependency. The adapter
exchanges forces and lengths with NEURON at a fixed, synchronous interval.
Each muscle pool is a set of representative units: its population MEAN force
drives the joint, matching the existing population-averaged force recordings.
"""

from array import array
from collections import namedtuple
from dataclasses import asdict, dataclass
import math

# Population mean of one muscle pool, in the units of cell.mechanical_state().
MuscleState = namedtuple("MuscleState", "force_n fiber_length_mm fiber_velocity_mm_s")


@dataclass(frozen=True)
class JointParameters:
    # Demonstration parameters, to be calibrated for the intended animal/muscle.
    inertia_kg_m2: float = 2e-5
    damping_nm_s: float = 0.002
    stiffness_nm_rad: float = 0.05
    moment_arm_m: float = 0.001
    rest_length_m: float = 0.005
    min_angle_rad: float = -0.6
    max_angle_rad: float = 0.6
    load_torque_nm: float = 0.0  # positive resists flexion

    def __post_init__(self):
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Joint parameters must be finite")
        for name in ("inertia_kg_m2", "moment_arm_m", "rest_length_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.damping_nm_s < 0 or self.stiffness_nm_rad < 0:
            raise ValueError("Joint damping and stiffness must be non-negative")
        if not self.min_angle_rad < 0 < self.max_angle_rad:
            raise ValueError("Joint angle limits must enclose the neutral angle 0")
        excursion = self.moment_arm_m * max(-self.min_angle_rad, self.max_angle_rad)
        if excursion >= self.rest_length_m:
            raise ValueError("Joint excursion would produce a non-positive muscle length")


class JointMechanics:
    """Positive angle = flexion; an active flexor produces positive torque."""

    def __init__(self, parameters=None):
        self.parameters = parameters or JointParameters()
        self.reset()

    def reset(self):
        self.angle_rad = 0.0
        self.velocity_rad_s = 0.0

    @property
    def lengths_m(self):
        p = self.parameters
        excursion = p.moment_arm_m * self.angle_rad
        return p.rest_length_m + excursion, p.rest_length_m - excursion  # E, F

    def step(self, dt_s, force_extensor_n, force_flexor_n):
        # Inputs are validated by run_coupled: dt > 0, finite non-negative forces.
        p = self.parameters
        torque = p.moment_arm_m * (force_flexor_n - force_extensor_n) - p.load_torque_nm

        def derivative(angle, velocity):
            return velocity, (torque - p.damping_nm_s * velocity
                              - p.stiffness_nm_rad * angle) / p.inertia_kg_m2

        # Bound the RK4 substep by both mechanical time scales, including for
        # parameter overrides. Electrical integration has its own, finer step.
        rate = max(math.sqrt(p.stiffness_nm_rad / p.inertia_kg_m2),
                   p.damping_nm_s / p.inertia_kg_m2, 1.0)
        count = max(1, math.ceil(dt_s * rate / 0.1))
        dt = dt_s / count
        for _ in range(count):
            q, v = self.angle_rad, self.velocity_rad_s
            k1q, k1v = derivative(q, v)
            k2q, k2v = derivative(q + dt * k1q / 2, v + dt * k1v / 2)
            k3q, k3v = derivative(q + dt * k2q / 2, v + dt * k2v / 2)
            k4q, k4v = derivative(q + dt * k3q, v + dt * k3v)
            self.angle_rad = q + dt * (k1q + 2*k2q + 2*k3q + k4q) / 6
            self.velocity_rad_s = v + dt * (k1v + 2*k2v + 2*k3v + k4v) / 6
            # Inelastic joint end stops; inward motion is always allowed.
            if self.angle_rad <= p.min_angle_rad:
                self.angle_rad = p.min_angle_rad
                self.velocity_rad_s = max(0.0, self.velocity_rad_s)
            elif self.angle_rad >= p.max_angle_rad:
                self.angle_rad = p.max_angle_rad
                self.velocity_rad_s = min(0.0, self.velocity_rad_s)


class LegJoint:
    """Bind one joint to separate extensor/flexor populations on every MPI rank."""

    def __init__(self, leg, pc, parameters=None, name=None):
        self.model = JointMechanics(parameters)
        self.name = name or ("left" if leg.leg_l else "right")
        self.record = int(pc.id()) == 0
        self.counts = (len(leg.muscle_E), len(leg.muscle_F))
        if not all(self.counts):
            raise ValueError("A joint needs both extensor and flexor muscle pools")
        # Cells are created with configure_mechanics(rest length) in utils_cpg.
        self.pools = tuple(tuple(pc.gid2cell(gid) for gid in gids if pc.gid_exists(gid))
                           for gids in (leg.muscle_E, leg.muscle_F))
        self.history = {}
        self.reset()

    def reset(self):
        self.model.reset()
        self.history.clear()
        self.apply_lengths()

    def apply_lengths(self):
        for cells, length_m in zip(self.pools, self.model.lengths_m):
            for cell in cells:
                cell.set_musculotendon_length(length_m * 1000)

    def local_values(self):
        """Force, fibre length, fibre velocity sums for E then F, plus error flag."""
        values = []
        invalid = 0.0
        for cells in self.pools:
            sums = [0.0, 0.0, 0.0]
            for cell in cells:
                sample = cell.mechanical_state()
                if not all(math.isfinite(value) for value in sample) or sample[0] < 0:
                    invalid = 1.0
                else:
                    for i, value in enumerate(sample):
                        sums[i] += value
            values.extend(sums)
        return values, invalid

    def sample(self, time_ms, extensor, flexor):
        if not self.record:
            return
        length_e, length_f = self.model.lengths_m
        row = {
            "time_ms": time_ms,
            "angle_rad": self.model.angle_rad,
            "angular_velocity_rad_s": self.model.velocity_rad_s,
            "length_mt_E_mm": length_e * 1000,
            "length_mt_F_mm": length_f * 1000,
            "force_E_N": extensor.force_n, "force_F_N": flexor.force_n,
            "fiber_length_E_mm": extensor.fiber_length_mm,
            "fiber_length_F_mm": flexor.fiber_length_mm,
            "fiber_velocity_E_mm_s": extensor.fiber_velocity_mm_s,
            "fiber_velocity_F_mm_s": flexor.fiber_velocity_mm_s,
        }
        for name, value in row.items():
            self.history.setdefault(name, array("d")).append(value)


def run_coupled(h, pc, joints, tstop_ms, exchange_ms=0.1, use_psolve=True):
    """Fixed-step force/length exchange, collective on ALL MPI ranks.

    Neural states advance with held path lengths. Forces at the end of each
    interval advance the joint, whose new lengths are held for the next interval.
    Histories use this exchange time grid, not the membrane recording grid.
    """
    dt = float(h.dt)
    if (not all(math.isfinite(value) for value in (dt, tstop_ms, exchange_ms))
            or dt <= 0 or exchange_ms <= 0 or tstop_ms <= 0):
        raise ValueError('Simulation times must be finite and positive')
    if not math.isclose(float(h.t), 0.0, abs_tol=1e-9):
        raise ValueError('Initialize NEURON before starting a coupled run')
    if h.cvode_active():
        raise ValueError('The coupled solver requires fixed-step electrical integration')
    ticks = round(exchange_ms / dt)
    end_tick = round(tstop_ms / dt)
    if (ticks < 1
            or not math.isclose(ticks * dt, exchange_ms, abs_tol=1e-9)
            or not math.isclose(end_tick * dt, tstop_ms, abs_tol=1e-9)):
        raise ValueError("Stop time and mechanical exchange must be multiples of h.dt")
    if not joints:
        raise ValueError("No joints supplied to the coupled solver")
    if int(pc.nhost()) > 1 and not use_psolve:
        raise ValueError("MPI runs require ParallelContext.psolve")

    def collect():
        values, invalid = [], 0.0
        for joint in joints:
            local, error = joint.local_values()
            values.extend(local)
            invalid += error
        # Include the error flag in the same collective so every rank exits
        # together, including ranks which do not own any muscle cells.
        vector = h.Vector(values + [invalid])
        pc.allreduce(vector, 1)
        if vector[len(values)] != 0:
            raise FloatingPointError("Non-finite muscle state or negative muscle tension")
        # Per joint: 3 extensor sums, then 3 flexor sums (see local_values).
        means = []
        for j, joint in enumerate(joints):
            means.append(tuple(
                MuscleState(*(vector[6*j + 3*m + k] / count for k in range(3)))
                for m, count in enumerate(joint.counts)))
        return means

    for joint in joints:
        joint.reset()
    for joint, (extensor, flexor) in zip(joints, collect()):
        joint.sample(float(h.t), extensor, flexor)

    current_tick = round(float(h.t) / dt)
    while current_tick < end_tick:
        next_tick = min(current_tick + ticks, end_tick)
        target = next_tick * dt
        if use_psolve:
            pc.psolve(target)
        else:
            for _ in range(next_tick - current_tick):
                h.fadvance()
        for joint, (extensor, flexor) in zip(joints, collect()):
            joint.model.step((next_tick - current_tick) * dt / 1000,
                             extensor.force_n, flexor.force_n)
            joint.apply_lengths()
            joint.sample(target, extensor, flexor)
        current_tick = next_tick
