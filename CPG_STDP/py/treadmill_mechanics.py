from collections import defaultdict
import hashlib
import math
import os
import xml.etree.ElementTree as ET

import numpy as np


SIDES = ('L', 'R')
JOINTS = ('hip', 'knee', 'ankle')
GROUPS = ('ankle_E', 'ankle_F', 'hip_E', 'hip_F', 'knee_E', 'knee_F')
PATH_ARMS_M = {'hip_E': -.004, 'hip_F': .004, 'knee_E': .003, 'knee_F': -.003}
PEAK_FORCE_N = {'ankle_E': 8., 'ankle_F': 2.8, 'hip_E': 4., 'hip_F': 3.,
                'knee_E': 4., 'knee_F': 3.}
NEUTRAL_ANGLES = dict(zip(JOINTS, (-.25, .65, .65)))


def source_path():
    from constants import CPG_TREADMILL_XML

    return CPG_TREADMILL_XML


def make_force_scene(xml, controlled_mass_kg, exchange_ms, belt_mps):
    root = ET.fromstring(xml)
    root.set('model', 'memcpg_treadmill')
    root.find('compiler').set('settotalmass', str(controlled_mass_kg))
    root.find('option').set('timestep', str(exchange_ms / 1000))
    world = root.find('worldbody')
    for side in SIDES:
        world.remove(world.find(f"body[@name='pelvis_ref_{side}']"))
    belt = world.find("geom[@name='treadmill']")
    belt.set('surfacevel', f'{-belt_mps} 0 0 0 0 0')
    constraints = root.find('actuator')
    sensors = root.find('sensor')
    root.remove(constraints)
    if sensors is not None:
        root.remove(sensors)
    actuators = ET.SubElement(root, 'actuator')
    tendons = root.find('tendon')
    for side in SIDES:
        toe = world.find(f".//joint[@name='toe_m_jnt_{side}']")
        toe.set('stiffness', '.004')
        for group in GROUPS:
            if group.startswith('ankle'):
                muscle = 'sol' if group.endswith('_E') else 'ta'
                tendon_name = f'{muscle}_tendon_m_{side}'
            else:
                joint = group.split('_')[0]
                tendon_name = f'{group}_path_{side}'
                tendon = ET.SubElement(tendons, 'fixed', name=tendon_name)
                ET.SubElement(tendon, 'joint', joint=f'{joint}_m_{side}',
                              coef=str(PATH_ARMS_M[group]))
            ET.SubElement(actuators, 'motor', name=f'tension_{group}_{side}',
                          tendon=tendon_name, gear='1')
    return ET.tostring(root, encoding='unicode')


class TreadmillBody:
    def __init__(self, path, exchange_ms, belt_mps):
        import mujoco

        self.mj = mujoco
        source = path.read_bytes()
        original = mujoco.MjModel.from_xml_string(source.decode())
        controlled_mass = self._controlled_mass(original)
        self.model = model = mujoco.MjModel.from_xml_string(
            make_force_scene(source, controlled_mass, exchange_ms, belt_mps))
        self.data = data = mujoco.MjData(model)
        self.pairs = [(side, group) for side in SIDES for group in GROUPS]
        self.names = [f'{side}_{group}' for side, group in self.pairs]
        self.motor_ids = [model.actuator(f'tension_{group}_{side}').id for side, group in self.pairs]
        self.tendon_ids = [int(model.actuator_trnid[i, 0]) for i in self.motor_ids]
        self.joint_ids = {side: {joint: model.joint(f'{joint}_m_{side}').id for joint in JOINTS}
                          for side in SIDES}
        self.belt_id = model.geom('treadmill').id
        self.paw_geoms = {side: {model.geom(f'{part}_m_geom_{side}').id for part in ('foot', 'toe')}
                          for side in SIDES}
        for joints in self.joint_ids.values():
            for joint, joint_id in joints.items():
                data.qpos[model.jnt_qposadr[joint_id]] = NEUTRAL_ANGLES[joint]
        mujoco.mj_forward(model, data)
        low = min(float(data.geom_xpos[g, 2] - model.geom_size[g, 0]
                        - model.geom_size[g, 1] * abs(data.geom_xmat[g, 8]))
                  for geoms in self.paw_geoms.values() for g in geoms)
        model.geom_pos[self.belt_id, 2] = low + .0005
        mujoco.mj_forward(model, data)
        self.path_offset_m = np.array([
            0. if group.startswith('ankle') else .025 - data.ten_length[tendon]
            for (_, group), tendon in zip(self.pairs, self.tendon_ids)])
        self.metadata = {
            'source_xml_sha256': hashlib.sha256(source).hexdigest(), 'source_xml': str(path),
            'engine_version': mujoco.__version__, 'belt_mps': belt_mps, 'exchange_ms': exchange_ms,
            'pelvis': 'fixed supports', 'muscle_groups': self.names,
            'controlled_mass_kg': controlled_mass, 'initial_angles_rad': NEUTRAL_ANGLES,
            'proximal_dlength_dangle_m': PATH_ARMS_M, 'peak_force_N': PEAK_FORCE_N,
            'initial_path_length_mm': self.lengths_mm().tolist(),
        }

    def _controlled_mass(self, model):
        roots = {model.body(f'pelvis_m_{side}').id for side in SIDES}
        total = 0.
        for body in range(1, model.nbody):
            ancestor = body
            while ancestor and ancestor not in roots:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor:
                total += model.body_mass[body]
        if total <= 0:
            raise ValueError('The treadmill XML has no mass in the controlled limbs')
        return float(total)

    def lengths_mm(self):
        lengths = 1000 * (self.data.ten_length[self.tendon_ids] + self.path_offset_m)
        if not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise FloatingPointError('Invalid musculotendon length returned by MuJoCo')
        return lengths

    def step(self, tension_n):
        self.data.ctrl[self.motor_ids] = -np.asarray(tension_n)
        self.mj.mj_step(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('Non-finite MuJoCo joint state')
        return self.lengths_mm().tolist()

    def sample(self, lengths, muscle_states, time_ms):
        row = {'time_ms': time_ms}
        model, data = self.model, self.data
        for side, joints in self.joint_ids.items():
            for joint, idx in joints.items():
                row[f'{side}_{joint}_angle_rad'] = data.qpos[model.jnt_qposadr[idx]]
                row[f'{side}_{joint}_velocity_rad_s'] = data.qvel[model.jnt_dofadr[idx]]
        forces = {side: 0. for side in SIDES}
        wrench = np.zeros(6)
        for i in range(data.ncon):
            contact = data.contact[i]
            if self.belt_id not in (contact.geom1, contact.geom2):
                continue
            for side, geoms in self.paw_geoms.items():
                if contact.geom1 in geoms or contact.geom2 in geoms:
                    self.mj.mj_contactForce(model, data, i, wrench)
                    forces[side] += wrench[0]
        for side, force in forces.items():
            row[f'{side}_contact_N'] = force
        for name, length, (force, fibre, velocity) in zip(self.names, lengths, muscle_states):
            row[f'{name}_path_length_mm'] = length
            row[f'{name}_fiber_length_mm'] = fibre
            row[f'{name}_fiber_velocity_mm_s'] = velocity
            row[f'{name}_force_N'] = force
        return row


class TreadmillCoupling:
    def __init__(self, h, pc, legs, exchange_ms, tstop_ms, feedback=None):
        if not math.isfinite(exchange_ms) or exchange_ms <= 0:
            raise ValueError('CPG_TREADMILL_DT_MS must be finite and positive')
        if len(legs) != len(SIDES):
            raise ValueError('Pass the left and right leg to the treadmill coupling')
        self.h, self.pc = h, pc
        self.master = int(pc.id()) == 0
        self.exchange_ms, self.tstop_ms = exchange_ms, tstop_ms
        self.feedback = feedback
        self.body = None
        self.history = defaultdict(list)
        self.metadata = self._on_master(self._make_body)
        if feedback is not None:
            self.metadata['sensory_feedback'] = feedback.metadata
        self.counts = [len(leg.treadmill_muscles[group]) for leg in legs for group in GROUPS]
        self.pools = [tuple(pc.gid2cell(gid) for gid in leg.treadmill_muscles[group]
                            if pc.gid_exists(gid)) for leg in legs for group in GROUPS]
        for cells, rest_mm, group in zip(self.pools, self.metadata['initial_path_length_mm'], GROUPS * len(legs)):
            for cell in cells:
                cell.configure_mechanics(rest_mm, fiber_length_mm=.6*rest_mm)
                for segment in cell.muscle_unit:
                    segment.p0_fHill = PEAK_FORCE_N[group]

    def _make_body(self):
        belt_mps = float(os.environ.get('CPG_BELT_MPS', '.2'))
        if not math.isfinite(belt_mps) or belt_mps < 0:
            raise ValueError('CPG_BELT_MPS must be finite and non-negative')
        self.body = TreadmillBody(source_path(), self.exchange_ms, belt_mps)
        return self.body.metadata

    def _on_master(self, operation):
        message = None
        if self.master:
            try:
                message = (None, operation())
            except Exception as exc:
                message = (f'{type(exc).__name__}: {exc}', None)
        error, value = self.pc.py_broadcast(message, 0)
        if error is not None:
            raise RuntimeError(f'Treadmill: {error}')
        return value

    def _collect(self):
        local = np.zeros((len(self.pools), 3))
        invalid = 0.
        for idx, cells in enumerate(self.pools):
            for cell in cells:
                sample = cell.mechanical_state()
                if not np.isfinite(sample).all() or sample[0] < 0:
                    invalid = 1.
                else:
                    local[idx] += sample
        vector = self.h.Vector(np.r_[local.ravel(), invalid])
        self.pc.allreduce(vector, 1)
        reduced = np.array(vector.to_python())
        if reduced[-1]:
            raise FloatingPointError('Non-finite muscle state or negative tension')
        return reduced[:-1].reshape(-1, 3) / np.asarray(self.counts)[:, None]

    def _sample_and_step(self, time_ms, muscle_states, advance):
        lengths = (self.body.step(muscle_states[:, 0]) if advance
                   else self.body.lengths_mm().tolist())
        if not math.isclose(self.body.data.time * 1000, time_ms, rel_tol=0, abs_tol=1e-6):
            raise RuntimeError('MuJoCo did not reach the requested time; physics may have reset')
        row = self.body.sample(lengths, muscle_states, time_ms)
        currents = None
        if self.feedback is not None:
            currents = self.feedback.update(row, time_ms, min(time_ms + self.exchange_ms, self.tstop_ms))
        for key, value in row.items():
            self.history[key].append(value)
        return lengths, currents

    def run(self, use_psolve):
        dt = float(self.h.dt)
        ticks = round(self.exchange_ms / dt)
        steps = round(self.tstop_ms / self.exchange_ms)
        if (ticks < 1 or steps < 1 or self.h.cvode_active()
                or not math.isclose(ticks * dt, self.exchange_ms, abs_tol=1e-9)
                or not math.isclose(steps * self.exchange_ms, self.tstop_ms, abs_tol=1e-9)
                or not math.isclose(self.h.t, 0., abs_tol=1e-9)):
            raise ValueError('Initialize at t=0 and use a fixed dt dividing the exchange and run duration')
        for step in range(steps + 1):
            target_ms = step * self.exchange_ms
            if step:
                if use_psolve:
                    self.pc.psolve(target_ms)
                else:
                    for _ in range(ticks):
                        self.h.fadvance()
            muscle_states = self._collect()
            lengths, currents = self._on_master(
                lambda: self._sample_and_step(target_ms, muscle_states, step > 0))
            for cells, length in zip(self.pools, lengths):
                for cell in cells:
                    cell.set_musculotendon_length(length)
            if currents is not None:
                self.feedback.apply(currents)
