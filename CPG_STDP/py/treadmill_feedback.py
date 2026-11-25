import random

from constants import (BWS, CV_number, IA_BASE_RATE_HZ, IA_LENGTH_GAIN_HZ_PER_MM,
                       IA_VELOCITY_GAIN_HZ_PER_MM_S, RANDOM_SEED, h, k, pc, rank, speed)
from treadmill_mechanics import GROUPS, PEAK_FORCE_N, SIDES
from utils_cpg import genconnect, get_gid


CUT_THRESHOLD_N = .04
CUT_FULL_SCALE_N = .6
CUT_MAX_HZ = 90.
IB_THRESHOLD_FRACTION = .015
IB_FULL_SCALE_FRACTION = .2
IB_MAX_HZ = 100.
PHASE_GROUPS = {phase: tuple(group for group in GROUPS if group.endswith(phase)) for phase in 'EF'}


def load_rate(load, threshold, full_scale, max_rate):
    return max_rate * min(1., max(0., (load - threshold) / (full_scale - threshold)))


def relative_tension(row, side, phase):
    groups = PHASE_GROUPS[phase]
    return sum(row[f'{side}_{group}_force_N'] / PEAK_FORCE_N[group] for group in groups) / len(groups)


class SensorySource:
    def __init__(self, leg, gid, phase):
        self.phase = phase
        self.gid = gid
        pc.set_gid2node(gid, 0)
        if rank == 0:
            relay = h.IntFire1()
            relay.refrac = 0
            self.event = h.NetCon(None, relay)
            self.event.weight[0] = 2
            output = h.NetCon(relay, None)
            spikes = h.Vector()
            output.record(spikes)
            pc.cell(gid, output)
            leg.stims.append(relay)
            leg.netcons.extend((self.event, output))
            leg.gen_spike_vectors.append((gid, spikes))

    def advance(self, rate_hz, start_ms, end_ms):
        if rate_hz <= 0 or end_ms <= start_ms:
            return
        cycles = rate_hz * (end_ms - start_ms) / 1000
        total = self.phase + cycles
        count = int(total)
        for spike in range(count):
            self.event.event(min(end_ms, start_ms + (spike + 1 - self.phase) * 1000 / rate_hz))
        self.phase = total - count


class TreadmillFeedback:
    def __init__(self, legs, *, cut, ib):
        self.cut = cut
        self.ib = ib
        self.sources = {}
        self.cut_gids = {side: [] for side in SIDES}
        self.metadata = {
            'cut': {'enabled': cut, 'threshold_N': CUT_THRESHOLD_N, 'full_scale_N': CUT_FULL_SCALE_N,
                    'max_Hz': CUT_MAX_HZ * (1 - BWS)},
            'ib': {'enabled': ib, 'threshold_fraction': IB_THRESHOLD_FRACTION,
                   'full_scale_fraction': IB_FULL_SCALE_FRACTION, 'max_Hz': IB_MAX_HZ,
                   'input': 'mean peak-normalized tendon tension of hip, knee and ankle for each phase'},
        }
        for side, leg in zip(SIDES, legs):
            cut_gids = [get_gid() for _ in range(CV_number)]
            ib_gids = {phase: get_gid() for phase in 'EF'}
            if cut:
                key = f'{side}_CUT_rate_Hz'
                offset = random.Random(f'{RANDOM_SEED}:{side}:CUT').random()
                self.sources[key] = []
                for layer, gid in enumerate(cut_gids):
                    source = SensorySource(leg, gid, (layer + offset) / CV_number)
                    self.sources[key].append(source)
                    self.cut_gids[side].append(source.gid)
                    genconnect(leg, source.gid, leg.dict_CV_pool[layer], .15 * k * speed, 2, N=20,
                               gen_name='CUT_load', target_name=f'CV_{layer + 1}')
            if ib:
                for phase, targets, contacts in (('E', leg.Ib_aff_E, 20), ('F', leg.Ib_aff_F, 30)):
                    offset = random.Random(f'{RANDOM_SEED}:{side}:Ib:{phase}').random()
                    source = SensorySource(leg, ib_gids[phase], offset)
                    self.sources[f'{side}_Ib_{phase}_rate_Hz'] = [source]
                    leg.gener_Ibgids.append(source.gid)
                    genconnect(leg, source.gid, targets, .01, 1, N=contacts,
                               gen_name=f'Ib_tension_{phase}', target_name=f'Ib_aff_{phase}')
        self.metadata['source_gids'] = {key: [source.gid for source in sources]
                                        for key, sources in self.sources.items()}

    def update(self, row, start_ms, end_ms):
        for side in SIDES:
            row[f'{side}_CUT_rate_Hz'] = (load_rate(row[f'{side}_contact_N'], CUT_THRESHOLD_N,
                                                   CUT_FULL_SCALE_N, CUT_MAX_HZ) * (1 - BWS)
                                         if self.cut else 0.)
            for phase in PHASE_GROUPS:
                row[f'{side}_Ib_{phase}_rate_Hz'] = (load_rate(relative_tension(row, side, phase), IB_THRESHOLD_FRACTION,
                                                            IB_FULL_SCALE_FRACTION, IB_MAX_HZ)
                                                   if self.ib else 0.)
        for key, sources in self.sources.items():
            for source in sources:
                source.advance(row[key], start_ms, end_ms)


class TreadmillReceptors:
    CUT_MAX_NA = 1.5
    IB_MAX_NA = 1.5
    IA_NA_PER_HZ = .012
    IA_MAX_NA = .95
    HIP_IA_REST_NA = .20
    HIP_IA_LENGTH_NA_PER_MM = .85
    HIP_IA_VELOCITY_NA_PER_MM_S = .018

    def __init__(self, legs):
        self.rest_length = {}
        self.clamps = {}
        for side, leg in zip(SIDES, legs):
            for name in ('CV', 'Ia_aff_E', 'Ia_aff_F', 'Ib_aff_E', 'Ib_aff_F', 'Ia_hip_F'):
                channel = 'CUT' if name == 'CV' else name.replace('_aff', '')
                self.clamps[f'{side}_{channel}_current_nA'] = [
                    self._clamp(gid) for gid in getattr(leg, name) if pc.gid_exists(gid)]
        self.metadata = {'model': 'membrane_currents', 'artificial_spike_sources': False,
                         'cut_max_nA': self.CUT_MAX_NA, 'ib_max_nA': self.IB_MAX_NA,
                         'cut_threshold_N': CUT_THRESHOLD_N, 'cut_full_scale_N': CUT_FULL_SCALE_N,
                         'ib_threshold_fraction': IB_THRESHOLD_FRACTION,
                         'ib_full_scale_fraction': IB_FULL_SCALE_FRACTION,
                         'ia_nA_per_Hz': self.IA_NA_PER_HZ, 'ia_max_nA': self.IA_MAX_NA,
                         'hip_ia_rest_nA': self.HIP_IA_REST_NA,
                         'hip_ia_length_nA_per_mm': self.HIP_IA_LENGTH_NA_PER_MM,
                         'hip_ia_velocity_nA_per_mm_s': self.HIP_IA_VELOCITY_NA_PER_MM_S}

    @staticmethod
    def _clamp(gid):
        clamp = h.IClamp(pc.gid2cell(gid).soma(.5))
        clamp.dur = 1e9
        return clamp

    def update(self, row, _start_ms, _end_ms):
        currents = {}
        for side in SIDES:
            currents[f'{side}_CUT_current_nA'] = (1 - BWS) * load_rate(
                row[f'{side}_contact_N'], CUT_THRESHOLD_N, CUT_FULL_SCALE_N, self.CUT_MAX_NA)
            for phase in PHASE_GROUPS:
                currents[f'{side}_Ib_{phase}_current_nA'] = load_rate(
                    relative_tension(row, side, phase), IB_THRESHOLD_FRACTION, IB_FULL_SCALE_FRACTION, self.IB_MAX_NA)
                group = f'{side}_ankle_{phase}'
                length = row[f'{group}_fiber_length_mm']
                rest = self.rest_length.setdefault(group, length)
                velocity = row[f'{group}_fiber_velocity_mm_s']
                spindle_drive = (IA_BASE_RATE_HZ + IA_LENGTH_GAIN_HZ_PER_MM * (length - rest)
                                 + IA_VELOCITY_GAIN_HZ_PER_MM_S * velocity)
                currents[f'{side}_Ia_{phase}_current_nA'] = min(self.IA_MAX_NA, max(0., spindle_drive * self.IA_NA_PER_HZ))
            group = f'{side}_hip_F'
            length = row[f'{group}_fiber_length_mm']
            stretch = length - self.rest_length.setdefault(group, length)
            velocity = row[f'{group}_fiber_velocity_mm_s']
            drive = (self.HIP_IA_REST_NA + self.HIP_IA_LENGTH_NA_PER_MM * stretch
                     + self.HIP_IA_VELOCITY_NA_PER_MM_S * velocity)
            currents[f'{side}_Ia_hip_F_current_nA'] = min(self.IA_MAX_NA, max(0., drive))
        row.update(currents)
        return currents

    def apply(self, currents):
        for name, clamps in self.clamps.items():
            for clamp in clamps:
                clamp.amp = currents[name]
