from neuron import h
import os
import numpy as np
import h5py as hdf5
import logging
import random
import re
from joint_mechanics import JointParameters

logging.basicConfig(filename='logs_new_new_2.log',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.info("let's get it started")

# --- ОТДЕЛЬНЫЕ ЛОГГЕРЫ ДЛЯ addgener и genconnect ---

formatter = logging.Formatter(
    fmt='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

# addgener log
logger_addgener = logging.getLogger("addgener")
logger_addgener.setLevel(logging.DEBUG)

handler_add = logging.FileHandler("addgener.log", mode="w")
handler_add.setFormatter(formatter)
logger_addgener.addHandler(handler_add)

# genconnect log
logger_genconnect = logging.getLogger("genconnect")
logger_genconnect.setLevel(logging.DEBUG)

handler_conn = logging.FileHandler("genconnect.log", mode="w")
handler_conn.setFormatter(formatter)
logger_genconnect.addHandler(handler_conn)

h.load_file("stdgui.hoc")
h.load_file('nrngui.hoc')
h.load_file('stdrun.hoc')

# paralleling NEURON stuff
def check_mpi_status():
    # nrniv -mpi initializes MPI before Python is entered. Do not replace the
    # real ranks with 0/1 when an optional version-specific diagnostic is absent.
    pc = h.ParallelContext()
    rank = int(pc.id())
    nhost = int(pc.nhost())
    print(f'MPI Status: rank {rank} of {nhost}')
    return pc, rank, nhost

# Initialize MPI properly
pc, rank, nhost = check_mpi_status()
file_name = 'res_alina_50_stdp'
RANDOM_SEED = 1729

N = 5
speed = 100
bs_fr = 100  # 40 # frequency of brainstem inputs
versions = 1
CV_number = 6
k = 0.017  # CV weights multiplier to take into account air and toe stepping
CV_0_len = 12  # 125 # Duration of the CV generator with no sensory inputs
extra_layers = 0  # 1 + layers

step_number = int(os.environ.get('CPG_STEP_NUMBER', 6))
if step_number < 1:
    raise ValueError('CPG_STEP_NUMBER must be positive')

one_step_time = int((6 * speed + CV_0_len) / (int(1000 / bs_fr))) * (int(1000 / bs_fr))
time_sim = (one_step_time * step_number + 30)*2

IB_MIN_RATE_HZ = 40  # Ib generator never drops below its baseline rate inside its phase
STDP_MAX_WEIGHT_FACTOR = 2.0  # prevent CV->RG_E weights from dominating the rhythm
STDP_HEBB_STEP_FRACTION = 0.01  # at most 1% of the initial weight per LTP event

'''Locomotion mode / injury, set per run from the environment (defaults = intact network):
   CPG_SPEED=125|100|50  CPG_K=0.017|0.01  CPG_BWS=0..0.9  CPG_INJURY=0..1  CPG_OUT=<results dir>'''
speed = int(os.environ.get("CPG_SPEED", speed))
# CUT strength per stepping mode, as in rat_cpg_stdp.py: plantar 0.017, TOE 0.01, QUAD 0.003, AIR 0.001
k = float(os.environ.get("CPG_K", k))
BWS = float(os.environ.get("CPG_BWS", 0.0))  # body weight support: scales the CUT afferent rate by (1 - BWS)
# injury of the extensor side, see leg.injury_scales(): fraction of the Ib_aff_E drive (to RG_E and
# directly to mns_E) removed at t=0; at INJURY = 1 CUT -> RG_E is CUT_RESIDUAL of the plantar weight
INJURY = float(os.environ.get("CPG_INJURY", 0.0))
CUT_RESIDUAL = float(os.environ.get("CPG_CUT_RESIDUAL", 0.015))  # calibrate_injury.py: visible in all modes
if not 0.0 <= BWS < 1.0:
    raise ValueError(f"CPG_BWS must be in [0, 1), got {BWS}")
if not 0.0 <= INJURY <= 1.0:
    raise ValueError(f"CPG_INJURY must be in [0, 1], got {INJURY}")
one_step_time = int((6 * speed + CV_0_len) / (int(1000 / bs_fr))) * (int(1000 / bs_fr))
time_sim = (one_step_time * step_number + 30)*2
if "CPG_OUT" in os.environ:
    file_name = os.environ["CPG_OUT"]
elif (speed, k, BWS, INJURY) != (100, 0.017, 0.0, 0.0):
    file_name = f"res_sp{speed}_k{k:g}_bws{BWS:g}_inj{INJURY:g}"

# One independent mechanical joint per leg. Set CPG_JOINT=0 for a fixed-path
# control run (the updated muscle/tendon model is still used).
if os.environ.get('CPG_JOINT', '1') not in ('0', '1'):
    raise ValueError('CPG_JOINT must be 0 or 1')
JOINT_ENABLED = os.environ.get('CPG_JOINT', '1') == '1'
JOINT_EXCHANGE_MS = float(os.environ.get('CPG_JOINT_DT_MS', '0.1'))
if not np.isfinite(JOINT_EXCHANGE_MS) or JOINT_EXCHANGE_MS <= 0:
    raise ValueError('CPG_JOINT_DT_MS must be finite and positive')
# Physical joint parameters are edited in joint_mechanics.JointParameters.
JOINT_PARAMETERS = JointParameters()
if JOINT_ENABLED and 'CPG_OUT' not in os.environ:
    file_name += '_joint'
if 'CPG_TSTOP_MS' in os.environ:
    time_sim = float(os.environ['CPG_TSTOP_MS'])
    if not np.isfinite(time_sim) or time_sim <= 0:
        raise ValueError('CPG_TSTOP_MS must be finite and positive')

'''Ia pathway: spindles now read physical contractile-fibre length. Keep the
neural gain explicit for comparisons; CPG_IA_SCALE=0 records spindle activity
without changing the CPG, and a positive value enables its synaptic drive.'''
IA_WEIGHT_SCALE = float(os.environ.get("CPG_IA_SCALE", 0.0))
IA_BASE_RATE_HZ = 50.0  # tonic Ia rate at resting length and zero stretch velocity

STDP_RECOVERY_WMAX_FACTOR = 1.0  # wmax = factor * INJURY * w_Ib
STDP_RECOVERY_HEBB_FRACTION = 0.02  # LTP step, fraction of the post-injury CUT weight per event
STDP_RECOVERY_LTD_RATIO = 1.0  # LTD step at w = wmax relative to the LTP step

k_nrns = 0
k_name = 1

global_gid = 0
