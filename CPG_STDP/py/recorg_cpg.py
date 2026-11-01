from constants import *
from dataclasses import asdict
import json

OVERVIEW_ORDER = ("RG_F", "RG_E", "InF", "InE", "V2a", "V0v", "V0d", "V3f")


def save_joint_recordings(joints, version):
    """Save mechanics on its own exchange grid, with explicit units/parameters."""
    if rank != 0:
        return
    for joint in joints:
        if not joint.history:
            continue
        stem = os.path.join(file_name, f'joint_{joint.name}_v{version}')
        with hdf5.File(stem + '.hdf5', 'w') as output:
            output.attrs['parameters_si'] = json.dumps(asdict(joint.model.parameters))
            output.attrs['force_aggregation'] = 'population mean of representative units'
            output.attrs['exchange_ms'] = JOINT_EXCHANGE_MS
            output.attrs['time_grid'] = 'mechanical exchange endpoints'
            for name, values in joint.history.items():
                output.create_dataset(name, data=np.asarray(values), compression='gzip')

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        figure = Figure(figsize=(10, 9), constrained_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.subplots(4, 1, sharex=True)
        history = joint.history
        time_s = np.asarray(history['time_ms']) / 1000
        axes[0].plot(time_s, np.degrees(history['angle_rad']))
        axes[0].set_ylabel('Angle (deg)')
        for side, color in (('E', 'tab:blue'), ('F', 'tab:orange')):
            axes[1].plot(time_s, history[f'length_mt_{side}_mm'], color=color, label=side)
            axes[2].plot(time_s, history[f'fiber_length_{side}_mm'], color=color, label=side)
            axes[3].plot(time_s, history[f'force_{side}_N'], color=color, label=side)
        for axis, label in zip(axes[1:], ('Path length (mm)', 'Fibre length (mm)', 'Tension (N)')):
            axis.set_ylabel(label)
            axis.legend(loc='upper right')
        for axis in axes:
            axis.grid(alpha=0.25)
        axes[-1].set_xlabel('Time (s)')
        figure.suptitle(f'{joint.name}: separate extensor E and flexor F')
        figure.savefig(stem + '.png', dpi=150)
        logging.info('Joint recordings saved: %s', stem)


def normalize_overview_name(raw_name):
    if not raw_name:
        return None
    name = str(raw_name)
    lower = name.lower()
    if name.startswith("RG") and name.endswith("_F"):
        return "RG_F"
    if name.startswith("RG") and name.endswith("_E"):
        return "RG_E"
    if lower.startswith("inf"):
        return "InF"
    if lower.startswith("ine"):
        return "InE"
    if lower.startswith("v2a"):
        return "V2a"
    if lower.startswith("v0v"):
        return "V0v"
    if lower.startswith("v0d"):
        return "V0d"
    if lower.startswith("v3f"):
        return "V3f"
    return None


def _average_trace_list(trace_list):
    if not trace_list:
        return np.empty(0, dtype=np.float32)
    min_len = min(np.asarray(t).size for t in trace_list if np.asarray(t).size > 0)
    if min_len <= 0:
        return np.empty(0, dtype=np.float32)
    acc = np.zeros(min_len, dtype=np.float32)
    used = 0
    for trace in trace_list:
        arr = np.asarray(trace, dtype=np.float32)
        if arr.size < min_len:
            continue
        acc += arr[:min_len]
        used += 1
    if used == 0:
        return np.empty(0, dtype=np.float32)
    return acc / used


def build_overview_series(left_named_traces, right_named_traces):
    def collect(entries):
        buckets = {key: [] for key in OVERVIEW_ORDER}
        for raw_name, trace in entries:
            key = normalize_overview_name(raw_name)
            if key is None:
                continue
            arr = np.asarray(trace, dtype=np.float32)
            if arr.size == 0:
                continue
            buckets[key].append(arr)
        ordered = []
        for key in OVERVIEW_ORDER:
            avg = _average_trace_list(buckets[key])
            if avg.size:
                ordered.append((key, avg))
        return ordered

    left_ordered = collect(left_named_traces)
    right_ordered = collect(right_named_traces)

    series = []
    for name, arr in left_ordered:
        series.append((f"{name} left", arr))
    left_count = len(series)
    for name, arr in right_ordered:
        series.append((f"{name} right", arr))
    return series, left_count


def gather_population_average(pool, v_vec):
    global rank
    pc.barrier()

    n_local = len(v_vec)
    if n_local > 0:
        outavg = np.asarray(v_vec[0], dtype=np.float32).copy()
        for j in range(1, n_local):
            outavg += np.asarray(v_vec[j], dtype=np.float32)
        outavg /= n_local
    else:
        outavg = np.empty(0, dtype=np.float32)

    vec = h.Vector()
    vec.from_python(outavg)

    pc.barrier()
    result = pc.py_gather(vec, 0)

    if rank != 0:
        return None

    if not result:
        return np.empty(0, dtype=np.float32)

    arrays = []
    for r in result:
        arr = np.asarray(r, dtype=np.float32)
        if arr.size > 0:
            arrays.append(arr)

    if not arrays:
        return np.empty(0, dtype=np.float32)

    min_len = min(arr.size for arr in arrays)
    if min_len <= 0:
        return np.empty(0, dtype=np.float32)

    out = np.zeros(min_len, dtype=np.float32)
    for arr in arrays:
        out += arr[:min_len]
    out /= len(arrays)
    return out


def draw_overview_plot(t_values, labeled_traces, output_path, title="CPG overview", separator_after=0):
    global rank
    if rank != 0:
        return
    if not labeled_traces:
        return

    import matplotlib
    if not os.environ.get("DISPLAY", "").strip():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_values = np.asarray(t_values, dtype=np.float32)
    fig_height = max(6, 0.45 * len(labeled_traces))
    fig, ax = plt.subplots(figsize=(14, fig_height))

    y_positions = []
    y_labels = []
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])

    for idx, (label, trace) in enumerate(labeled_traces):
        arr = np.asarray(trace, dtype=np.float32)
        if arr.size == 0:
            continue
        n = min(arr.size, t_values.size)
        if n == 0:
            continue
        arr = arr[:n]
        t_local = t_values[:n]

        span = float(np.max(arr) - np.min(arr))
        if span < 1e-9:
            norm = np.zeros_like(arr)
        else:
            norm = (arr - np.median(arr)) / span

        row = len(labeled_traces) - idx
        y = row + 0.8 * norm
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(t_local, y, lw=1.0, color=color)
        y_positions.append(row)
        y_labels.append(label)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    if separator_after > 0 and separator_after < len(labeled_traces):
        y_sep = len(labeled_traces) - separator_after + 0.5
        ax.axhline(y=y_sep, color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel("t (ms)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    if t_values.size:
        ax.set_xlim(float(t_values[0]), float(t_values[-1]))
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close(fig)


def spike_record(pool, extra=False, location='soma', max_units = None, seed = 0):
    ''' Records spikes from gids
      Parameters
      ----------
      pool: list
        list of neurons gids
      extra: bool
          extracellular or intracellular voltages to record
      Returns
      -------
      v_vec: list of h.Vector()
          recorded voltages
    '''

    pool = list(pool)
    pool = [gid for gid in pool if pc.gid_exists(gid)]

    if max_units is not None and len(pool) > max_units:
        rng = np.random.default_rng(seed)
        pool = rng.choice(pool, size=max_units, replace=False).tolist()

    n = len(pool)
    n_pts = int(time_sim / 0.025 + 200)
    v_vec = [None] * n
    _gid2cell = pc.gid2cell
    _Vector = h.Vector

    if extra:
        get_ref = lambda c: c.soma(0.5)._ref_vext[0]
    elif location == 'axon':
        get_ref = lambda c: c.node[0](1.0)._ref_v
    elif location == 'muscle':
        get_ref = lambda c: c.muscle_unit(0.5)._ref_v
    elif location == 'am':
        get_ref = lambda c: c.muscle_unit(0.5)._ref_AM_CaSP
    else:
        get_ref = lambda c: c.soma(0.5)._ref_v

    for idx in range(n):
        cell = _gid2cell(pool[idx])
        vec = _Vector(n_pts)
        vec.record(get_ref(cell))
        v_vec[idx] = vec
    return v_vec


def force_record(pool):
    ''' Records force from gids of motor neurons muscle unit
      Parameters
      ----------
      Returns
      -------
      v_vec: list of h.Vector()
          recorded voltage
    '''
    v_vec = []
    for i in pool:
        if not pc.gid_exists(i):
            continue
        cell = pc.gid2cell(i)
        vec = h.Vector(np.zeros(int(time_sim / 0.025 + 1), dtype=np.float32))
        vec.record(cell.muscle_unit(0.5)._ref_F_fHill)
        v_vec.append(vec)
    return v_vec


def velocity_record(gids, attr='_ref_vel'):
    """
    Records velocity-related variable (vel or v0) from IbGenerator instances

    Parameters
    ----------
    gids : list of int
        gids of IbGenerator processes
    attr : str
        which attribute to record ('_ref_vel' or '_ref_v0')

    Returns
    -------
    vecs : list of h.Vector()
        list of recorded vectors for each gid
    """
    vecs = []
    for gid in gids:
        if not pc.gid_exists(gid):
            continue
        cell = pc.gid2cell(gid)
        vec = h.Vector(np.zeros(int(time_sim / 0.025 + 1), dtype=np.float32))
        vec.record(getattr(cell, attr))
        vecs.append(vec)
    return vecs


def spikeout(pool, name, version, v_vec, leg):
    ''' Reports simulation results
      Parameters
      ----------
      pool: list
        list of neurons gids
      name: string
        pool name
      version: int
          test number
      v_vec: list of h.Vector()
          recorded voltage
    '''
    global rank
    pc.barrier()

    out = gather_population_average(pool, v_vec)
    if rank == 0:
        logging.info("start recording " + name)
        if out.size == 0:
            logging.warning(f"skip recording {name}: empty averaged trace")
            return

        bs_offset = int(1000 / bs_fr) * 40
        step_width = one_step_time * 40
        fname = f'./{file_name}/{name}_sp_{speed}_CVs_{CV_number}_bs_{bs_fr}_{leg}.hdf5'
        with hdf5.File(fname, 'w') as file:
            for i in range(step_number * 2):
                start = bs_offset + i * step_width
                end = start + step_width
                if start >= out.size:
                    logging.warning(f"{name}: step {i} start={start} exceeds recorded length {out.size}, stopping early")
                    break
                actual_end = min(end, out.size)
                if actual_end < end:
                    logging.warning(f"{name}: step {i} truncated at {actual_end} (expected {end}), recorded data too short")
                file.create_dataset(f'#0_step_{i}', data=out[start:actual_end], compression="gzip")
        logging.info("done recording average")
    else:
        logging.info(rank)

def setup_recorders(leg, recorder_list, group_attr, group_name):
    """Настраивает рекордеры для указанной группы нейронов"""
    print(f"      Setting up {group_name} recorders...")
    recorder_list.extend([spike_record(group[k_nrns], max_units=2, seed=123) for group in getattr(leg, group_attr)])


def generator_spikeout(gen_vecs, name, version, leg):
    """
    Saves generator spike times
    Parameters
    ----------
    gen_vecs : list of (gid, h.Vector)
        recorded generator spike times
    name : str
        generator group name
    version : int
        test number
    leg : str
        left / right
    """
    global rank
    pc.barrier()

    if rank == 0:
        gen_dir = f'./{file_name}/{name}_generators'
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)

    pc.barrier()

    local_data = [(gid, list(vec)) for gid, vec in gen_vecs]

    gathered = pc.py_gather(local_data, 0)

    if rank == 0:
        logging.info(f"start recording generators {name}")

        # gathered = [rank0_data, rank1_data, ...]
        for rank_data in gathered:
            if rank_data is None:
                continue
            for gid, spikes in rank_data:
                fname = (
                    f'{gen_dir}/gen_{gid}_rank{rank}_'
                    f'sp_{speed}_CVs_{CV_number}_bs_{bs_fr}_{leg}_v{version}.hdf5'
                )
                with hdf5.File(fname, 'w') as f:
                    f.create_dataset(
                        'spike_times',
                        data=np.array(spikes, dtype=np.float32),
                        compression="gzip"
                    )

        logging.info("done recording generators")
