from bioaffrat import bioaffrat
from interneuron import interneuron
from motoneuron import motoneuron
from muscle import muscle
from constants import *


def addpool(leg, num, name, neurontype="int") -> list:
    '''
    Creates pool of cells determined by the neurontype and returns gids of the pool
    Parameters
    ----------
    num: int
        neurons number in pool
    name: string
        the name of the pool
    neurontype: string
        int: interneuron
        delay: interneuron with 5ht
        bursting: interneuron with bursting
        moto: motor neuron
        aff: afferent
        muscle: muscle fiber
    Returns
    -------
    gids: list
        the list of cells gids
    '''
    gids = []
    all_gids = []  # All GIDs for this pool across all ranks

    delaytype = False
    if neurontype.lower() == "delay":
        delaytype = True

    if neurontype.lower() == "moto":
        diams = motodiams(num)

    # Create GIDs for all neurons in pool (distributed across ranks)
    for i in range(num):
        gid = get_gid()
        all_gids.append(gid)
        owner_rank = i % nhost
        pc.set_gid2node(gid, owner_rank)
        # Only create cell if this rank is responsible for this neuron
        if owner_rank == rank:
            if neurontype.lower() == "moto":
                cell = motoneuron(diams[i])
                leg.motos.append(cell)
            elif neurontype.lower() == "aff":
                cell = bioaffrat()
                leg.affs.append(cell)
            elif neurontype.lower() == "muscle":
                cell = muscle()
                leg.muscles.append(cell)
            elif neurontype.lower() == "bursting":
                cell = interneuron(False, bursting_mode=True)
                leg.ints.append(cell)
            else:
                cell = interneuron(delaytype)
                leg.ints.append(cell)

            gids.append(gid)
            # pc.set_gid2node(gid, rank)
            nc = cell.connect2target(None)
            pc.cell(gid, nc)
            # log_gid_by_lookup(leg, gid, neurontype.lower())
            leg.netcons.append(nc)

    # Groups - store all GIDs, not just local ones
    if neurontype.lower() == "muscle":
        leg.musclegroups.append((all_gids, name))
    elif neurontype.lower() == "moto":
        leg.motogroups.append((all_gids, name))
    elif neurontype.lower() == "aff":
        leg.affgroups.append((all_gids, name))
    else:
        leg.intgroups.append((all_gids, name))

    return all_gids


def connectcells(leg, pre_cells, post_cells, weight=1.0, delay=1, threshold=10, inhtype=False,
                 stdptype=False, N=50, sect="int", pre_name="UNKNOWN_PRE", post_name="UNKNOWN_POST"):
    #print(f"🔗 [rank {rank}] connectcells: pre_cells={len(pre_cells)}, post_cells={len(post_cells)}")
    #print(f"   weight={weight}, delay={delay}, threshold={threshold}, inhtype={inhtype}, stdptype={stdptype}")
    logging.info(
        f"connectcells start | "
        f"{pre_name}({len(pre_cells)}) -> {post_name}({len(post_cells)}) | "
        f"stdp={stdptype}, inh={inhtype}"
    )

    connection_count = 0
    pre_cells_list = list(pre_cells)
    connection_key = (
        f"{pre_name}:{post_name}:{pre_cells_list}:"
        f"{weight}:{delay}:{inhtype}:{stdptype}"
    )

    for post_idx, post_gid in enumerate(post_cells):
        if pc.gid_exists(post_gid):
            try:
                rng = random.Random(f"{connection_key}:{post_gid}")

                target = pc.gid2cell(post_gid)
                target_type = type(target).__name__
                logging.info(f"Target {post_gid} type: {target_type}")

                nsyn_requested = rng.randint(N, N + 15)

                if stdptype:
                    avail = len(getattr(target, 'synlistexstdp', []))
                elif inhtype:
                    avail = len(getattr(target, 'synlistinh', []))
                else:
                    avail = len(getattr(target, 'synlistex', []))

                nsyn = min(nsyn_requested, avail)
                if nsyn < nsyn_requested:
                    logging.warning(
                        f"connectcells: nsyn clamped {nsyn_requested}->{nsyn} "
                        f"for {target_type} gid={post_gid}"
                    )

                stdp_dummy = h.Section(name=f"stdp_dummy_{post_gid}") if stdptype else None

                for i in range(nsyn):
                    src_gid = rng.choice(pre_cells_list)

                    if stdptype:
                        try:
                            if not hasattr(target, 'synlistexstdp') or len(target.synlistexstdp) <= i:
                                logging.error(f"synlistexstdp index {i} out of range in {target_type}")
                                continue

                            syn = target.synlistexstdp[i]

                            nc = pc.gid_connect(src_gid, syn)
                            nc.delay = delay
                            nc.weight[0] = weight
                            leg.netcons.append(nc)

                            try:
                                stdpmech = h.STDP(0, stdp_dummy)
                                leg.stdpmechs.append(stdpmech)
                            except Exception as stdp_error:
                                logging.error(f"STDP creation error: {stdp_error}")
                                continue

                            presyn = pc.gid_connect(src_gid, stdpmech)
                            presyn.delay = delay
                            presyn.weight[0] = 2
                            leg.presyns.append(presyn)

                            pstsyn = pc.gid_connect(post_gid, stdpmech)
                            pstsyn.delay = delay
                            pstsyn.weight[0] = -2
                            leg.postsyns.append(pstsyn)

                            pointer_ok = False
                            try:
                                h.setpointer(nc._ref_weight[0], 'synweight', stdpmech)
                                pointer_ok = True
                            except Exception as pointer_error:
                                logging.error(f"Pointer error: {pointer_error}")

                            if pointer_ok:
                                weight_changes = h.Vector()
                                weight_changes.record(stdpmech._ref_synweight, 10.0)
                                leg.weight_changes_vectors.append((src_gid, post_gid, weight_changes))

                            connection_count += 1

                        except Exception as stdp_conn_error:
                            logging.error(f"STDP connection error {src_gid}->{post_gid}: {stdp_conn_error}")

                    else:
                        # print(f"     🔗 Creating regular connection...")
                        try:
                            if inhtype:
                                if not hasattr(target, 'synlistinh'):
                                    # print(f"     ❌ Target {target_type} has no synlistinh")
                                    continue
                                syn = target.synlistinh[i]
                                # print(f"     ✅ Got inhibitory synapse")
                            else:
                                if not hasattr(target, 'synlistex'):
                                    # print(f"     ❌ Target {target_type} has no synlistex")
                                    continue
                                syn = target.synlistex[i]
                                # print(f"     ✅ Got excitatory synapse")

                            nc = pc.gid_connect(src_gid, syn)
                            nc.weight[0] = rng.gauss(weight, weight / 5)
                            nc.delay = rng.gauss(delay, delay / 5)
                            leg.netcons.append(nc)
                            # print(f"     ✅ Regular NetCon created")
                            connection_count += 1

                        except Exception as reg_conn_error:
                            # print(f"     ❌ Regular connection error: {reg_conn_error}")
                            logging.error(f"Regular connection error {src_gid}->{post_gid}: {reg_conn_error}")

            except Exception as target_error:
                # print(f"   ❌ Error getting target for GID {post_gid}: {target_error}")
                logging.error(f"Target error {post_gid}: {target_error}")

        else:
            pass

    # print(f"🏁 connectcells finished: {connection_count} connections created")
    logging.info(f"connectcells end: {connection_count} connections created")


def genconnect(leg, gen_gid, afferents_gids, weight, delay, inhtype=False, N=50, gen_name="GEN", target_name="TARGET"):
    logger_genconnect.info(
        f"genconnect start | leg={leg.name} | "
        f"{gen_name}({gen_gid}) -> {target_name}({len(afferents_gids)}) | "
        f"weight={weight} | delay={delay} | inhtype={inhtype}"
    )
    for i in afferents_gids:
        if pc.gid_exists(i):
            try:
                rng = random.Random(f"{gen_name}_{target_name}_{i}_{gen_gid}")
                nsyn_requested = rng.randint(max(0, N - 5), N)

                target = pc.gid2cell(i)
                target_type = type(target).__name__

                syn_list_name = 'synlistinh' if inhtype else 'synlistex'
                if not hasattr(target, syn_list_name):
                    logging.error(f"Target {target_type} (gid={i}) has no attribute {syn_list_name}")
                    continue

                syn_list = getattr(target, syn_list_name)
                avail = len(syn_list)
                nsyn = min(nsyn_requested, avail)

                if nsyn < nsyn_requested:
                    logging.warning(
                        f"genconnect: nsyn clamped {nsyn_requested}->{nsyn} "
                        f"for {target_type} gid={i}"
                    )
                for j in range(nsyn):
                    if len(syn_list) <= j:
                        break
                    syn = syn_list[j]

                    nc = pc.gid_connect(gen_gid, syn)
                    nc.delay = rng.gauss(delay, delay / 5)
                    nc.weight[0] = rng.gauss(weight, weight / 6)

                    # ---------------------------------------
                    # ЛОГИРУЕМ СОЕДИНЕНИЕ
                    # ---------------------------------------
                    logger_genconnect.info(
                        "NetCon created | %s(%s) -> %s(%s) | syn_index=%s | "
                        "threshold=%.4f | delay=%.4f | weight=%.4f | inhtype=%s",
                        gen_name,
                        gen_gid,
                        target_name,
                        i,
                        j,
                        nc.threshold,
                        nc.delay,
                        nc.weight[0],
                        inhtype
                    )
                    # ---------------------------------------
                    leg.stimnclist.append(nc)

            except Exception as e:
                logging.error(f"Error in genconnect for target gid {i}: {e}")


def motodiams(number):
    nrn_number = number
    standby_percent = 70
    standby_size = int(nrn_number * standby_percent / 100)
    active_size = nrn_number - standby_size

    loc_active, scale_active = 27, 3
    loc_stanby, scale_stanby = 44, 4

    rng = np.random.default_rng(RANDOM_SEED + number)
    x2 = np.concatenate([
        rng.normal(loc=loc_active, scale=scale_active, size=active_size),
        rng.normal(loc=loc_stanby, scale=scale_stanby, size=standby_size),
    ])

    return x2


def add_bs_geners(freq, LEG_L, LEG_R):
    left_E_bs_gids = []
    left_F_bs_gids = []
    right_E_bs_gids = []
    right_F_bs_gids = []

    interval = int(1000 / freq)
    number = int(one_step_time / interval) - 2
    _is_rank0 = (rank == 0)
    _NetStim = h.NetStim
    _NetCon = h.NetCon
    _Vector = h.Vector
    _set_gid2node = pc.set_gid2node
    _cell = pc.cell

    for step in range(step_number):
        f_start = one_step_time * (2 * step + 1)
        e_start = int(one_step_time * 2 * step) + 10

        for leg_obj, start, gid_list in (
            (LEG_R, f_start, right_F_bs_gids),
            (LEG_R, e_start, right_E_bs_gids),
            (LEG_L, f_start, left_F_bs_gids),
            (LEG_L, e_start, left_E_bs_gids),
        ):
            gid = get_gid()
            if _is_rank0:
                stim = _NetStim()
                stim.start = start
                stim.interval = interval
                stim.number = number
                logger_addgener.info(
                    "STIM created | gid=%s | start=%.3f | interval=%s | number=%s  | cv=%s | r=%s",
                    gid, stim.start, interval, number, False, False
                )
                leg_obj.stims.append(stim)
                _set_gid2node(gid, rank)
                ncstim = _NetCon(stim, None)
                spike_times = _Vector()
                ncstim.record(spike_times)
                leg_obj.gen_spike_vectors.append((gid, spike_times))
                leg_obj.netcons.append(ncstim)
                _cell(gid, ncstim)
                log_gid_by_lookup(leg_obj, gid, "gen")
            else:
                _set_gid2node(gid, 0)
            leg_obj.gener_gids.append(gid)
            gid_list.append(gid)

    return left_E_bs_gids, left_F_bs_gids, right_E_bs_gids, right_F_bs_gids

def log_gid_by_lookup(leg, gid: int, name):
    if not pc.gid_exists(gid):
        print(f"[rank {rank}] GID {gid} not assigned to this process.")
        return

    obj = pc.gid2cell(gid)
    typename = type(obj).__name__
    if name:
        print(f"[rank {rank}] Added GID {gid} (type: {typename}) - name: {name}")
    else:
        print(f"[rank {rank}] Added GID {gid} (type: {typename})")


def addgener(leg, start, freq, cv=False, r=True):
    '''
    Creates generator and returns generator gid
    Parameters
    ----------
    start: int
        generator start up
    freq: int
        generator frequency
    nums: int
        signals number
    Returns
    -------
    gid: int
        generator gid
    '''
    gid = get_gid()
    # Only create generator on rank 0 to avoid duplicates
    if rank == 0:
        stim = h.NetStim()

        if r:
            rng = random.Random(gid)
            stim.start = rng.uniform(start - 3, start + 3)
            stim.noise = 0.05
            if hasattr(stim, "noiseFromRandom"):
                nrn_rng = h.Random()
                nrn_rng.Random123(RANDOM_SEED, gid, 0)
                stim.noiseFromRandom(nrn_rng)
                leg.netstims.append(nrn_rng)
        else:
            stim.start = start

        interval = int(1000 / freq)
        stim.interval = interval

        if cv:
            base_cv = int(one_step_time / interval) // CV_number
            stim.number = int(1.45 * base_cv)
        else:
            stim.number = int(one_step_time / interval) - 2

        # -----------------------------------------
        # ЛОГИРУЕМ ВСЕ ПАРАМЕТРЫ STIM
        # -----------------------------------------
        logger_addgener.info(
            "STIM created | gid=%s | start=%.3f | interval=%s | number=%s  | cv=%s | r=%s",
            gid, stim.start, interval, stim.number, cv, r
        )
        # -----------------------------------------

        leg.stims.append(stim)
        pc.set_gid2node(gid, rank)
        ncstim = h.NetCon(stim, None)
        spike_times = h.Vector()
        ncstim.record(spike_times)
        leg.gen_spike_vectors.append((gid, spike_times))
        leg.netcons.append(ncstim)
        pc.cell(gid, ncstim)
        # log_gid_by_lookup(leg, gid, "gen")

    else:
        # Other ranks just need to know the GID is assigned to rank 0
        pc.set_gid2node(gid, 0)

    leg.gener_gids.append(gid)

    return gid


def create_connect_bs(LEG_L, LEG_R):
    LEG_L.left_E_bs_gids, LEG_L.left_F_bs_gids, LEG_R.right_E_bs_gids, LEG_R.right_F_bs_gids = add_bs_geners(bs_fr, LEG_L, LEG_R)

    ''' BS '''
    '''Left leg'''
    for E_bs_gid in LEG_L.left_E_bs_gids:
        for layer in range(CV_number):
            genconnect(LEG_L, E_bs_gid, LEG_L.dict_RG_E[layer], 1.75, 1, gen_name="E_bs_gid", target_name=f"LEG_L_RG_E_{layer+1}")

    for F_bs_gid in LEG_L.left_F_bs_gids:
        for layer in range(CV_number):
            genconnect(LEG_L, F_bs_gid, LEG_L.dict_RG_F[layer], 1.75, 1, gen_name="F_bs_gid", target_name=f"LEG_L_RG_F_{layer+1}")

    '''Right leg'''
    for E_bs_gid in LEG_R.right_E_bs_gids:
        for layer in range(CV_number):
            genconnect(LEG_R, E_bs_gid, LEG_R.dict_RG_E[layer], 1.75, 1, gen_name="E_bs_gid", target_name=f"LEG_R_RG_E_{layer+1}")

    for F_bs_gid in LEG_R.right_F_bs_gids:
        for layer in range(CV_number):
            genconnect(LEG_R, F_bs_gid, LEG_R.dict_RG_F[layer], 1.75, 1, gen_name="F_bs_gid", target_name=f"LEG_R_RG_F_{layer+1}")


def create_connect_bs_command(LEG_L, LEG_R, mode=None):
    from bs_command import BSCommand, apply_bs_mode_to_cpg, WALK
    if mode is None:
        mode = WALK
    cmd = BSCommand(mode=mode)
    cmd.connect(LEG_L, LEG_R)
    apply_bs_mode_to_cpg(LEG_L, LEG_R, mode=mode)
    logging.info("create_connect_bs_command: mode=%s done", mode.name)
    return cmd


def add_external_connections(LEG_L, LEG_R, mode=None):
    if mode is None:
        from bs_command import WALK
        mode = WALK

    if mode.v3f_cross_weight > 0:
        connectcells(
            LEG_L, LEG_L.V3F, LEG_R.RG_F,
            weight=mode.v3f_cross_weight, delay=3,
            pre_name="V3F_L", post_name="RG_F_R",
        )
        connectcells(
            LEG_R, LEG_R.V3F, LEG_L.RG_F,
            weight=mode.v3f_cross_weight, delay=3,
            pre_name="V3F_R", post_name="RG_F_L",
        )
    connectcells(LEG_L, LEG_L.V0v, LEG_R.In1, weight=1.3, delay=3)
    connectcells(LEG_R, LEG_R.V0v, LEG_L.In1, weight=1.3, delay=3)
    connectcells(LEG_L, LEG_L.V0d, LEG_R.RG_F, weight=1.3, delay=3, inhtype=True)
    connectcells(LEG_R, LEG_R.V0d, LEG_L.RG_F, weight=1.3, delay=3, inhtype=True)


def safe_filename(name: str) -> str:
    """Преобразует строку в безопасное имя файла."""
    return re.sub(r'[^\w\-_.]', '_', name)


def get_gid():
    global global_gid
    global_gid += 1
    return global_gid
