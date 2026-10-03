from constants import *
from utils_cpg import *


CUT_REFERENCE_WEIGHT = 0.0035 * 0.017 * 100  # CUT -> RG_E of plantar walking at speed 100


def injury_scales(injury=None, cut_residual=None):
    '''Weight factors of the extensor pathways after injury (all 1 for the intact network).
    Every input below fires RG_E / mns_E on its own when left at full strength: the Ia_aff_E
    and CUT pools fire in synchronous volleys, which drive RG_E even at 1% of their weight.'''
    injury = INJURY if injury is None else injury
    cut_residual = CUT_RESIDUAL if cut_residual is None else cut_residual
    w_cut = 0.0035 * k * speed
    return {
        "ia_rg": 1 - injury,  # Ia_aff_E -> RG_E
        "ia_mn": 1 - injury,  # Ia_aff_E -> mns_E: monosynaptic path that bypasses RG_E
        # CUT -> RG_E: after a complete injury the same weak synapse (cut_residual of the plantar
        # weight, near the RG_E firing threshold) in every mode; the modes then differ in how
        # many CUT volleys arrive (speed, BWS)
        "cut": 1 - injury + injury * cut_residual * CUT_REFERENCE_WEIGHT / w_cut,
    }


def cut_recovery_stdp(w_ia, w_cut, injury=None, hebb_fraction=None, ltd_ratio=None):
    '''STDP parameters of CUT -> RG_E after injury ({} for the intact network).
    w_cut is the CUT weight right after the injury; LTP steps are a fraction of it, so a
    few pairings cannot undo the injury at once.'''
    injury = INJURY if injury is None else injury
    hebb_fraction = STDP_RECOVERY_HEBB_FRACTION if hebb_fraction is None else hebb_fraction
    ltd_ratio = STDP_RECOVERY_LTD_RATIO if ltd_ratio is None else ltd_ratio
    if injury <= 0:
        return {}
    # CUT -> RG_E may grow up to the removed Ia weight
    cut_wmax = STDP_RECOVERY_WMAX_FACTOR * injury * w_ia
    hebbwt = hebb_fraction * w_cut
    return dict(
        stdp_wmax=cut_wmax,
        stdp_hebbwt=hebbwt,
        # stdp.mod scales LTD by synweight**2/wmax: at w = wmax it equals ltd_ratio * the LTP step
        stdp_antiwt=-ltd_ratio * hebbwt / cut_wmax,
    )


class LEG:

    def __init__(self, speed, bs_fr, inh_p, step_number, n, leg_l=False):
        logging.info(f"Hello from rank {rank} of {nhost}")
        logging.info("NEURON version: " + h.nrnversion())
        self.name = "LEG left?=" + str(leg_l)
        self.leg_l = leg_l
        self.threshold = 10
        self.delay = 1
        self.nAff = 5 #15 #35 #5
        self.nInt = 5 #10 #21 #5
        self.nMn = 5 #10 #21 #5
        self.ncell = n
        self.affs = []
        self.ints = []
        self.motos = []
        self.muscles = []
        self.affgroups = []
        self.intgroups = []
        self.motogroups = []
        self.musclegroups = []
        self.gener_gids = []
        self.gener_Iagids = []
        self.gen_spike_vectors = []
        # self.n_gid = get_gid()

        self.RG_E = []  # Rhythm generators of extensors
        self.RG_F = []  # Rhythm generators of flexor
        self.CV = []
        self.netstims = []
        self.stims = []

        self.stdpmechs = []
        self.netcons = []
        self.stimnclist = []

        self.presyns = []
        self.postsyns = []

        self.weight_changes_vectors = []
        self.time_t_vectors = []

        self.C_0 = []
        self.V0v = []
        self.V3F = []
        self.V0d = []
        self.V2a = []

        self.dict_CV_pool = {} # CV pools
        self.dict_RG_E ={} # RG Extensor pools
        self.dict_RG_F = {} # RG Flexor pools
        self.dict_V3F ={} # V3F pools
        self.dict_CV_gener = {} # cutaneous generators

        '''cut and muscle feedback'''
        for layer in range(CV_number):
            '''Cutaneous pools'''
            self.dict_CV_pool[layer] = addpool(self, self.ncell, "CV" + str(layer + 1) + "_1", "aff")
            '''Rhythm generator pools'''
            self.dict_RG_E[layer] = addpool(self, self.ncell, "RG" + str(layer + 1) + "_E", "int")
            self.dict_RG_F[layer] = addpool(self, self.ncell, "RG" + str(layer + 1) + "_F", "int")
            self.dict_V3F[layer] = addpool(self, self.ncell, "V3" + str(layer + 1) + "_F", "int")
            self.RG_E.append(self.dict_RG_E[layer])
            self.RG_F.append(self.dict_RG_F[layer])
            self.CV.append(self.dict_CV_pool[layer])
            self.V3F.append(self.dict_V3F[layer])

        '''RG'''
        self.RG_E = sum(self.RG_E, [])
        self.InE = addpool(self, self.nInt, "InE", "int")
        self.RG_F = sum(self.RG_F, [])
        self.InF = addpool(self, self.nInt, "InF", "int")
        self.In1 = addpool(self, self.nInt, "In1", "int")
        self.V0v = addpool(self, self.nInt, "V0v", "int")
        self.V2a = addpool(self, self.nInt, "V2a", "int")
        self.V0d = addpool(self, self.nInt, "V0d", "int")

        self.CV = sum(self.CV, [])
        self.V3F = sum(self.V3F, [])

        '''sensory and muscle afferents and brainstem and V3F'''
        self.Ia_aff_E = addpool(self, self.nAff, "Ia_aff_E", "aff")
        self.Ia_aff_F = addpool(self, self.nAff, "Ia_aff_F", "aff")
        # self.BS_aff_E = addpool(self, self.nAff, "BS_aff_E", "aff")
        # self.BS_aff_F = addpool(self, self.nAff, "BS_aff_F", "aff")
        
        '''muscles'''
        self.muscle_E = addpool(self, self.nMn, "muscle_E", "muscle")
        self.muscle_F = addpool(self, self.nMn, "muscle_F", "muscle")

        '''moto neuron pools'''
        self.mns_E = addpool(self, self.nMn, "mns_E", "moto")
        self.mns_F = addpool(self, self.nMn, "mns_F", "moto")

        # '''reflex arc'''
        self.Ia_E = addpool(self, self.nInt, "Ia_E", "int")
        self.R_E = addpool(self, self.nInt, "R_E", "int")  # Renshaw cells
        self.Ia_F = addpool(self, self.nInt, "Ia_F", "int")
        self.R_F = addpool(self, self.nInt, "R_F", "int")  # Renshaw cells



        '''cutaneous inputs'''
        cfr = 90
        c_int = 1000 / cfr

        '''cutaneous inputs generators'''
        for layer in range(CV_number):
            self.dict_CV_gener[layer] = []
            for i in range(step_number):
                step_leg = 10 + speed * layer + i * (2 * one_step_time) + 7 - layer * 12
                if leg_l:
                    step_leg += one_step_time
                self.dict_CV_gener[layer].append(
                    addgener(self, step_leg, cfr, cv=True, rate_scale=1 - BWS))
                    ## int((one_step_time / CV_number) * 0.15), cv=True))
        #
        # '''Generators'''
        # '''TODO: need it?'''
        # for i in range(step_number):
        #     self.C_0.append(
        #         self.addgener(25 + speed * 6 + i * (speed * 6 + CV_0_len), cfr, int(CV_0_len / c_int), False))
        #

        # ''' BS '''
        # for E_bs_gid in self.E_bs_gids:
        #     self.genconnect(E_bs_gid, self.BS_aff_E, 3.5, 3)
        #
        # for F_bs_gid in self.F_bs_gids:
        #     self.genconnect(F_bs_gid, self.BS_aff_F, 3.5, 3)

        # connectcells(self, self.BS_aff_F, self.V3F, 1.5, 3)
        '''STDP synapse'''
        #connectcells(self, self.BS_aff_F, self.RG_F, 0.1, 3, stdptype=False)
        #connectcells(self, self.BS_aff_E, self.RG_E, 0.1, 3, stdptype=False)

        '''Ia inputs'''
        self.E_ia_gids, self.F_ia_gids = self.add_ia_geners(leg_l)

        for E_ia_gids in self.E_ia_gids:
            genconnect(self, E_ia_gids, self.Ia_aff_E, 0.01, 1, inhtype=False, N=20)

        for F_ia_gids in self.F_ia_gids:
            genconnect(self, F_ia_gids, self.Ia_aff_F, 0.01, 1, inhtype=False, N=30)

        # # '''muscle afferents generators'''
        # self.Iagener_E = self.addIagener(self.muscle_E, self.muscle_F, 10, weight=3)
        # self.Iagener_F = self.addIagener(self.muscle_F, self.muscle_E, one_step_time, weight=3)
        #
        # # # '''Create connectcells'''
        # self.genconnect(self.Iagener_E, self.Ia_aff_E, 5.5, 1, False, 20)
        # self.genconnect(self.Iagener_F, self.Ia_aff_F, 5.5, 1, False, 30)

        # connectcells(self, self.muscle_E, self.Ia_aff_E, 3.5, 1, 10, False)
        # connectcells(self, self.muscle_F, self.Ia_aff_F, 3.5, 1, 10, False)

        w_Ia =  0.3 #0.3 #1.3
        stdp_Ia = False
        stdp_CV = True

        # injury: the extensor pathways keep their synapses and delays (same as the intact
        # network), only the weights are scaled, see injury_scales()
        scales = injury_scales()
        self._scaled(scales["ia_rg"], lambda: connectcells(
            self, self.Ia_aff_E, self.RG_E, weight=w_Ia, delay=3, stdptype=stdp_Ia))
        connectcells(self, self.Ia_aff_F, self.RG_F, weight=w_Ia, delay=3, stdptype=stdp_Ia)

        for layer in range(CV_number):
            for gen_gid in self.dict_CV_gener[layer]:
                genconnect(self, gen_gid, self.dict_CV_pool[layer], 0.15 * k * speed, 2, False, 20)

        '''cutaneous inputs'''
        # connect with the plantar weight (k = 0.017) so the random wiring does not depend on k
        # (connectcells seeds it with the weight), then scale to this mode and injury
        w_cut_plantar = 0.0035 * 0.017 * speed
        cut_factor = k / 0.017 * scales["cut"]
        w_cut = w_cut_plantar * cut_factor
        cut_stdp = cut_recovery_stdp(w_Ia, w_cut) or dict(
            stdp_wmax=w_cut * STDP_MAX_WEIGHT_FACTOR, stdp_hebbwt=w_cut * STDP_HEBB_STEP_FRACTION)
        for layer in range(CV_number):
            self._scaled(cut_factor, lambda: connectcells(
                self, self.dict_CV_pool[layer], self.dict_RG_E[layer], w_cut_plantar, 3,
                stdptype=stdp_CV, **cut_stdp))

        '''Ia2motor'''
        self._scaled(scales["ia_mn"], lambda: connectcells(self, self.Ia_aff_E, self.mns_E, 1.55, 2))
        connectcells(self, self.Ia_aff_F, self.mns_F, 1.55, 2)

        for layer in range(CV_number):
            '''Internal to RG topology'''
            self.connectinsidenucleus(self.dict_RG_F[layer])
            self.connectinsidenucleus(self.dict_RG_E[layer])
            '''RG2Motor'''
            connectcells(self, self.dict_RG_E[layer], self.mns_E, 2.75, 3)
            connectcells(self, self.dict_RG_F[layer], self.mns_F, 2.75, 3)
            connectcells(self, self.dict_RG_F[layer], self.V2a, 0.75, 3)
            connectcells(self, self.dict_RG_F[layer], self.V0d, 0.75, 3)
            connectcells(self, self.dict_RG_E[layer], self.InE, 2.75, 3)
            connectcells(self, self.dict_RG_F[layer], self.InF, 2.75, 3)
            connectcells(self, self.dict_RG_F[layer], self.dict_V3F[layer], 1.5, 3)

        '''motor2muscles'''
        connectcells(self, self.mns_E, self.muscle_E, 10, 2, inhtype=False, N=45, sect="muscle")
        connectcells(self, self.mns_F, self.muscle_F, 10, 2, inhtype=False, N=45, sect="muscle")

        '''Ia2RG, RG2Motor'''
        connectcells(self, self.InE, self.RG_F, 0.8, 1, inhtype=True)
        connectcells(self, self.InF, self.RG_E, 0.8, 1, inhtype=True)

        connectcells(self, self.InE, self.mns_F, 0.8, 1, inhtype=True)
        connectcells(self, self.InF, self.mns_E, 0.4, 1, inhtype=True)

        connectcells(self, self.InE, self.Ia_aff_F, 1.2, 1, inhtype=True)
        connectcells(self, self.InF, self.Ia_aff_E, 0.5, 1, inhtype=True)

        connectcells(self, self.InE, self.InF, 0.04, 1, inhtype=True)
        connectcells(self, self.InF, self.InE, 0.04, 1, inhtype=True)

        connectcells(self, self.In1, self.RG_F, 0.5, 1, inhtype=True)

        connectcells(self, self.Ia_aff_E, self.Ia_E, 0.08, 1, inhtype=False)
        connectcells(self, self.Ia_aff_F, self.Ia_F, 0.08, 1, inhtype=False)

        connectcells(self, self.mns_E, self.R_E, 0.015, 1, inhtype=False)
        connectcells(self, self.mns_F, self.R_F, 0.015, 1, inhtype=False)

        connectcells(self, self.R_E, self.mns_E, 0.015, 1, inhtype=True)
        connectcells(self, self.R_F, self.mns_F, 0.015, 1, inhtype=True)

        connectcells(self, self.R_E, self.Ia_E, 0.001, 1, inhtype=True)
        connectcells(self, self.R_F, self.Ia_F, 0.001, 1, inhtype=True)

        connectcells(self, self.Ia_E, self.mns_F, 0.08, 1, inhtype=True)
        connectcells(self, self.Ia_F, self.mns_E, 0.08, 1, inhtype=True)

        connectcells(self, self.R_E, self.R_F, 0.04, 1, inhtype=True)
        connectcells(self, self.R_F, self.R_E, 0.04, 1, inhtype=True)
        connectcells(self, self.Ia_E, self.Ia_F, 0.08, 1, inhtype=True)
        connectcells(self, self.Ia_F, self.Ia_E, 0.08, 1, inhtype=True)

        ''' Commisural projections '''
        #connectcells(self, self.RG_F, self.V2a, 0.75, 3)
        #connectcells(self, self.RG_F, self.V0d, 0.75, 3)
        connectcells(self, self.V2a, self.V0v, 1.2, 3)

    def addIagener(self, mn: list, mn2: list, start, weight=1.0):
        '''
        Creates self.Ia generators and returns generator gid
        Parameters
        ----------
        mn: list
            motor neurons of agonist muscle that contract spindle
        mn2: list
            motor neurons of antagonist muscle that extend spindle
        start: int
            generator start up
        weight: float
            weight of the connection
        Returns
        -------
        gid: int
            generators gid
        '''
        pair_count = min(len(mn), len(mn2))
        if pair_count == 0:
            raise ValueError("IaGenerator requires non-empty agonist and antagonist muscle pools")

        rng = random.Random(f"{self.name}:Ia:{start}")
        pair_index = rng.randrange(pair_count)
        muscle_gid = mn[pair_index]
        antagonist_gid = mn2[pair_index]
        owner_rank = pair_index % nhost
        gid = get_gid()

        if rank == owner_rank:
            if not pc.gid_exists(muscle_gid) or not pc.gid_exists(antagonist_gid):
                raise RuntimeError(
                    f"IaGenerator gid={gid}: paired muscles {muscle_gid}/{antagonist_gid} "
                    f"are not both local on expected rank {owner_rank}"
                )

            muscle_cell = pc.gid2cell(muscle_gid)
            antagonist_cell = pc.gid2cell(antagonist_gid)
            if not hasattr(muscle_cell, "muscle_unit") or not hasattr(antagonist_cell, "muscle_unit"):
                raise TypeError(
                    f"IaGenerator gid={gid}: selected GIDs are not muscle cells"
                )

            interval = int(1000 / bs_fr)
            number = int(one_step_time / interval) - 2
            stim = h.IaGenerator()
            stim.start = start
            stim.interval = interval
            stim.number = number
            try:
                stim.dur = interval * (number - 1)
                stim.vmin = IA_MIN_RATE_HZ
            except LookupError as err:
                raise RuntimeError(
                    "IaGenerator has no 'dur'/'vmin' parameters: recompile the "
                    "mechanisms (nrnivmodl mod_files) after updating iagen.mod"
                ) from err

            h.setpointer(
                muscle_cell.muscle_unit(0.5)._ref_F_fHill, "fhill", stim
            )
            h.setpointer(
                antagonist_cell.muscle_unit(0.5)._ref_F_fHill, "fhill2", stim
            )

            pc.set_gid2node(gid, owner_rank)
            ncstim = h.NetCon(stim, None)
            ncstim.weight[0] = weight
            spike_times = h.Vector()
            ncstim.record(spike_times)
            pc.cell(gid, ncstim)

            self.stims.append(stim)
            self.netcons.append(ncstim)
            self.gen_spike_vectors.append((gid, spike_times))
            logging.info(
                "IaGenerator created: gid=%s rank=%s pair_index=%s muscles=%s/%s",
                gid, owner_rank, pair_index, muscle_gid, antagonist_gid,
            )
        else:
            pc.set_gid2node(gid, owner_rank)

        self.gener_Iagids.append(gid)
        return gid

    def _scaled(self, factor, connect):
        '''Run connect() and scale the weights of the NetCons it created.'''
        n_before = len(self.netcons)
        connect()
        for nc in self.netcons[n_before:]:
            nc.weight[0] *= factor

    def connectinsidenucleus(self, nucleus, weight=None):
        w = weight if weight is not None else 0.25
        connectcells(self, nucleus, nucleus, pre_name="", post_name="", weight=w, delay=0.5)

    def add_ia_geners(self, leg_l):
        E_ia_gids = []
        F_ia_gids = []
        for step in range(step_number):
            # E_ia_gids.append(self.addIagener(self.muscle_E, self.muscle_F, 10, weight=5))
            # F_ia_gids.append(self.addIagener(self.muscle_F, self.muscle_E, one_step_time, weight=8))
            start_time_e = 15 + one_step_time * 2 * step
            start_time_f = 15 + one_step_time * (1 + 2 * step)
            if leg_l:
                start_time_e, start_time_f = start_time_f, start_time_e
            E_ia_gids.append(self.addIagener(self.muscle_E, self.muscle_F, start_time_e, weight=0.1))
            F_ia_gids.append(
                self.addIagener(self.muscle_F, self.muscle_E, start_time_f, weight=0.1))
        return E_ia_gids, F_ia_gids
