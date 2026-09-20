from constants import *
from utils_cpg import *

class LEG:

    def __init__(self, speed, bs_fr, inh_p, step_number, n, leg_l=False):
        logging.info(f"Hello from rank {rank} of {nhost}")
        logging.info("NEURON version: " + h.nrnversion())
        self.name = "LEG left?=" + str(leg_l)
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

        '''sensory and muscle afferents and brainstem and V3F'''
        self.Ia_aff_E = addpool(self, self.nAff, "Ia_aff_E", "aff")
        self.Ia_aff_F = addpool(self, self.nAff, "Ia_aff_F", "aff")
        # self.BS_aff_E = addpool(self, self.nAff, "BS_aff_E", "aff")
        # self.BS_aff_F = addpool(self, self.nAff, "BS_aff_F", "aff")
        self.V3F = addpool(self, self.nInt, "V3F", "int")
        
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
                step_leg = 10 + speed * layer + i * (speed * CV_number + CV_0_len + one_step_time) + 7 - layer * 12
                if leg_l:
                    step_leg += one_step_time
                self.dict_CV_gener[layer].append(
                    addgener(self, step_leg, cfr, cv=True))
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
        
        connectcells(self, self.Ia_aff_E, self.RG_E, weight=w_Ia, delay=3, stdptype=stdp_Ia)
        connectcells(self, self.Ia_aff_F, self.RG_F, weight=w_Ia, delay=3, stdptype=stdp_Ia)
        
        for layer in range(CV_number):
            for gen_gid in self.dict_CV_gener[layer]:
                genconnect(self, gen_gid, self.dict_CV_pool[layer], 0.15 * k * speed, 2, False, 20)
                
        '''cutaneous inputs'''
        for layer in range(CV_number):
            connectcells(self, self.dict_CV_gener[layer], self.dict_CV_pool[layer], 0.15 * k * speed, 2)
            connectcells(self, self.dict_CV_pool[layer], self.dict_RG_E[layer], 0.0035 * k * speed, 3, stdptype=stdp_CV)

        '''Ia2motor'''
        connectcells(self, self.Ia_aff_E, self.mns_E, 1.55, 2)
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
            connectcells(self, self.dict_RG_F[layer], self.V3F, 1.5, 3)

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
        if not mn or not mn2:
            # print(f"❌ [rank {rank}] Empty motor neuron lists for IaGenerator")
            logging.error("Empty motor neuron lists for IaGenerator")
            return None

        rng = random.Random(start)
        moto_gid = rng.choice(mn)
        moto2_gid = rng.choice(mn2)

        gid = get_gid()
        # print(f"   Assigned GID: {gid}")
        owner_rank = pc.gid2node(moto_gid)
        pc.set_gid2node(gid, owner_rank)
        # Only create on rank 0 to avoid conflicts
        if rank == owner_rank:
            try:

                interval = int(1000 / bs_fr)
                number = int(one_step_time / interval) - 2

                stim = h.IaGenerator()
                stim.start = start
                stim.interval = interval
                stim.number = number

                self.stims.append(stim)

                ncstim = h.NetCon(stim, None)
                ncstim.weight[0] = weight
                self.netcons.append(ncstim)
                pc.cell(gid, ncstim)

                if pc.gid_exists(moto_gid):
                    moto = pc.gid2cell(moto_gid)
                    if hasattr(moto, 'muscle_unit'):
                        try:
                            h.setpointer(moto.muscle_unit(0.5)._ref_F_fHill, 'fhill', stim)
                            print(f"   ✅ First pointer set (fhill)")
                        except Exception as ptr1_error:
                            print(f"   ⚠️ First pointer warning: {ptr1_error}")

                if pc.gid_exists(moto2_gid):
                    moto2 = pc.gid2cell(moto2_gid)
                    if hasattr(moto2, 'muscle_unit'):
                        try:
                            h.setpointer(moto2.muscle_unit(0.5)._ref_F_fHill, 'fhill2', stim)
                            print(f"   ✅ Second pointer set (fhill2)")
                        except Exception as ptr2_error:
                            print(f"   ⚠️ Second pointer warning: {ptr2_error}")

                # log_gid_by_lookup(self, gid, "Ia")
                # print(f"🎯 IaGenerator creation completed successfully: GID={gid}")
                logging.info(f"IaGenerator creation completed: GID={gid}")

            except Exception as ia_error:
                # print(f"❌ IaGenerator creation failed: {ia_error}")
                logging.error(f"IaGenerator creation failed: {ia_error}")
                # Still increment GID to maintain consistency
                pass

        self.gener_Iagids.append(gid)
        return gid

    def connectinsidenucleus(self, nucleus, weight=None):
        w = weight if weight is not None else 0.25
        connectcells(self, nucleus, nucleus, pre_name="", post_name="", weight=w, delay=0.5)

    def setup_autonomous_rhythm(self):
        logging.info("setup_autonomous_rhythm: %s", self.name)

        for layer in range(CV_number):
            connectcells(self, self.dict_RG_E[layer], self.dict_RG_E[layer],
                         pre_name="RG_E_self", post_name="RG_E_self",
                         weight=0.35, delay=0.5)
            connectcells(self, self.dict_RG_F[layer], self.dict_RG_F[layer],
                         pre_name="RG_F_self", post_name="RG_F_self",
                         weight=0.35, delay=0.5)

        connectcells(self, self.InE, self.InF, 0.12, 1, inhtype=True,
                     pre_name="InE_inh_InF", post_name="InF")
        connectcells(self, self.InF, self.InE, 0.12, 1, inhtype=True,
                     pre_name="InF_inh_InE", post_name="InE")

        stim_gid = get_gid()
        owner_rank = 0
        pc.set_gid2node(stim_gid, owner_rank)

        kick_E = None
        if rank == owner_rank:
            kick_E = h.NetStim()
            kick_E.start = 8.0
            kick_E.interval = 5.0
            kick_E.number = 8
            kick_E.noise = 0.0
            self.stims.append(kick_E)

            nc_dummy = h.NetCon(kick_E, None)
            pc.cell(stim_gid, nc_dummy)
            self.netcons.append(nc_dummy)

        n_kick_syn = 10

        for layer in range(CV_number):
            for target_gid in self.dict_RG_E[layer]:
                if pc.gid_exists(target_gid):
                    cell = pc.gid2cell(target_gid)
                    if hasattr(cell, 'synlistex') and cell.synlistex:
                        for si in range(min(n_kick_syn, len(cell.synlistex))):
                            nc = pc.gid_connect(stim_gid, cell.synlistex[si])
                            nc.weight[0] = 0.6
                            nc.delay = 0.5
                            self.stimnclist.append(nc)

            for target_gid in self.dict_RG_F[layer]:
                if pc.gid_exists(target_gid):
                    cell = pc.gid2cell(target_gid)
                    if hasattr(cell, 'synlistex') and cell.synlistex:
                        for si in range(min(n_kick_syn, len(cell.synlistex))):
                            nc = pc.gid_connect(stim_gid, cell.synlistex[si])
                            nc.weight[0] = 0.4
                            nc.delay = 0.5
                            self.stimnclist.append(nc)

        logging.info("setup_autonomous_rhythm done: %s", self.name)

    def add_ia_geners(self, leg_l):
        E_ia_gids = []
        F_ia_gids = []
        for step in range(step_number):
            # E_ia_gids.append(self.addIagener(self.muscle_E, self.muscle_F, 10, weight=5))
            # F_ia_gids.append(self.addIagener(self.muscle_F, self.muscle_E, one_step_time, weight=8))
            start_time_e = 15 + one_step_time * 2 * step
            start_time_f = 15 + one_step_time * (1 + 2 * step)
            if leg_l:
                start_time_e += one_step_time + 4
                start_time_f += one_step_time + 4
            E_ia_gids.append(self.addIagener(self.muscle_E, self.muscle_F, start_time_e, weight=0.1))
            F_ia_gids.append(
                self.addIagener(self.muscle_F, self.muscle_E, start_time_f, weight=0.1))
        return E_ia_gids, F_ia_gids
