import random
from typing import Iterable
import re
import numpy as np
import logging
import h5py as hdf5
import os

from neuron import h
from bioaffrat import bioaffrat
from interneuron import interneuron
from motoneuron import motoneuron
from muscle import muscle

logging.basicConfig(filename='logs_new_new_2.log',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.info("let's get it started")

h.load_file("stdgui.hoc")
h.load_file('nrngui.hoc')
h.load_file('stdrun.hoc')

# paralleling NEURON stuff
def check_mpi_status():
    try:
        # h.nrnmpi_init()
        pc = h.ParallelContext()
        rank = int(pc.id())
        nhost = int(pc.nhost())

        print(f"MPI Status:")
        print(f"  Rank: {rank}")
        print(f"  Number of hosts: {nhost}")
        print(f"  MPI initialized: {h.nrnmpi_is_initialized()}")

        return pc, rank, nhost
    except Exception as e:
        print(f"MPI Error: {e}")
        # Fallback to single process
        pc = h.ParallelContext()
        return pc, 0, 1

# Initialize MPI properly
pc, rank, nhost = check_mpi_status()

file_name = 'res_alina_50_stdp'

N = 2
speed = 100
bs_fr = 100  # 40 # frequency of brainstem inputs
versions = 1
CV_number = 6
k = 0.017  # CV weights multiplier to take into account air and toe stepping
CV_0_len = 12  # 125 # Duration of the CV generator with no sensory inputs
extra_layers = 0  # 1 + layers

step_number = 10

one_step_time = int((6 * speed + CV_0_len) / (int(1000 / bs_fr))) * (int(1000 / bs_fr))
time_sim = one_step_time * step_number + 30

'''
network topology https://github.com/max-talanov/bypass/blob/main/figs/CPG_feedback_loops.png
'''


class CPG:

    def __init__(self, speed, bs_fr, inh_p, step_number, n):
        if not check_mechanisms():
            raise RuntimeError("Required NEURON mechanisms not available")
        logging.info(f"Hello from rank {rank} of {nhost}")
        logging.info("NEURON version: " + h.nrnversion())
        self.threshold = 10
        self.delay = 1
        self.nAff = 35
        self.nInt = 21
        self.nMn = 21
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
        self.gener_Ibgids = []
        self.n_gid = 0

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

        self.C_1 = []
        self.C_0 = []
        # self.V0v = []

        for layer in range(CV_number):
            '''cut and muscle feedback'''
            self.dict_CV_1 = {layer: 'CV{}_1'.format(layer + 1)}
            self.dict_RG_E = {layer: 'RG{}_E'.format(layer + 1)}
            self.dict_RG_F = {layer: 'RG{}_F'.format(layer + 1)}
            # self.dict_V3F = {layer: 'V3{}_F'.format(layer + 1)}
            self.dict_C = {layer: 'C{}'.format(layer + 1)}

        for layer in range(CV_number):
            '''Cutaneous pools'''
            self.dict_CV_1[layer] = self.addpool(self.ncell, "CV" + str(layer + 1) + "_1", "aff")
            '''Rhythm generator pools'''
            self.dict_RG_E[layer] = self.addpool(self.ncell, "RG" + str(layer + 1) + "_E", "int")
            self.dict_RG_F[layer] = self.addpool(self.ncell, "RG" + str(layer + 1) + "_F", "int")
            self.RG_E.append(self.dict_RG_E[layer])
            self.RG_F.append(self.dict_RG_F[layer])
            self.CV.append(self.dict_CV_1[layer])

        '''RG'''
        self.RG_E = sum(self.RG_E, [])
        self.InE = self.addpool(self.nInt, "InE", "int")
        self.RG_F = sum(self.RG_F, [])
        self.InF = self.addpool(self.nInt, "InF", "int")

        self.CV = sum(self.CV, [])

        '''sensory and muscle afferents and brainstem and V3F'''
        self.Ib_aff_E = self.addpool(self.nAff, "Ib_aff_E", "aff")
        self.Ib_aff_F = self.addpool(self.nAff, "Ib_aff_F", "aff")
        # self.BS_aff_E = self.addpool(self.nAff, "BS_aff_E", "aff")
        # self.BS_aff_F = self.addpool(self.nAff, "BS_aff_F", "aff")
        # self.V3F = self.addpool(self.nAff, "V3F", "int")

        '''moto neuron pools'''
        self.mns_E = self.addpool(self.nMn, "mns_E", "moto")
        self.mns_F = self.addpool(self.nMn, "mns_F", "moto")

        '''muscles'''
        self.muscle_E = self.addpool(self.nMn, "muscle_E", "muscle")
        self.muscle_F = self.addpool(self.nMn, "muscle_F", "muscle")

        # '''reflex arc'''
        self.Ib_E = self.addpool(self.nInt, "Ib_E", "int")
        self.R_E = self.addpool(self.nInt, "R_E", "int")  # Renshaw cells
        self.Ib_F = self.addpool(self.nInt, "Ib_F", "int")
        self.R_F = self.addpool(self.nInt, "R_F", "int")  # Renshaw cells

        '''BS'''
        # periodic stimulation
        # periodic stimulation
        self.E_bs_gids, self.F_bs_gids = self.add_bs_geners(bs_fr, 1)

        # self.E_bs_gids = sum(pc.py_allgather(self.E_bs_gids), [])
        # self.F_bs_gids = sum(pc.py_allgather(self.F_bs_gids), [])
        # ''' BS '''
        # for E_bs_gid in self.E_bs_gids:
        #     self.genconnect(E_bs_gid, self.muscle_E, 0.5, 1)
        #
        # for F_bs_gid in self.F_bs_gids:
        #     self.genconnect(F_bs_gid, self.muscle_F, 0.5, 1)

        # ''' BS '''
        for E_bs_gid in self.E_bs_gids:
            for layer in range(CV_number):
                self.genconnect(E_bs_gid, self.dict_RG_E[layer], 3.75, 1)

        for F_bs_gid in self.F_bs_gids:
            for layer in range(CV_number):
                self.genconnect(F_bs_gid, self.dict_RG_F[layer], 3.75, 1)

        self.E_ib_gids, self.F_ib_gids = self.add_ib_geners()

        for E_ib_gids in self.E_ib_gids:
            self.genconnect(E_ib_gids, self.Ib_aff_E, 0.01, 1, False, 20)

        for F_ib_gids in self.F_ib_gids:
            self.genconnect(F_ib_gids, self.Ib_aff_F, 0.01, 1, False, 30)

        # # '''muscle afferents generators'''
        # self.Ibgener_E = self.addIbgener(self.muscle_E, self.muscle_F, 10, weight=3)
        # self.Ibgener_F = self.addIbgener(self.muscle_F, self.muscle_E, one_step_time, weight=3)
        #
        # # # '''Create connectcells'''
        # self.genconnect(self.Ibgener_E, self.Ib_aff_E, 5.5, 1, False, 20)
        # self.genconnect(self.Ibgener_F, self.Ib_aff_F, 5.5, 1, False, 30)

        # self.connectcells(self.muscle_E, self.Ib_aff_E, 3.5, 1, 10, False)
        # self.connectcells(self.muscle_F, self.Ib_aff_F, 3.5, 1, 10, False)

        '''cutaneous inputs'''
        cfr = 90
        c_int = 1000 / cfr

        '''cutaneous inputs generators'''
        for layer in range(CV_number):
            self.dict_C[layer] = []
            for i in range(step_number):
                self.dict_C[layer].append(
                    self.addgener(
                        10 + speed * layer + i * (speed * CV_number + CV_0_len + one_step_time) + 7 - layer * 12, cfr, True, int((one_step_time / CV_number) * 0.15), cv=True))
        #
        # '''Generators'''
        # '''TODO: need it?'''
        # for i in range(step_number):
        #     self.C_0.append(
        #         self.addgener(25 + speed * 6 + i * (speed * 6 + CV_0_len), cfr, int(CV_0_len / c_int), False))
        #
        # '''TODO: need it?'''
        for layer in range(CV_number):
            self.C_1.append(self.dict_CV_1[layer])
        self.C_1 = sum(self.C_1, [])

        # ''' BS '''
        # for E_bs_gid in self.E_bs_gids:
        #     self.genconnect(E_bs_gid, self.BS_aff_E, 3.5, 3)
        #
        # for F_bs_gid in self.F_bs_gids:
        #     self.genconnect(F_bs_gid, self.BS_aff_F, 3.5, 3)

        # self.connectcells(self.BS_aff_F, self.V3F, 1.5, 3)
        '''STDP synapse'''
        # self.connectcells(self.BS_aff_F, self.RG_F, 0.1, 3, stdptype=False)
        # self.connectcells(self.BS_aff_E, self.RG_E, 0.1, 3, stdptype=False)

        self.connectcells(self.Ib_aff_E, self.RG_E, weight=1.3, delay=3, stdptype=True)
        self.connectcells(self.Ib_aff_F, self.RG_F, weight=1.3, delay=3, stdptype=True)

        # '''cutaneous inputs'''
        for layer in range(CV_number):
            self.connectcells(self.dict_C[layer], self.dict_CV_1[layer], 0.15 * k * speed, 2)
            self.connectcells(self.dict_CV_1[layer], self.dict_RG_E[layer], 0.0035 * k * speed, 3)

        # '''Ib2motor'''
        self.connectcells(self.Ib_aff_E, self.mns_E, 1.55, 2)
        self.connectcells(self.Ib_aff_F, self.mns_F, 1.55, 2)

        for layer in range(CV_number):
            '''Internal to RG topology'''
            self.connectinsidenucleus(self.dict_RG_F[layer])
            self.connectinsidenucleus(self.dict_RG_E[layer])

            '''RG2Motor'''
            self.connectcells(self.dict_RG_E[layer], self.mns_E, 2.75, 3)
            self.connectcells(self.dict_RG_F[layer], self.mns_F, 2.75, 3)
            #
            self.connectcells(self.dict_RG_E[layer], self.InE, 2.75, 3)
            self.connectcells(self.dict_RG_F[layer], self.InF, 2.75, 3)

            # self.connectcells(self.dict_RG_F[layer], self.V3F, 1.5, 3)

        '''motor2muscles'''
        self.connectcells(self.mns_E, self.muscle_E, 10, 2, inhtype=False, N=45, sect="muscle")
        self.connectcells(self.mns_F, self.muscle_F, 10, 2, inhtype=False, N=45, sect="muscle")

        # '''Ib2RG, RG2Motor'''
        self.connectcells(self.InE, self.RG_F, 0.5, 1, inhtype=True)
        self.connectcells(self.InF, self.RG_E, 0.8, 1, inhtype=True)

        self.connectcells(self.Ib_aff_E, self.Ib_E, 0.08, 1, inhtype=False)
        self.connectcells(self.Ib_aff_F, self.Ib_F, 0.08, 1, inhtype=False)

        self.connectcells(self.mns_E, self.R_E, 0.015, 1, inhtype=False)
        self.connectcells(self.mns_F, self.R_F, 0.015, 1, inhtype=False)

        self.connectcells(self.R_E, self.mns_E, 0.015, 1, inhtype=True)
        self.connectcells(self.R_F, self.mns_F, 0.015, 1, inhtype=True)

        self.connectcells(self.R_E, self.Ib_E, 0.001, 1, inhtype=True)
        self.connectcells(self.R_F, self.Ib_F, 0.001, 1, inhtype=True)
        #
        self.connectcells(self.Ib_E, self.mns_F, 0.08, 1, inhtype=True)
        self.connectcells(self.Ib_F, self.mns_E, 0.08, 1, inhtype=True)

    def addpool(self, num, name, neurontype = "int") -> list:
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
            diams = self.motodiams(num)
            
        # Create GIDs for all neurons in pool (distributed across ranks)
        for i in range(num):
            gid = self.n_gid + i
            all_gids.append(gid)
            
            # Only create cell if this rank is responsible for this neuron
            if i % nhost == rank:
                if neurontype.lower() == "moto":
                    cell = motoneuron(diams[i])
                    self.motos.append(cell)
                elif neurontype.lower() == "aff":
                    cell = bioaffrat()
                    self.affs.append(cell)
                elif neurontype.lower() == "muscle":
                    cell = muscle()
                    self.muscles.append(cell)
                elif neurontype.lower() == "bursting":
                    cell = interneuron(False, bursting_mode=True)
                    self.ints.append(cell)
                else:
                    cell = interneuron(delaytype)
                    self.ints.append(cell)

                gids.append(gid)
                pc.set_gid2node(gid, rank)
                nc = cell.connect2target(None)
                pc.cell(gid, nc)
                self.log_gid_by_lookup(gid, neurontype.lower())
                self.netcons.append(nc)

        # Update n_gid for next pool
        self.n_gid += num

        # Groups - store all GIDs, not just local ones
        if neurontype.lower() == "muscle":
            self.musclegroups.append((all_gids, name))
        elif neurontype.lower() == "moto":
            self.motogroups.append((all_gids, name))
        elif neurontype.lower() == "aff":
            self.affgroups.append((all_gids, name))
        else:
            self.intgroups.append((all_gids, name))

        return all_gids

    def connectcells(self, pre_cells, post_cells, weight=1.0, delay=1, threshold=10, inhtype=False,
                     stdptype=False, N=50, sect="int"):
        print(f"🔗 [rank {rank}] connectcells: pre_cells={len(pre_cells)}, post_cells={len(post_cells)}")
        print(f"   weight={weight}, delay={delay}, threshold={threshold}, inhtype={inhtype}, stdptype={stdptype}")
        logging.info(f"connectcells start: pre={len(pre_cells)}, post={len(post_cells)}, stdp={stdptype}")

        nsyn = random.randint(N, N + 15)
        print(f"   nsyn={nsyn}")

        connection_count = 0

        for post_idx, post_gid in enumerate(post_cells):
            print(f"   Processing post_cell {post_idx + 1}/{len(post_cells)}: gid={post_gid}")

            if pc.gid_exists(post_gid):
                print(f"   ✅ GID {post_gid} exists on this rank")

                try:
                    target = pc.gid2cell(post_gid)
                    target_type = type(target).__name__
                    print(f"   Target type: {target_type}")
                    logging.info(f"Target {post_gid} type: {target_type}")

                    for i in range(nsyn):
                        src_gid = random.randint(pre_cells[0], pre_cells[-1])
                        print(f"     Synapse {i + 1}/{nsyn}: src_gid={src_gid} -> post_gid={post_gid}")

                        if stdptype:
                            print(f"     🧠 Creating STDP connection...")
                            logging.info(f"STDP connection: {src_gid} -> {post_gid}")

                            try:
                                # Проверяем наличие STDP синапсов
                                if not hasattr(target, 'synlistexstdp'):
                                    print(f"     ❌ Target {target_type} has no synlistexstdp")
                                    logging.error(f"No synlistexstdp in {target_type}")
                                    continue

                                if len(target.synlistexstdp) <= i:
                                    print(f"     ❌ synlistexstdp[{i}] out of range (len={len(target.synlistexstdp)})")
                                    logging.error(f"synlistexstdp index {i} out of range")
                                    continue

                                syn = target.synlistexstdp[i]
                                print(f"     ✅ Got STDP synapse: {type(syn).__name__}")

                                # Создаем основное соединение
                                nc = pc.gid_connect(src_gid, syn)
                                nc.delay = delay
                                nc.weight[0] = weight
                                nc.threshold = threshold
                                pc.threshold(src_gid, threshold)
                                self.netcons.append(nc)
                                print(f"     ✅ Main NetCon created")

                                # Создаем STDP механизм
                                print(f"     Creating STDP mechanism...")
                                dummy = h.Section()  # Create a dummy section to put the point processes in
                                print(f"     ✅ Dummy section created")

                                try:
                                    stdpmech = h.STDP(0, dummy)
                                    print(f"     ✅ STDP mechanism created: {type(stdpmech).__name__}")
                                    self.stdpmechs.append(stdpmech)
                                except Exception as stdp_error:
                                    print(f"     ❌ STDP creation failed: {stdp_error}")
                                    logging.error(f"STDP creation error: {stdp_error}")
                                    continue

                                # Пресинаптическое соединение
                                print(f"     Creating presynaptic connection...")
                                presyn = pc.gid_connect(src_gid, stdpmech)
                                presyn.delay = delay
                                presyn.weight[0] = 2
                                presyn.threshold = threshold
                                self.presyns.append(presyn)
                                print(f"     ✅ Presynaptic NetCon created")

                                # Постсинаптическое соединение
                                print(f"     Creating postsynaptic connection...")
                                pstsyn = pc.gid_connect(post_gid, stdpmech)
                                pstsyn.delay = delay
                                pstsyn.weight[0] = -2
                                pstsyn.threshold = threshold
                                self.postsyns.append(pstsyn)
                                pc.threshold(post_gid, threshold)
                                print(f"     ✅ Postsynaptic NetCon created")

                                # Установка указателя
                                print(f"     Setting pointer...")
                                try:
                                    h.setpointer(nc._ref_weight[0], 'synweight', stdpmech)
                                    print(f"     ✅ Pointer set successfully")
                                except Exception as pointer_error:
                                    print(f"     ❌ Pointer setting failed: {pointer_error}")
                                    logging.error(f"Pointer error: {pointer_error}")

                                # Запись изменений весов
                                weight_changes = h.Vector()
                                weight_changes.record(stdpmech._ref_synweight)
                                self.weight_changes_vectors.append((src_gid, post_gid, weight_changes))
                                print(f"     ✅ Weight recording set up")

                                connection_count += 1

                            except Exception as stdp_conn_error:
                                print(f"     ❌ STDP connection error: {stdp_conn_error}")
                                logging.error(f"STDP connection error {src_gid}->{post_gid}: {stdp_conn_error}")

                        else:
                            print(f"     🔗 Creating regular connection...")
                            try:
                                if inhtype:
                                    if not hasattr(target, 'synlistinh'):
                                        print(f"     ❌ Target {target_type} has no synlistinh")
                                        continue
                                    syn = target.synlistinh[i]
                                    print(f"     ✅ Got inhibitory synapse")
                                else:
                                    if not hasattr(target, 'synlistex'):
                                        print(f"     ❌ Target {target_type} has no synlistex")
                                        continue
                                    syn = target.synlistex[i]
                                    print(f"     ✅ Got excitatory synapse")

                                nc = pc.gid_connect(src_gid, syn)
                                nc.weight[0] = random.gauss(weight, weight / 5)
                                nc.threshold = threshold
                                nc.delay = random.gauss(delay, delay / 5)
                                self.netcons.append(nc)
                                print(f"     ✅ Regular NetCon created")
                                connection_count += 1

                            except Exception as reg_conn_error:
                                print(f"     ❌ Regular connection error: {reg_conn_error}")
                                logging.error(f"Regular connection error {src_gid}->{post_gid}: {reg_conn_error}")

                except Exception as target_error:
                    print(f"   ❌ Error getting target for GID {post_gid}: {target_error}")
                    logging.error(f"Target error {post_gid}: {target_error}")

            else:
                print(f"   ⏭️ GID {post_gid} not on this rank")

        print(f"🏁 connectcells finished: {connection_count} connections created")
        logging.info(f"connectcells end: {connection_count} connections created")

    def genconnect(self, gen_gid, afferents_gids, weight, delay, inhtype=False, N=50):
        nsyn = random.randint(N - 5, N)
        for i in afferents_gids:
            if pc.gid_exists(i):
                for j in range(nsyn):
                    target = pc.gid2cell(i)
                    if inhtype:
                        syn = target.synlistinh[j]
                    else:
                        syn = target.synlistex[j]
                    nc = pc.gid_connect(gen_gid, syn)
                    nc.threshold = self.threshold
                    nc.delay = random.gauss(delay, delay / 5)
                    nc.weight[0] = random.gauss(weight, weight / 6)
                    self.stimnclist.append(nc)

    def motodiams(self, number):
        nrn_number = number
        standby_percent = 70
        active_percent = 100 - standby_percent

        standby_size = int(nrn_number * standby_percent / 100)
        active_size = nrn_number - standby_size

        loc_active, scale_active = 27, 3
        loc_stanby, scale_stanby = 44, 4

        x2 = np.concatenate([np.random.normal(loc=loc_active, scale=scale_active, size=active_size),
                             np.random.normal(loc=loc_stanby, scale=scale_stanby, size=standby_size)])

        return x2

    def addIbgener(self, mn: list, mn2: list, start, weight=1.0):
        '''
        Creates self.Ib generators and returns generator gid
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
        print(f"🎯 [rank {rank}] Creating IbGenerator: start={start}, weight={weight}")
        logging.info(f"IbGenerator creation start: start={start}, weight={weight}")

        gid = self.n_gid
        print(f"   Assigned GID: {gid}")

        # Only create on rank 0 to avoid conflicts
        if rank == 0:
            try:
                # Check if we have motor neurons available
                if not mn or not mn2:
                    print(f"   ❌ Empty motor neuron lists")
                    logging.error("Empty motor neuron lists for IbGenerator")
                    self.n_gid += 1
                    return gid

                # Get random motor neurons from the lists
                moto_gid = random.choice(mn)
                moto2_gid = random.choice(mn2)
                print(f"   Selected motor neurons: {moto_gid}, {moto2_gid}")

                # Check if motor neurons exist on any rank
                moto_rank = None
                moto2_rank = None
                for r in range(nhost):
                    if pc.gid_exists(moto_gid):
                        moto_rank = r
                    if pc.gid_exists(moto2_gid):
                        moto2_rank = r

                if moto_rank is None or moto2_rank is None:
                    print(f"   ⚠️ Motor neurons not found locally, creating simplified IbGenerator")
                    # Create simplified generator without muscle connections
                    stim = h.IbGenerator()
                    stim.start = start
                    stim.interval = int(1000 / bs_fr)
                    stim.number = int(one_step_time / stim.interval)
                    
                    self.stims.append(stim)
                    pc.set_gid2node(gid, rank)
                    ncstim = h.NetCon(stim, None)
                    ncstim.weight[0] = weight
                    self.netcons.append(ncstim)
                    pc.cell(gid, ncstim)
                    
                else:
                    # Create full IbGenerator with muscle connections
                    moto = pc.gid2cell(moto_gid)
                    moto2 = pc.gid2cell(moto2_gid)
                    print(f"   Got motor neuron objects: {type(moto).__name__}, {type(moto2).__name__}")

                    stim = h.IbGenerator()
                    print(f"   ✅ IbGenerator created: {type(stim).__name__}")
                    logging.info(f"IbGenerator object created successfully")

                    stim.start = start
                    stim.interval = int(1000 / bs_fr)
                    stim.number = int(one_step_time / stim.interval)
                    print(f"   Parameters set: start={stim.start}, interval={stim.interval}, number={stim.number}")

                    self.stims.append(stim)

                    # Set pointers if motor neurons have muscle_unit
                    try:
                        if hasattr(moto, 'muscle_unit'):
                            h.setpointer(moto.muscle_unit(0.5)._ref_F_fHill, 'fhill', stim)
                            print(f"   ✅ First pointer set (fhill)")
                    except Exception as ptr1_error:
                        print(f"   ⚠️ First pointer warning: {ptr1_error}")

                    try:
                        if hasattr(moto2, 'muscle_unit'):
                            h.setpointer(moto2.muscle_unit(0.5)._ref_F_fHill, 'fhill2', stim)
                            print(f"   ✅ Second pointer set (fhill2)")
                    except Exception as ptr2_error:
                        print(f"   ⚠️ Second pointer warning: {ptr2_error}")

                    pc.set_gid2node(gid, rank)
                    ncstim = h.NetCon(stim, None)
                    ncstim.weight[0] = weight
                    self.netcons.append(ncstim)
                    pc.cell(gid, ncstim)

                self.log_gid_by_lookup(gid, "Ib")
                print(f"🎯 IbGenerator creation completed successfully: GID={gid}")
                logging.info(f"IbGenerator creation completed: GID={gid}")

            except Exception as ib_error:
                print(f"❌ IbGenerator creation failed: {ib_error}")
                logging.error(f"IbGenerator creation failed: {ib_error}")
                # Still increment GID to maintain consistency
                pass

        else:
            # Other ranks just register the GID assignment
            pc.set_gid2node(gid, 0)

        self.gener_Ibgids.append(gid)
        self.n_gid += 1
        return gid

    def addgener(self, start, freq, flg_interval, interval, cv=False, r=True):
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
        gid = self.n_gid
        # Only create generator on rank 0 to avoid duplicates
        if rank == 0:
            stim = h.NetStim()
            # stim.number = nums
            if r:
                stim.start = random.uniform(start - 3, start + 3)
                stim.noise = 0.05
            else:
                stim.start = start
            if cv:
                stim.interval = int(1000 / freq)
                stim.number = int(int(one_step_time / stim.interval) / CV_number) + 0.45 * int(int(one_step_time / stim.interval) / CV_number)
            else:
                stim.interval = int(1000 / freq)
                stim.number = int(one_step_time / stim.interval) - 7
            self.stims.append(stim)
            pc.set_gid2node(gid, rank)
            ncstim = h.NetCon(stim, None)
            self.netcons.append(ncstim)
            pc.cell(gid, ncstim)
            self.log_gid_by_lookup(gid, "gen")
        else:
            # Other ranks just need to know the GID is assigned to rank 0
            pc.set_gid2node(gid, 0)
            
        self.gener_gids.append(gid)
        self.n_gid += 1

        return gid

    def connectinsidenucleus(self, nucleus):
        self.connectcells(nucleus, nucleus, 0.25, 0.5)

    def add_bs_geners(self, freq, spikes_per_step):
        E_bs_gids = []
        F_bs_gids = []
        for step in range(step_number):
            F_bs_gids.append(self.addgener(int(one_step_time * (2 * step + 1)), freq, False, 1))
            E_bs_gids.append(self.addgener(int(one_step_time * 2 * step) + 10, freq, False, 1))
        return E_bs_gids, F_bs_gids

    def add_ib_geners(self):
        E_ib_gids = []
        F_ib_gids = []
        for step in range(step_number):
            # E_ib_gids.append(self.addIbgener(self.muscle_E, self.muscle_F, 10, weight=5))
            # F_ib_gids.append(self.addIbgener(self.muscle_F, self.muscle_E, one_step_time, weight=8))
            E_ib_gids.append(self.addIbgener(self.muscle_E, self.muscle_F, 15 + one_step_time * 2 * step, weight=0.1))
            F_ib_gids.append(self.addIbgener(self.muscle_F, self.muscle_E, 15 + one_step_time * (1 + 2 * step), weight=0.1))
        return E_ib_gids, F_ib_gids

    def log_gid_by_lookup(self, gid: int, name):
        if not pc.gid_exists(gid):
            print(f"[rank {rank}] GID {gid} not assigned to this process.")
            return

        obj = pc.gid2cell(gid)
        typename = type(obj).__name__
        if name:
            print(f"[rank {rank}] Added GID {gid} (type: {typename}) - name: {name}")
        else: print(f"[rank {rank}] Added GID {gid} (type: {typename})")


# def draw(weight_changes, time_t):
#     i = 0
#     # Convert weight_changes to a NumPy array
#     for w, t in zip(weight_changes, time_t):
#         weight_changes_array = np.array(w.as_numpy())
#         time_array = np.array(t.as_numpy())
#
#         figur = figure(x_axis_label='time (ms)', y_axis_label='weight')
#         output_file(f'results_stdp/change_res_{i}.html')
#         i += 1
#         figur.line(time_array, weight_changes_array, line_width=2)
#         show(figur)


def spike_record(pool, extra=False, location='soma'):
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
    v_vec = []

    for i in pool:
        cell = pc.gid2cell(i)
        vec = h.Vector(np.zeros(int(time_sim / 0.025 + 1), dtype=np.float32))
        if extra:
            vec.record(cell.soma(0.5)._ref_vext[0])
        else:
            if location == 'axon':
                # Изменение на запись из первого узла аксона (здесь будут нормальные спайки)
                vec.record(cell.node[0](1.0)._ref_v)
            elif location == 'muscle':
                vec.record(cell.muscle_unit(0.5)._ref_v)
            elif location == 'am':
                vec.record(cell.muscle_unit(0.5)._ref_AM_CaSP)
            else:
                # Запись из сомы (как было раньше)
                vec.record(cell.soma(0.5)._ref_v)
        v_vec.append(vec)
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
        cell = pc.gid2cell(gid)
        vec = h.Vector(np.zeros(int(time_sim / 0.025 + 1), dtype=np.float32))
        vec.record(getattr(cell, attr))
        vecs.append(vec)
    return vecs


def spikeout(pool, name, version, v_vec):
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
    vec = h.Vector()

    # Создаем директорию для индивидуальных записей, если её нет
    if rank == 0:
        individual_dir = f'./{file_name}/{name}_individual'
        if not os.path.exists(individual_dir):
            os.makedirs(individual_dir)

    for i in range(nhost):
        if i == rank:
            # Подготовка средних значений (как в оригинале)
            outavg = []
            for j in range(len(pool)):
                outavg.append(list(v_vec[j]))

                # Сохранение индивидуальных значений для каждого нейрона
                if rank == 0:  # Сохраняем только на узле 0 для простоты
                    individual_file = f'{individual_dir}/neuron_{pool[j]}_sp_{speed}_CVs_{CV_number}_bs_{bs_fr}.hdf5'
                    with hdf5.File(individual_file, 'w') as indiv_file:
                        neuron_data = list(v_vec[j])
                        for step in range(step_number):
                            sl = slice((int(1000 / bs_fr) * 40 + step * one_step_time * 40),
                                       (int(1000 / bs_fr) * 40 + (step + 1) * one_step_time * 40))
                            indiv_file.create_dataset(f'#0_step_{step}', data=np.array(neuron_data)[sl],
                                                      compression="gzip")

            # Продолжение обработки средних значений
            outavg = np.mean(np.array(outavg), axis=0, dtype=np.float32)
            vec = vec.from_python(outavg)
        pc.barrier()

    pc.barrier()
    result = pc.py_gather(vec, 0)

    # Сохранение средних значений (как в оригинале)
    if rank == 0:
        logging.info("start recording " + name)
        result = np.mean(np.array(result), axis=0, dtype=np.float32)
        with hdf5.File(f'./{file_name}/{name}_sp_{speed}_CVs_{CV_number}_bs_{bs_fr}.hdf5', 'w') as file:
            for i in range(step_number):
                sl = slice((int(1000 / bs_fr) * 40 + i * one_step_time * 40),
                           (int(1000 / bs_fr) * 40 + (i + 1) * one_step_time * 40))
                file.create_dataset('#0_step_{}'.format(i), data=np.array(result)[sl], compression="gzip")
        logging.info("done recording average")
        logging.info("done recording individual neurons")
    else:
        logging.info(rank)


def prun(speed, step_number):
    print(f"🚀 [rank {rank}] Starting simulation: speed={speed}, steps={step_number}")
    logging.info(f"prun start: speed={speed}, steps={step_number}, time_sim={time_sim}")

    try:
        print(f"   Setting timeout...")
        pc.timeout(0)

        print(f"   Creating time vector...")
        t = h.Vector().record(h._ref_t)

        print(f"   Setting tstop={time_sim}")
        h.tstop = time_sim  # Убедитесь что h.tstop установлен

        # Добавьте проверку на macOS
        import platform
        if platform.system() == "Darwin":  # macOS
            print(f"   macOS detected - using alternative approach")
            h.dt = 0.025  # Установите явно шаг времени
            h.cvode_active(0)  # Отключите переменный шаг
            pc.set_maxstep(1)  # Уменьшите maxstep для macOS
        else:
            pc.set_maxstep(10)

        print(f"   Initializing with h.finitialize(-65)...")
        h.finitialize(-65)
        print(f"   ✅ finitialize completed")

        print(f"   Starting main simulation...")
        if platform.system() == "Darwin":
            # Альтернативный подход для macOS
            while h.t < time_sim:
                h.fadvance()
                if int(h.t) % 500 == 0:  # Логирование каждые 500ms
                    print(f"   Progress: {h.t:.1f}/{time_sim} ms")
        else:
            pc.psolve(time_sim)

        print(f"🏁 Simulation completed successfully")
        return t

    except Exception as sim_error:
        print(f"❌ Simulation error: {sim_error}")
        logging.error(f"Simulation error: {sim_error}")
        raise sim_error


def finish():
    ''' proper exit '''
    pc.runworker()
    pc.done()
    # print("hi after finish")
    h.quit()


# if __name__ == '__main__':
#     # h.dt = 0.1
#     # h.tstop = 50
#     # cpg = CPG()
#     #
#     # h.run()
#     #
#     # w = cpg.weight_changes_vectors
#     # t = cpg.time_t_vectors
#     #
#     # draw(w, t)

def safe_filename(name: str) -> str:
    """Преобразует строку в безопасное имя файла."""
    return re.sub(r'[^\w\-_.]', '_', name)


def check_mechanisms():
    """Проверка доступности необходимых механизмов"""
    try:
        # Проверяем STDP
        dummy = h.Section()
        stdp_test = h.STDP(0, dummy)
        print("✅ STDP mechanism available")
        del stdp_test, dummy

        # Проверяем IbGenerator
        ib_test = h.IbGenerator()
        print("✅ IbGenerator mechanism available")
        del ib_test

        return True
    except Exception as e:
        print(f"❌ Mechanism check failed: {e}")
        return False

if __name__ == '__main__':
    '''
    cpg_ex: cpg
        topology of central pattern generation + reflex arc
    '''
    print(f"🎬 [rank {rank}] MAIN EXECUTION START")
    print(f"   Rank {rank} of {nhost} processes")
    print(f"   Parameters: N={N}, speed={speed}, bs_fr={bs_fr}, versions={versions}")
    print(f"   Step number: {step_number}, one_step_time: {one_step_time}")
    print(f"   Total simulation time: {time_sim} ms")
    logging.info("=== MAIN EXECUTION START ===")
    logging.info(f"Rank {rank}/{nhost}, N={N}, speed={speed}, versions={versions}")

    k_nrns = 0
    k_name = 1

    if rank == 0 and not os.path.isdir(file_name):
        os.mkdir(file_name)
        print(f"   ✅ Created directory: {file_name}")

    # Synchronize all ranks before proceeding
    pc.barrier()

    for i in range(versions):
        print(f"🔄 [rank {rank}] VERSION {i + 1}/{versions} START")
        logging.info(f"=== VERSION {i + 1} START ===")

        try:
            print(f"   Creating CPG network...")
            cpg_ex = CPG(speed, bs_fr, 100, step_number, N)
            print(f"   ✅ CPG network created successfully")
            logging.info("CPG created successfully")

            # Synchronize after network creation
            pc.barrier()

            print(f"   Setting up voltage recorders...")
            motorecorders = []
            motorecorders_mem = []
            musclerecorders = []
            muscle_units_recorders = []
            muscle_am_recorders = []
            force_recorders = []

            print(f"      Setting up motor neuron recorders...")
            for group in cpg_ex.motogroups:
                motorecorders_mem.append(spike_record(group[k_nrns]))
            print(f"      ✅ {len(motorecorders_mem)} motor recorder groups")

            print(f"      Setting up afferent recorders...")
            affrecorders = []
            for group in cpg_ex.affgroups:
                affrecorders.append(spike_record(group[k_nrns]))
            print(f"      ✅ {len(affrecorders)} afferent recorder groups")

            print(f"      Setting up interneuron recorders...")
            recorders = []
            for group in cpg_ex.intgroups:
                recorders.append(spike_record(group[k_nrns]))
            print(f"      ✅ {len(recorders)} interneuron recorder groups")

            print(f"      Setting up muscle recorders...")
            for group in cpg_ex.musclegroups:
                musclerecorders.append(spike_record(group[k_nrns]))
                force_recorders.append(force_record(group[k_nrns]))
                muscle_units_recorders.append(spike_record(group[k_nrns], location='muscle'))
                muscle_am_recorders.append(spike_record(group[k_nrns], location='am'))
            print(f"      ✅ {len(musclerecorders)} muscle recorder groups")

            print(f"      Setting up Ib generator recorders...")
            vel_vecs_recorders = velocity_record(cpg_ex.gener_Ibgids, attr='_ref_vel')
            v0_vecs_recorders = velocity_record(cpg_ex.gener_Ibgids, attr='_ref_v0')
            print(f"      ✅ {len(vel_vecs_recorders)} Ib generator recorders")

            print(f"   ✅ All recorders set up successfully")
            logging.info("Added recorders")

            # Synchronize before simulation
            pc.barrier()

            print(f"   🚀 Starting simulation...")
            print("- " * 20)
            t = prun(speed, step_number)
            print("- " * 20)
            print(f"   ✅ Simulation completed")

            logging.info("Simulation done")

            print(f"   💾 Saving results...")

            if rank == 0:
                print(f"      Saving time data...")
                with open(f'./{file_name}/time.txt', 'w') as time_file:
                    for time in t:
                        time_file.write(str(time) + "\n")
                print(f"      ✅ Time data saved")

            print(f"      Saving spike data...")
            for group, recorder in zip(cpg_ex.musclegroups, musclerecorders):
                spikeout(group[k_nrns], group[k_name], i, recorder)
            for group, recorder in zip(cpg_ex.motogroups, motorecorders_mem):
                spikeout(group[k_nrns], 'mem_{}'.format(group[k_name]), i, recorder)
            for group, recorder in zip(cpg_ex.affgroups, affrecorders):
                spikeout(group[k_nrns], group[k_name], i, recorder)
            for group, recorder in zip(cpg_ex.intgroups, recorders):
                spikeout(group[k_nrns], group[k_name], i, recorder)
            for group, recorder in zip(cpg_ex.musclegroups, force_recorders):
                spikeout(group[k_nrns], 'force_{}'.format(group[k_name]), i, recorder)
            for group, recorder in zip(cpg_ex.musclegroups, muscle_units_recorders):
                spikeout(group[k_nrns], 'units_{}'.format(group[k_name]), i, recorder)
            for group, recorder in zip(cpg_ex.musclegroups, muscle_am_recorders):
                spikeout(group[k_nrns], 'am_{}'.format(group[k_name]), i, recorder)

            spikeout(cpg_ex.gener_Ibgids, 'vel', i, vel_vecs_recorders)
            spikeout(cpg_ex.gener_Ibgids, 'v0', i, v0_vecs_recorders)
            print(f"      ✅ Spike data saved")

            if rank == 0:
                print(f"      Saving STDP weight changes...")
                stdp_dir = f'./{file_name}/stdp_1'
                if not os.path.exists(stdp_dir):
                    os.makedirs(stdp_dir)
                    print(f"      ✅ Created STDP directory: {stdp_dir}")

                stdp_count = 0
                for src_gid, post_gid, weight_vec in cpg_ex.weight_changes_vectors:
                    try:
                        src_obj = pc.gid2cell(src_gid) if pc.gid_exists(src_gid) else None
                        post_obj = pc.gid2cell(post_gid) if pc.gid_exists(post_gid) else None

                        src_type = type(src_obj).__name__ if src_obj is not None else "None"
                        post_type = type(post_obj).__name__ if post_obj is not None else "None"

                        # Сформировать безопасное имя файла
                        safe_name = safe_filename(f'{src_type}_{src_gid}_to_{post_type}_{post_gid}.hdf5')
                        fname = f'{stdp_dir}/{safe_name}'

                        with hdf5.File(fname, 'w') as file:
                            file.create_dataset(f'#0_step_{i}', data=np.array(weight_vec), compression="gzip")
                        stdp_count += 1

                    except Exception as e:
                        print(f"      ⚠️ Error saving STDP weight {src_gid} → {post_gid}: {e}")

                print(f"      ✅ Saved {stdp_count} STDP weight change files")

            print(f"   ✅ All results saved successfully")
            logging.info("Results recorded")

            print(f"🏁 [rank {rank}] VERSION {i + 1} COMPLETED SUCCESSFULLY")

        except Exception as version_error:
            print(f"❌ [rank {rank}] VERSION {i + 1} FAILED: {version_error}")
            logging.error(f"Version {i + 1} error: {version_error}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
            break

    print(f"🏁 [rank {rank}] MAIN EXECUTION FINISHED")
    logging.info("=== MAIN EXECUTION END ===")
    finish()