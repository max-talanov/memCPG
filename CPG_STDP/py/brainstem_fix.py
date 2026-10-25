import random
import numpy as np
import logging
import h5py as hdf5
import os
from neuron import h
from bioaffrat import bioaffrat
from interneuron import interneuron
from motoneuron import motoneuron
from muscle import muscle

logging.basicConfig(filename='logs_new_new_3.log',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)
logging.info("let's get it started")

h.load_file("stdgui.hoc")
h.load_file('nrngui.hoc')
h.load_file('stdrun.hoc')

# paralleling NEURON stuff

pc = h.ParallelContext()
rank = int(pc.id())
nhost = int(pc.nhost())

file_name = 'res_alina_new_new'

N = 7  # 50
speed = 100
bs_fr = 70 #50 #150 #100  # 40 # frequency of brainstem inputs
versions = 1
CV_number = 6
k = 0.017  # CV weights multiplier to take into account air and toe stepping
CV_0_len = 12  # 125 # Duration of the CV generator with no sensory inputs
extra_layers = 0  # 1 + layers

step_number = 6

one_step_time = int((6 * speed + CV_0_len) / (int(1000 / bs_fr))) * (int(1000 / bs_fr))
time_sim = one_step_time * step_number + 20

'''
network topology https://github.com/max-talanov/bypass/blob/main/figs/CPG_feedback_loops.png
'''


class CPG:

    def __init__(self, speed, bs_fr, inh_p, step_number, n):
        self.threshold = -20
        self.delay = 1
        self.nAff = 7  # 12  # 12
        self.nInt = 7 # 5
        self.nMn = 7  # 21  # 21
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
        self.recorded_spikes = []
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
        #
        for layer in range(CV_number):
            '''cut and muscle feedback'''
            # self.dict_CV_1 = {layer: 'CV{}_1'.format(layer + 1)}
            self.dict_RG_E = {layer: 'RG{}_E'.format(layer + 1)}
            self.dict_RG_F = {layer: 'RG{}_F'.format(layer + 1)}
            # self.dict_V3F = {layer: 'V3{}_F'.format(layer + 1)}
            # self.dict_C = {layer: 'C{}'.format(layer + 1)}

        for layer in range(CV_number):
            '''Cutaneous pools'''
            # self.dict_CV_1[layer] = self.addpool(self.ncell, "CV" + str(layer + 1) + "_1", "aff")
            '''Rhythm generator pools'''
            self.dict_RG_E[layer] = self.addpool(self.ncell, "RG" + str(layer + 1) + "_E", "int")
            self.dict_RG_F[layer] = self.addpool(self.ncell, "RG" + str(layer + 1) + "_F", "int")
            self.RG_E.append(self.dict_RG_E[layer])
            self.RG_F.append(self.dict_RG_F[layer])
            # self.CV.append(self.dict_CV_1[layer])
        #
        '''RG'''
        self.RG_E = sum(self.RG_E, [])
        self.InE = self.addpool(self.nInt, "InE", "int")
        self.RG_F = sum(self.RG_F, [])
        self.InF = self.addpool(self.nInt, "InF", "bursting")

        # self.CV = sum(self.CV, [])
        #
        # '''sensory and muscle afferents and brainstem and V3F'''
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
        #
        '''reflex arc'''
        # self.Ib_E = self.addpool(self.nInt, "Ib_E", "int")
        self.R_E = self.addpool(self.nInt, "R_E", "int")  # Renshaw cells
        # self.Ib_F = self.addpool(self.nInt, "Ib_F", "bursting")
        self.R_F = self.addpool(self.nInt, "R_F", "bursting")  # Renshaw cells

        logging.info("done addpool")
        '''BS'''
        # periodic stimulation
        self.E_bs_gids, self.F_bs_gids = self.add_bs_geners(bs_fr, 0.5, 0.5)

        # ''' BS '''
        # for E_bs_gid in self.E_bs_gids:
        #     self.genconnect(E_bs_gid, self.BS_aff_E, 2, 1)
        #
        # for F_bs_gid in self.F_bs_gids:
        #     self.genconnect(F_bs_gid, self.BS_aff_F, 2, 1)

        ''' BS '''
        for E_bs_gid in self.E_bs_gids:
            for layer in range(CV_number):
                self.genconnect(E_bs_gid, self.dict_RG_E[layer], 0.002, 0.5)

        for F_bs_gid in self.F_bs_gids:
            for layer in range(CV_number):
                self.genconnect(F_bs_gid, self.dict_RG_F[layer], 0.002, 0.5)


        # for layer in range(CV_number):
        #     self.connectcells(self.BS_aff_F, self.dict_RG_F[layer], 4, 1, stdptype=False)
        #     self.connectcells(self.BS_aff_E, self.dict_RG_E[layer], 4, 1, stdptype=False)
        # #
        for layer in range(CV_number):
            '''Internal to RG topology'''
            self.connectinsidenucleus(self.dict_RG_F[layer])
            self.connectinsidenucleus(self.dict_RG_E[layer])

            # '''RG2Motor'''
            self.connectcells(self.dict_RG_E[layer], self.mns_E, 0.5, 1)
            self.connectcells(self.dict_RG_F[layer], self.mns_F, 0.5, 1)

            self.connectcells(self.dict_RG_E[layer], self.InE, 0.01, 1)
            self.connectcells(self.dict_RG_F[layer], self.InF, 0.01, 1)
        #
        # # '''motor2muscles'''
        self.connectcells(self.mns_E, self.muscle_E, 0.1, 1, inhtype=False)
        self.connectcells(self.mns_F, self.muscle_F, 0.1, 1, inhtype=False)
        #
        # # '''muscle afferents generators'''
        #
        self.E_ib_gids, self.F_ib_gids = self.add_ib_geners(self.muscle_E, self.muscle_F)
        # ''' BS '''
        for E_ib_gid in self.E_ib_gids:
            self.genconnect(E_ib_gid, self.Ib_aff_E, 0.05, 3)
        #
        for F_ib_gid in self.F_ib_gids:
            self.genconnect(F_ib_gid, self.Ib_aff_F, 0.05, 3)
        # self.Ibgener_E = self.addIbgener(self.muscle_E, self.muscle_F, 15, weight=20)
        # self.Ibgener_F = self.addIbgener(self.muscle_F, self.muscle_E, one_step_time + 15, weight=30)
        #
        # '''Create connectcells'''
        # # self.genconnect(self.Ibgener_E, self.Ib_aff_E, 0.5, 1, False, 20)
        # # self.genconnect(self.Ibgener_F, self.Ib_aff_F, 0.5, 1, False, 30)
        #
        # self.connectcells(self.muscle_E, self.Ib_aff_E, 3.5, 1, 10, False)
        # self.connectcells(self.muscle_F, self.Ib_aff_F, 3.5, 1, 10, False)
        #
        # '''cutaneous inputs'''
        # cfr = 200
        # c_int = 1000 / cfr
        #
        # # '''cutaneous inputs generators'''
        # # for layer in range(CV_number):
        # #     self.dict_C[layer] = []
        # #     for i in range(step_number):
        # #         self.dict_C[layer].append(
        # #             self.addgener(
        # #                 10 + speed * layer + i * (speed * CV_number + CV_0_len + one_step_time) + 7 - layer * 12,
        # #                 int(random.gauss(cfr, cfr / 10)), True,
        # #                 int((one_step_time / CV_number) * 0.15), cv=True))
        #
        # '''Generators'''
        # '''TODO: need it?'''
        # # for i in range(step_number):
        # #     self.C_0.append(
        # #         self.addgener(25 + speed * 6 + i * (speed * 6 + CV_0_len), cfr, False, 1, cv=True, r=False))
        # #
        # # # '''TODO: need it?'''
        # # for layer in range(CV_number):
        # #     self.C_1.append(self.dict_CV_1[layer])
        # # self.C_1 = sum(self.C_1, [])
        #
        # # self.connectcells(self.BS_aff_F, self.V3F, 1.5, 3)
        #
        # # '''cutaneous inputs'''
        # # for layer in range(CV_number):
        # #     self.genconnect(self.dict_C[layer], self.dict_CV_1[layer], 3 * k * speed, 2)  # 0.15
        # #     self.connectcells(self.dict_CV_1[layer], self.dict_RG_E[layer], 2 * k * speed, 3)  # 0.0035
        #
        # # '''Ib2motor'''
        # self.connectcells(self.Ib_aff_E, self.mns_E, 1, 2)
        # self.connectcells(self.Ib_aff_F, self.mns_F, 1, 2)
        # #
        # for layer in range(CV_number):
        #     '''STDP synapse'''
        #     self.connectcells(self.Ib_aff_E, self.dict_RG_E[layer], weight=1, delay=3, stdptype=True)
        #     self.connectcells(self.Ib_aff_F, self.dict_RG_F[layer], weight=1, delay=3, stdptype=True)
        #
        # #     # self.connectcells(self.dict_RG_F[layer], self.V3F, 1.5, 3)
        # # #
        '''Ib2RG, RG2Motor'''
        self.connectcells(self.InE, self.RG_F, 0.009, 1, inhtype=True)
        self.connectcells(self.InF, self.RG_E, 0.009, 1, inhtype=True)

        # self.connectcells(self.Ib_aff_E, self.Ib_E, 1, 1, inhtype=False)
        # self.connectcells(self.Ib_aff_F, self.Ib_F, 1, 1, inhtype=False)

        self.connectcells(self.mns_E, self.R_E, 0.005, 1, inhtype=False)
        self.connectcells(self.mns_F, self.R_F, 0.005, 1, inhtype=False)

        self.connectcells(self.R_E, self.mns_E, 0.009, 1, inhtype=True)
        self.connectcells(self.R_F, self.mns_F, 0.009, 1, inhtype=True)

        # self.connectcells(self.R_E, self.Ib_E, 0.7, 1, inhtype=True)
        # self.connectcells(self.R_F, self.Ib_F, 0.7, 1, inhtype=True)

        # self.connectcells(self.Ib_E, self.mns_F, 0.8, 1, inhtype=True)
        # self.connectcells(self.Ib_F, self.mns_E, 0.8, 1, inhtype=True)

        logging.info("done connectcells")

    def addpool(self, num, name, neurontype="int") -> list:
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
        gid = self.n_gid

        delaytype = False
        if neurontype.lower() == "delay":
            delaytype = True

        if neurontype.lower() == "moto":
            diams = self.motodiams(num)
        for i in range(rank, num, nhost):
            if neurontype.lower() == "moto":
                cell = motoneuron(diams[i])
                self.motos.append(cell)
            elif neurontype.lower() == "aff":
                cell = bioaffrat()
                if "_F" in name:
                    cell.neuron_type = "flexor"
                else: cell.neuron_type = "extensor"
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

            pc.set_gid2node(gid, rank)
            nc = cell.connect2target(None)
            nc.threshold = self.threshold
            pc.cell(gid, nc)
            pc.threshold(gid, nc.threshold)
            self.netcons.append(nc)
            gids.append(gid)
            gid += 1

        # Groups
        if neurontype.lower() == "muscle":
            self.musclegroups.append((gids, name))
        elif neurontype.lower() == "moto":
            self.motogroups.append((gids, name))
        elif neurontype.lower() == "aff":
            self.affgroups.append((gids, name))
        else:
            self.intgroups.append((gids, name))

        self.n_gid = gid

        return gids

    def connectcells(self, pre_cells, post_cells, weight=1.0, delay=1, threshold=-20, inhtype=False,
                     stdptype=False, N=80):
        nsyn = random.randint(N, N + 15)
        for post_gid in post_cells:
            if pc.gid_exists(post_gid):
                for i in range(nsyn):
                    src_gid = random.randint(pre_cells[0], pre_cells[-1])
                    target = pc.gid2cell(post_gid)
                    if stdptype:
                        syn = target.synlistexstdp[i]
                        nc = pc.gid_connect(src_gid, syn)
                        nc.delay = delay
                        nc.weight[0] = weight
                        nc.threshold = threshold
                        pc.threshold(src_gid, threshold)
                        self.netcons.append(nc)

                        dummy = h.Section()  # Create a dummy section to put the point processes in
                        stdpmech = h.STDP(0, dummy)
                        self.stdpmechs.append(stdpmech)
                        presyn = pc.gid_connect(src_gid,
                                                stdpmech)  # threshold, delay, 1)  # Feed presynaptic spikes to the STDP mechanism -- must have weight >0
                        presyn.delay = delay
                        presyn.weight[0] = 2
                        presyn.threshold = threshold
                        self.presyns.append(presyn)
                        pstsyn = pc.gid_connect(post_gid,
                                                stdpmech)  # threshold, delay, -1)  # Feed postsynaptic spikes to the STDP mechanism -- must have weight <0
                        pstsyn.delay = delay
                        pstsyn.weight[0] = -2
                        pstsyn.threshold = threshold
                        self.postsyns.append(pstsyn)
                        pc.threshold(post_gid, threshold)
                        h.setpointer(nc._ref_weight[0], 'synweight', stdpmech)

                        weight_changes = h.Vector()
                        weight_changes.record(stdpmech._ref_synweight)
                        self.weight_changes_vectors.append((src_gid, post_gid, weight_changes))
                    else:
                        if inhtype:
                            syn = target.synlistinh[i]
                        else:
                            syn = target.synlistex[i]
                        nc = pc.gid_connect(src_gid, syn)
                        nc.weight[0] = random.gauss(weight, weight / 5)
                        nc.threshold = threshold
                        nc.delay = random.gauss(delay, delay / 5)
                        pc.threshold(src_gid, nc.threshold)
                        self.netcons.append(nc)

    def genconnect(self, gen_gid, afferents_gids, weight, delay, inhtype=False, N=50):
        nsyn = random.randint(N - 5, N)
        for i in afferents_gids:
            if pc.gid_exists(i):
                for j in range(nsyn):
                    target = pc.gid2cell(i)
                    if inhtype:
                        syn = target.synlistinh[j]
                    else:
                        # syn = target.synlistees[j]
                        syn = target.synlistex[j]
                    nc = pc.gid_connect(gen_gid, syn)
                    nc.threshold = self.threshold
                    nc.delay = random.gauss(delay, delay / 5)
                    nc.weight[0] = weight #random.gauss(weight, weight / 5)
                    pc.threshold(gen_gid, nc.threshold)
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

    def addIbgener(self, mn: list, mn2: list, start: int, weight: float = 1.0) -> int:
        """
        Creates an Ib generator and returns its gid.

        Parameters
        ----------
        mn : list
            of motor neuron gids for the agonist muscle.
        mn2 : list
            of motor neuron gids for the antagonist muscle.
        start : int
            Start time for the generator (in ms).
        weight : float, optional
            Connection weight, default is 1.0.

        Returns
        -------
        int
            The gid of the created generator.
        """

        # Choose a random motor neuron cell from the provided lists using random.choice.
        moto = pc.gid2cell(random.choice(mn))
        moto2 = pc.gid2cell(random.choice(mn2))

        # Create the Ib generator with an initial value of 0.5.
        stim = h.IbGenerator(0.5)
        stim.start = start

        # Set generator parameters:
        freq = 70 #70 #100  # frequency in Hz
        stim.interval = int(1000 / freq)  # interval between stimuli in ms
        # 'one_step_time' is assumed to be a global variable representing the simulation step duration (in ms).
        stim.number = int(one_step_time / stim.interval) - 5

        # Set pointers for the muscle units associated with the motor neurons.
        muscle_unit1 = moto.muscle_unit(0.5)
        muscle_unit2 = moto2.muscle_unit(0.5)
        h.setpointer(muscle_unit1._ref_F_fHill, 'fhill', stim)
        h.setpointer(muscle_unit2._ref_F_fHill, 'fhill2', stim)

        # Append the generator to the list of stimulations.
        self.stims.append(stim)

        # Assign a unique gid to the generator.
        gid = self.n_gid
        # 'rank' is assumed to be a global variable responsible for distributing gids across nodes.
        pc.set_gid2node(gid, rank)

        # Create a NetCon for the generator and configure its parameters.
        ncstim = h.NetCon(stim, None)
        ncstim.weight[0] = weight
        self.netcons.append(ncstim)
        pc.cell(gid, ncstim)

        # Record the spike times of the generator.
        spike_times = h.Vector()
        ncstim.record(spike_times)
        self.recorded_spikes.append(spike_times)

        # Append the generator gid to the list and increment the gid counter.
        self.gener_gids.append(gid)
        self.n_gid += 1

        return gid

    def addgener(self, start, freq, flg_interval, interval, weight = 0.5, cv=False, r=True):
        '''
        Creates generator and returns generator gid
        Parameters
        ----------
        weight: float
            weight of the connection
        start: int
            generator start up
        freq: int
            generator frequency
        flg_interval: bool
            have interval_CV
        interval: int
            time interval_CV
        Returns
        -------
        gid: int
            generator gid
        '''
        gid = self.n_gid
        stim = h.NetStim()
        # stim.number = nums
        if r:
            stim.start = random.uniform(start - 3, start + 3)
            # stim.noise = 0.01
        else:
            stim.start = start
        if flg_interval:
            stim.interval = interval
            stim.number = 10
        else:
            stim.interval = int(1000 / freq)
            stim.number = int(one_step_time / stim.interval)
        stim.noise = 0.01
        self.stims.append(stim)
        pc.set_gid2node(gid, rank)
        ncstim = h.NetCon(stim, None)
        ncstim.weight[0] = weight
        self.netcons.append(ncstim)
        pc.cell(gid, ncstim)
        self.gener_gids.append(gid)
        self.n_gid += 1

        return gid

    def connectinsidenucleus(self, nucleus):
        self.connectcells(nucleus, nucleus, 3, delay=0.5, threshold=-20, inhtype=False, stdptype=False, N=60)

    def add_bs_geners(self, freq, weight_E, weight_F):
        E_bs_gids = []
        F_bs_gids = []
        for step in range(step_number):
            F_bs_gids.append(self.addgener(int(one_step_time * (2 * step + 1)), freq, False, 1, weight_E))
            E_bs_gids.append(self.addgener(int(one_step_time * 2 * step) + 20, freq, False, 1, weight_F))
        return E_bs_gids, F_bs_gids

    def add_ib_geners(self, mus_E, mus_F):
        E_ib_gids = []
        F_ib_gids = []
        for step in range(step_number):
            E_ib_gids.append(self.addIbgener(mus_E, mus_F, 15 + one_step_time * 2 * step, weight=1))
            F_ib_gids.append(self.addIbgener(mus_F, mus_E, 15 + one_step_time * (1 + 2 * step), weight=2))
        return E_ib_gids, F_ib_gids


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
      location: str
          'axon' or 'soma', where to record spikes
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
            else:
                # Запись из сомы (как было раньше)
                vec.record(cell.soma(0.5)._ref_v)

        v_vec.append(vec)
    return v_vec


def force_record(pool):
    ''' Records force from gids of motor neurons muscle unit
      Parameters
      ----------
      pool: list
        neurons gids
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


def spikeout(pool, name, version, v_vec):
    ''' Reports simulation results
      Parameters
      ----------
      pool: list
        neurons gids
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
    for i in range(nhost):
        if i == rank:
            outavg = []
            for j in range(len(pool)):
                outavg.append(list(v_vec[j]))
            outavg = np.mean(np.array(outavg), axis=0, dtype=np.float32)
            vec = vec.from_python(outavg)
        pc.barrier()
    pc.barrier()
    result = pc.py_gather(vec, 0)
    if rank == 0:
        logging.info("start recording")
        result = np.mean(np.array(result), axis=0, dtype=np.float32)
        with hdf5.File(f'./{file_name}/{name}_sp_{speed}_CVs_{CV_number}_bs_{bs_fr}.hdf5', 'w') as file:
            for i in range(step_number):
                sl = slice((int(1000 / bs_fr) * 40 + i * one_step_time * 40),
                           (int(1000 / bs_fr) * 40 + (i + 1) * one_step_time * 40))
                file.create_dataset('#0_step_{}'.format(i), data=np.array(result)[sl], compression="gzip")
        logging.info("done recording")
    else:
        logging.info(rank)


def spikeout_individual_files(pool, name, version, v_vec):
    ''' Reports individual neuron simulation results into separate files with slices
      Parameters
      ----------
      pool: list
        neurons gids
      name: string
        pool name
      version: int
          test number
      v_vec: list of h.Vector()
          recorded voltage
    '''
    global rank

    if len(pool) > 5:
        pool = pool[:5]
        v_vec = v_vec[:5]

        # Синхронизация всех процессов перед началом работы
    pc.barrier()

    # Создаем для каждого нейрона из списка свой объект h.Vector для локального хранения данных
    neuron_vecs = {}
    for neuron in pool:
        neuron_vecs[neuron] = h.Vector()

    # Каждый процесс по очереди передаёт свои данные для каждого нейрона.
    # Важно, чтобы в каждый момент времени только один процесс занимался передачей данных.
    for proc in range(nhost):
        if proc == rank:
            for idx, neuron in enumerate(pool):
                # Преобразуем данные из v_vec в список и записываем в h.Vector
                data = list(v_vec[idx])
                neuron_vecs[neuron].from_python(data)
        pc.barrier()  # синхронизация между процессами
    pc.barrier()  # дополнительная синхронизация после заполнения данных

    # Собираем данные с каждого процесса для каждого нейрона с помощью py_gather
    gathered_neuron_data = {}
    for neuron in pool:
        gathered_neuron_data[neuron] = pc.py_gather(neuron_vecs[neuron], 0)

    # На процессе с rank == 0 производится запись данных в отдельные файлы
    if rank == 0:
        logging.info("Начало записи индивидуальных файлов для 5 нейронов.")
        for neuron in pool:
            # Объединяем данные, например, усредняя полученные векторы с разных процессов
            vectors_list = [list(vec) for vec in gathered_neuron_data[neuron]]
            combined = np.mean(np.array(vectors_list), axis=0, dtype=np.float32)

            # Формируем имя файла для данного нейрона, например:
            # "./{file_name}/{name}_neuron_{neuron}_v{version}.hdf5"
            file_path = f"./{file_name}/{name}_neuron_{neuron}_v{version}.hdf5"
            with hdf5.File(file_path, 'w') as file:
                # Разбиваем объединённый вектор на срезы и записываем каждый срез в отдельный датасет.
                for i in range(step_number):
                    # Расчёт начала и конца среза. Формулу можно корректировать под ваши данные.
                    start = int(1000 / bs_fr) * 40 + i * one_step_time * 40
                    end = int(1000 / bs_fr) * 40 + (i + 1) * one_step_time * 40
                    dataset_name = f"step_{i}"
                    file.create_dataset(dataset_name, data=combined[start:end], compression="gzip")
        logging.info("Запись индивидуальных файлов для 5 нейронов завершена.")
    else:
        logging.info(f"Процесс с rank {rank} завершил сбор данных для индивидуальных файлов.")


def prun(speed, step_number):
    ''' simulation control
    Parameters
    ----------
    speed: int
      duration of each layer

    Returns
    -------
    t: list of h.Vector()
      recorded time
    '''
    pc.timeout(0)
    t = h.Vector().record(h._ref_t)
    tstop = time_sim
    pc.set_maxstep(10)
    h.stdinit()
    pc.psolve(tstop)
    return t


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


if __name__ == '__main__':
    '''
    cpg_ex: cpg
        topology of central pattern generation + reflex arc
    '''
    k_nrns = 0
    k_name = 1
    if not os.path.exists(file_name):
        os.mkdir(file_name)

    for i in range(versions):
        cpg_ex = CPG(speed, bs_fr, 100, step_number, N)
        logging.info("created")
        motorecorders = []
        motorecorders_mem = []
        musclerecorders = []
        musclerecorders_extra = []
        force_recorders = []
        for group in cpg_ex.motogroups:
            motorecorders.append(spike_record(group[k_nrns], True))

        for group in cpg_ex.motogroups:
            motorecorders_mem.append(spike_record(group[k_nrns]))
        affrecorders = []
        affrecorders_extra = []
        affrecorders_axon = []
        for group in cpg_ex.affgroups:
            affrecorders.append(spike_record(group[k_nrns]))
            # affrecorders_extra.append(spike_record(group[k_nrns], True))
            # affrecorders_axon.append(spike_record(group[k_nrns], False, 'axon'))
        recorders = []
        recorders_extra = []
        for group in cpg_ex.intgroups:
            # recorders_extra.append(spike_record(group[k_nrns], True))
            recorders.append(spike_record(group[k_nrns]))
        for group in cpg_ex.musclegroups:
            musclerecorders.append(spike_record(group[k_nrns]))
            # musclerecorders_extra.append(spike_record(group[k_nrns], True))
            force_recorders.append(force_record(group[k_nrns]))

        logging.info("added recorders")

        print("- " * 10, "\nstart")
        logging.info("---------------------start")
        t = prun(speed, step_number)
        logging.info("---------------------------end")
        print("- " * 10, "\nend")

        logging.info("simulation done")

        with open(f'./{file_name}/time.txt', 'w') as time_file:
            for time in t:
                time_file.write(str(time) + "\n")

        # for cpg_weight in cpg_ex.weight_changes_vectors:
        #     with hdf5.File('./res_alina/stdp_1/{}_{}.hdf5'.format(str(pc.gid2cell(cpg_weight[0])).split('.')[0],
        #                                                         str(pc.gid2cell(cpg_weight[1])).split('.')[0]),
        #                    'w') as file:
        #         file.create_dataset('#0_step_{}'.format(i), data=cpg_weight[2], compression="gzip")

        for group, recorder in zip(cpg_ex.motogroups, motorecorders):
            spikeout(group[k_nrns], group[k_name], i, recorder)
        for group, recorder in zip(cpg_ex.musclegroups, musclerecorders):
            spikeout(group[k_nrns], group[k_name], i, recorder)
        # for group, recorder in zip(cpg_ex.musclegroups, musclerecorders_extra):
        #     spikeout(group[k_nrns], group[k_name], i, recorder)
        for group, recorder in zip(cpg_ex.motogroups, motorecorders_mem):
            spikeout(group[k_nrns], 'mem_{}'.format(group[k_name]), i, recorder)
        for group, recorder in zip(cpg_ex.affgroups, affrecorders):
            spikeout(group[k_nrns], group[k_name], i, recorder)
        # for group, recorder in zip(cpg_ex.affgroups, affrecorders_extra):
        #     spikeout(group[k_nrns], group[k_name], i, recorder)
        # for group, recorder in zip(cpg_ex.affgroups, affrecorders_axon):
        #     spikeout(group[k_nrns], group[k_name] + "axon", i, recorder)
        for group, recorder in zip(cpg_ex.intgroups, recorders):
            spikeout(group[k_nrns], group[k_name], i, recorder)
        # for group, recorder in zip(cpg_ex.intgroups, recorders_extra):
        #     spikeout(group[k_nrns], group[k_name], i, recorder)
        for group, recorder in zip(cpg_ex.musclegroups, force_recorders):
            spikeout(group[k_nrns], 'force_{}'.format(group[k_name]), i, recorder)
        # for group, recorder in zip(cpg_ex.affgroups, affrecorders):
        #     spikeout_individual_files(group[k_nrns], group[k_name], i, recorder)

        for idx, spike_vec in enumerate(cpg_ex.recorded_spikes):
            with open(f'./{file_name}/spike_times_{idx}.txt', 'w') as f:
                for t in spike_vec:
                    f.write(f"{t}\n")
        logging.info("recorded")

    finish()
