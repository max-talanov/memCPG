import logging

logging.basicConfig(filename='logs.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.info("let's get it started")
import numpy as np
from neuron import h
import h5py as hdf5

h.load_file('nrngui.hoc')

# paralleling NEURON stuff
pc = h.ParallelContext()
rank = int(pc.id())
nhost = int(pc.nhost())

# param
speed = 50  # duration of layer 25 = 21 cm/s; 50 = 15 cm/s; 125 = 6 cm/s
# 100 Hz is the motor cortex frequency
bs_fr = 100  # 40 # frequency of brainstem inputs
versions = 1

step_number = 5  # number of steps

CV_number = 6
nMN = 21  # 21 # 210 # Number of motor neurons
nAff = 12  # 120 # Number of afferents
nInt = 5  # 19 # 196 # Number of neurons in interneuronal pools
N = 5  # 5 #50

k = 0.017  # CV weights multiplier to take into account air and toe stepping
CV_0_len = 12  # 125 # Duration of the CV generator with no sensory inputs
extra_layers = 0  # 1 + layers

one_step_time = int((6 * speed + CV_0_len) / (int(1000 / bs_fr))) * (int(1000 / bs_fr))
time_sim = 25 + one_step_time * step_number

exnclist = []
inhnclist = []
exstdpnclist = []
inhstdpnclist = []
eesnclist = []
stimnclist = []

from interneuron import interneuron
from motoneuron import motoneuron
from bioaffrat import bioaffrat
from muscle import muscle

import random

'''
network topology https://github.com/max-talanov/bypass/blob/main/figs/CPG_feedback_loops.png
'''


class CPG:
    def __init__(self, speed, bs_fr, inh_p, step_number, N):

        self.interneurons = []
        self.motoneurons = []
        self.muscles = []
        self.afferents = []
        self.stims = []
        self.ncell = N
        self.groups = []
        self.motogroups = []
        self.musclegroups = []
        self.affgroups = []
        self.RG_E = []  # Rhythm generators of extensors
        self.RG_F = []  # Rhythm generators of flexor
        self. = []

        for layer in range(CV_number):
            '''cut and muscle feedback'''
            ## self.dict_CV = {layer: 'CV{}'.format(layer + 1)}
            self.dict_CV_1 = {layer: 'CV{}_1'.format(layer + 1)}
            self.dict_RG_E = {layer: 'RG{}_E'.format(layer + 1)}
            self.dict_RG_F = {layer: 'RG{}_F'.format(layer + 1)}
            self.dict_V3F = {layer: 'V3{}_F'.format(layer + 1)}
            self.dict_C = {layer: 'C{}'.format(layer + 1)}

        for layer in range(CV_number):
            '''Cutaneous pools'''
            ##self.dict_CV[layer] = self.addpool(self.ncell, "CV" + str(layer + 1), "aff")
            self.dict_CV_1[layer] = self.addpool(self.ncell, "CV" + str(layer + 1) + "_1", "aff")

            '''Rhythm generator pools'''
            self.dict_RG_E[layer] = self.addpool(self.ncell, "RG" + str(layer + 1) + "_E", "int")
            self.dict_RG_F[layer] = self.addpool(self.ncell, "RG" + str(layer + 1) + "_F", "int")
            self.RG_E.append(self.dict_RG_E[layer])
            self.RG_F.append(self.dict_RG_F[layer])

        '''RG'''
        self.RG_E = sum(self.RG_E, [])
        self.InE = self.addpool(nInt, "InE", "int")
        self.RG_F = sum(self.RG_F, [])
        self.InF = self.addpool(nInt, "InF", "int")

        '''sensory and muscle afferents and brainstem and V3F'''
        ## self.sens_aff = self.addpool(nAff, "sens_aff", "aff")
        self.Ib_aff_E = self.addpool(nAff, "Ib_aff_E", "aff")
        self.Ib_aff_F = self.addpool(nAff, "Ib_aff_F", "aff")

        '''muscles'''
        self.muscle_E = self.addpool(nMN * 30, "muscle_E", "muscle")
        self.muscle_F = self.addpool(nMN * 20, "muscle_F", "muscle")

        '''reflex arc'''
        self.Ib_E = self.addpool(nInt, "Ib_E", "int")
        self.Ib_F = self.addpool(nInt, "Ib_F", "int")
        # self.Ibgener_E = []
        # self.Ibgener_F = []

        '''BS'''
        # periodic stimulation
        self.E_bs_gids, self.F_bs_gids = self.add_bs_geners(bs_fr, 10)

        '''muscle afferents generators'''
        self.Ibgener_E = self.addIbgener(self.muscle_E, self.muscle_E, 10, weight=20)
        self.Ibgener_F = self.addIbgener(self.muscle_F, self.muscle_F, speed * 6, weight=20)

        '''cutaneous inputs'''
        cfr = 200
        c_int = 1000 / cfr

        '''cutaneous inputs generators'''
        for layer in range(CV_number):
            self.dict_C[layer] = []
            for i in range(step_number):
                self.dict_C[layer].append(self.addgener(25 + speed * layer + i * (speed * CV_number + CV_0_len),
                                                        random.gauss(cfr, cfr / 10), (speed / c_int + 1)))

        self.C_1 = []
        self.C_0 = []
        self.V0v = []

        '''Generators'''
        for i in range(step_number):
            self.C_0.append(self.addgener(25 + speed * 6 + i * (speed * 6 + CV_0_len), cfr, CV_0_len / c_int, False))
            ## TODO possibly we don't need this
            self.V0v.append(self.addgener(40 + speed * 6 + i * (speed * 6 + CV_0_len), cfr, 100 / c_int, False))

        # self.C_0.append(self.addgener(0, cfr, (speed / c_int)))

        for layer in range(CV_number):
            self.C_1.append(self.dict_CV_1[layer])
        self.C_1 = sum(self.C_1, [])
        # self.C_0 = sum(self.C_0, [])

        ## TODO possibly project to RG_F
        ## connectcells(self.V0v, self.RG_F, 3.75, 3)


        '''generators of Ib aff'''
        ## TODO originally: 00005 and 0001
        ## TODO fix Ibgener
        genconnect(self.Ibgener_E, self.Ib_aff_E, 0.5, 1, False, 5)
        genconnect(self.Ibgener_F, self.Ib_aff_F, 1.5, 1, False, 15)
        # genconnect(Ibgener_E_1000, self.Ib_aff_E, 5.0, 1, False, 5)
        # genconnect(Ibgener_F_1000, self.Ib_aff_F, 5.0, 1, False, 15)


        for layer in range(CV_number):
            '''Internal to RG topology'''
            connectinsidenucleus(self.dict_RG_F[layer])
            connectinsidenucleus(self.dict_RG_E[layer])


            '''Neg feedback RG -> Ib'''
            ## TODO why do we have this neg feedback ?
            if layer > 3:
                connectcells(self.dict_RG_E[layer], self.Ib_aff_E, layer * 0.0002, 1, True)
            else:
                '''RG2Ib'''
                connectcells(self.dict_RG_E[layer], self.Ib_aff_E, 0.0001, 1, True)

            '''Neg feedback loop RG->Ib'''
            connectcells(self.dict_RG_F[layer], self.Ib_aff_F, 0.95, 1, True)

        '''cutaneous inputs'''
        for layer in range(CV_number):
            connectcells(self.dict_C[layer], self.dict_CV_1[layer], 0.15 * k * speed, 2)
            connectcells(self.dict_CV_1[layer], self.dict_RG_E[layer], 0.00035 * k * speed, 3)

        # connectcells(self.IP_F, self.Ib_aff_F, 0.0015, 2, True)
        # connectcells(self.IP_E, self.Ib_aff_E, 0.0015, 2, True)

        '''Rhythm generators'''
        for layer in range(CV_number):
            connectcells(self.dict_RG_E[layer], self.InE, 0.001, 1)
            ## TODO weight 0.0001
            connectcells(self.dict_RG_F[layer], self.InF, 0.001, 1)

        '''Ib2RG, RG2Motor'''
        connectcells(self.InE, self.RG_F, 0.5, 1, True)
        '''STDP synapse'''
        connectcells(self.Ib_aff_F, self.RG_F, 0.5, 1, stdptype=True)

        # TODO check this too many reciprocal inh connections
        connectcells(self.InE, self.Ib_aff_F, 1.2, 1, True)

        connectcells(self.InF, self.RG_E, 0.8, 1, True)
        '''STDP synapse'''
        connectcells(self.Ib_aff_E, self.RG_E, 0.5, 1, stdptype=True)

        # TODO check this too many reciprocal inh connections
        ## connectcells(self.InF, self.InE, 0.5, 1, True)
        connectcells(self.InF, self.Ib_aff_E, 0.5, 1, True)

        '''reflex arc'''
        connectcells(self.InE, self.Ib_E, 0.001, 1)
        connectcells(self.Ib_aff_E, self.Ib_E, 0.008, 1)

        connectcells(self.InF, self.Ib_F, 0.001, 1)
        connectcells(self.Ib_aff_F, self.Ib_F, 0.008, 1)

        connectcells(self.Ib_E, self.Ib_F, 0.08, 1, True)
        connectcells(self.Ib_F, self.Ib_E, 0.08, 1, True)
        ## TODO check the inh connection
        connectcells(self.InE, self.InF, 0.04, 1, True)
        connectcells(self.InF, self.InE, 0.04, 1, True)

    def addpool(self, num, name="test", neurontype="int"):
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
        gid = 0
        if neurontype.lower() == "delay":
            delaytype = True
        else:
            delaytype = False

        if neurontype.lower() == "moto":
            diams = motodiams(num)
        for i in range(rank, num, nhost):
            if neurontype.lower() == "moto":
                cell = motoneuron(diams[i])
                self.motoneurons.append(cell)
            elif neurontype.lower() == "aff":
                cell = bioaffrat()
                self.afferents.append(cell)
            elif neurontype.lower() == "muscle":
                cell = muscle()
                self.motoneurons.append(cell)
                self.muscles.append(cell)
            elif neurontype.lower() == "bursting":
                cell = interneuron(False, bursting_mode=True)
                self.interneurons.append(cell)
            else:
                cell = interneuron(delaytype)
                self.interneurons.append(cell)

            while pc.gid_exists(gid) != 0:
                gid += 1
            gids.append(gid)
            pc.set_gid2node(gid, rank)
            nc = cell.connect2target(None)
            pc.cell(gid, nc)

        # Groups
        if (neurontype.lower() == "muscle"):
            self.musclegroups.append((gids, name))
            self.motogroups.append((gids, name))
        elif (neurontype.lower() == "moto"):
            self.motogroups.append((gids, name))
        elif neurontype.lower() == "aff":
            self.affgroups.append((gids, name))
        else:
            self.groups.append((gids, name))

        return gids

    def add_bs_geners(self, freq, spikes_per_step):
        E_bs_gids = []
        F_bs_gids = []
        for step in range(step_number):
            F_bs_gids.append(self.addgener(int(one_step_time * (step + 0.5)), freq, spikes_per_step, False))
            E_bs_gids.append(self.addgener(int(one_step_time * step), freq, spikes_per_step, False))
        return E_bs_gids, F_bs_gids

    def addgener(self, start, freq, nums, noise=True):
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
        noise: bool
            generates noizy output
        Returns
        -------
        gid: int
            generator gid
        '''
        gid = 0
        stim = h.NetStim()
        stim.number = nums
        if noise:
            stim.start = random.uniform(start - 3, start + 3)
            stim.noise = 0.05
        else:
            stim.start = start
        stim.interval = int(1000 / freq)
        # skinstim.noise = 0.1
        self.stims.append(stim)
        while pc.gid_exists(gid) != 0:
            gid += 1
        pc.set_gid2node(gid, rank)
        ncstim = h.NetCon(stim, None)
        pc.cell(gid, ncstim)
        return gid

    def addIbgener(self, mn, mn2, start, weight=1.0):
        '''
        Creates self.Ib generators and returns generator gids
        Parameters
        ----------
        mn:
            motor neurons of agonist muscle that contract spindle
        mn2:
            motor neurons of antagonist muscle that extend spindle
        start: int
            generator start up
        num: int
            number in pool
        w_in: int
            weight of the connection
        Returns
        -------
        gids: list
            generators gids
        '''
        gid = 0
        moto = pc.gid2cell(random.randint(mn[0], mn[-1]))
        moto2 = pc.gid2cell(random.randint(mn2[0], mn2[-1]))
        stim = h.IbGenerator(0.5)
        stim.start = start
        h.setpointer(moto.muscle_unit(0.5)._ref_F_fHill, 'fhill', stim)
        h.setpointer(moto2.muscle_unit(0.5)._ref_F_fHill, 'fhill2', stim)
        self.stims.append(stim)
        while pc.gid_exists(gid) != 0:
            gid += 1
        pc.set_gid2node(gid, rank)
        ncstim = h.NetCon(stim, None)
        ncstim.weight[0] = weight
        pc.cell(gid, ncstim)

        return gid


def connectcells(pre, post, weight, delay=1, inhtype=False, N=15, stdptype=False, threshold=10):
    ''' Connects with excitatory synapses
      Parameters
      ----------
      pre: list
          list of presynase neurons gids
      post: list
          list of postsynapse neurons gids
      weight: float
          weight of synapse
          used with Gaussself.Ian distribution
      delay: int
          synaptic delay
          used with Gaussself.Ian distribution
      nsyn: int
          numder of synapses
      inhtype: bool
          is this connection inhibitory?
      N: int
          number of synapses
      stdptype: bool
           is connection stdp?
      threshold: int
            voltage thershold
    '''
    nsyn = random.randint(N - 10, N)
    for post_gid in post:
        if pc.gid_exists(post_gid):
            for j in range(nsyn):
                src_gid = random.randint(pre[0], pre[-1])
                target = pc.gid2cell(post_gid)
                if stdptype:
                    if inhtype:
                        syn = target.synlistinhstdp[j]
                        nc = pc.gid_connect(src_gid, syn)
                        nc.delay = delay
                        pc.threshold(src_gid, threshold)
                        """Create STDP synapses"""
                        dummy = h.Section()  # Create a dummy section to put the point processes in
                        stdpmech = h.STDP(0, sec=dummy)  # Create the STDP mechanism
                        # TODO check target, threshold,
                        presyn = pc.gid_connect(src_gid,
                                                stdpmech)  # threshold, delay, 1)  # Feed presynaptic spikes to the STDP mechanism -- must have weight >0
                        presyn.delay = delay
                        presyn.weight = 1
                        pstsyn = pc.gid_connect(post_gid,
                                                stdpmech)  # threshold, delay, -1)  # Feed postsynaptic spikes to the STDP mechanism -- must have weight <0
                        pstsyn.delay = delay
                        pstsyn.weight = -1
                        pc.threshold(post_gid, threshold)
                        h.setpointer(nc._ref_weight[0], 'synweight',
                                     stdpmech)  # Point the STDP mechanism to the connection weight
                        inhstdpnclist.append(nc)
                    else:
                        syn = target.synlistex[j]
                        nc = pc.gid_connect(src_gid, syn)
                        nc.delay = delay
                        pc.threshold(src_gid, threshold)
                        """Create STDP synapses"""
                        dummy = h.Section()  # Create a dummy section to put the point processes in
                        stdpmech = h.STDP(0, sec=dummy)  # Create the STDP mechanism
                        # TODO check target, threshold,
                        presyn = pc.gid_connect(src_gid,
                                                stdpmech)  # threshold, delay, 1)  # Feed presynaptic spikes to the STDP mechanism -- must have weight >0
                        presyn.delay = delay
                        presyn.weight[0] = 1
                        pstsyn = pc.gid_connect(post_gid,
                                                stdpmech)  # threshold, delay, -1)  # Feed postsynaptic spikes to the STDP mechanism -- must have weight <0
                        pstsyn.delay = delay
                        pstsyn.weight[0] = -1
                        pc.threshold(post_gid, threshold)
                        h.setpointer(nc._ref_weight[0], 'synweight',
                                     stdpmech)  # Point the STDP mechanism to the connection weight
                        exstdpnclist.append(nc)
                        # nc.weight[0] = random.gauss(weight, weight / 6) # str

                else:
                    if inhtype:
                        syn = target.synlistinh[j]
                        nc = pc.gid_connect(src_gid, syn)
                        inhnclist.append(nc)
                    else:
                        syn = target.synlistex[j]
                        nc = pc.gid_connect(src_gid, syn)
                        exnclist.append(nc)
                        # nc.weight[0] = random.gauss(weight, weight / 6) # str

                # if mode == 'STR':
                #     nc.weight[0] = 0 # str
                # else:
                nc.weight[0] = random.gauss(weight, weight / 5)
                nc.delay = random.gauss(delay, delay / 5)


def genconnect(gen_gid, afferents_gids, weight, delay, inhtype=False, N=50):
    ''' Connects with generator
      Parameters
      ----------
      afferents_gids: list
          list of presynase neurons gids
      gen_gid: int
          generator gid
      weight: float
          weight of synapse
          used with Gaussian distribution
      delay: int
          synaptic delay
          used with Gaussian distribution
      nsyn: int
          numder of synapses
      inhtype: bool
          is this connection inhibitory?
    '''
    nsyn = random.randint(N, N + 5)
    for i in afferents_gids:
        if pc.gid_exists(i):
            for j in range(nsyn):
                target = pc.gid2cell(i)
                if inhtype:
                    syn = target.synlistinh[j]
                    # nc = pc.gid_connect(gen_gid, syn)
                    # stimnclist.append(nc)
                    # nc.delay = random.gauss(delay, delay / 6)
                    # nc.weight[0] = 0
                else:
                    syn = target.synlistees[j]
                    # nc = pc.gid_connect(gen_gid, syn)
                    # stimnclist.append(nc)
                    # nc.delay = random.gauss(delay, delay / 6)
                    # nc.weight[0] = random.gauss(weight, weight / 6)
                nc = pc.gid_connect(gen_gid, syn)
                stimnclist.append(nc)
                nc.delay = random.gauss(delay, delay / 5)
                nc.weight[0] = random.gauss(weight, weight / 5)


def connectinsidenucleus(nucleus):
    connectcells(nucleus, nucleus, 0.25, 0.5)


def spike_record(pool, extra=False):
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
            vec.record(cell.soma(0.5)._ref_v)
        v_vec.append(vec)
    return v_vec


def force_record(pool):
    ''' Records force from gids of motor neurons muscle unit
      Parameters
      ----------
      pool: list
        list of neurons gids
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


def motodiams(number):
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
        with hdf5.File('./res_test/{}_sp_{}_CVs_{}_bs_{}.hdf5'.format(name, speed, CV_number, bs_fr), 'w') as file:
            for i in range(step_number):
                sl = slice((int(1000 / bs_fr) * 40 + i * one_step_time * 40),
                           (int(1000 / bs_fr) * 40 + (i + 1) * one_step_time * 40))
                file.create_dataset('#0_step_{}'.format(i), data=np.array(result)[sl], compression="gzip")
    else:
        logging.info(rank)


def prun(speed, step_number):
    ''' simulation control
    Parameters
    ----------
    speed: int
      duration of each layer
    '''
    pc.timeout(0)
    tstop = time_sim  # 25 + (6 * speed + 125) * step_number
    pc.set_maxstep(10)
    h.stdinit()
    pc.psolve(tstop)


def finish():
    ''' proper exit '''
    pc.runworker()
    pc.done()
    # print("hi after finish")
    h.quit()


if __name__ == '__main__':
    '''
    cpg_ex: cpg
        topology of central pattern generation + reflex arc
    '''
    k_nrns = 0
    k_name = 1

    for i in range(versions):
        cpg_ex = CPG(speed, bs_fr, 100, step_number, N)
        logging.info("created")
        motorecorders = []
        motorecorders_mem = []
        force_recorders = []
        for group in cpg_ex.motogroups:
            motorecorders.append(spike_record(group[k_nrns], True))

        for group in cpg_ex.motogroups:
            motorecorders_mem.append(spike_record(group[k_nrns]))
        affrecorders = []
        for group in cpg_ex.affgroups:
            affrecorders.append(spike_record(group[k_nrns], i))
        recorders = []
        for group in cpg_ex.groups:
            recorders.append(spike_record(group[k_nrns], i))
        for group in cpg_ex.musclegroups:
            force_recorders.append(force_record(group[k_nrns]))

        logging.info("added recorders")

        print("- " * 10, "\nstart")
        prun(speed, step_number)
        print("- " * 10, "\nend")

        logging.info("simulation done")

        for group, recorder in zip(cpg_ex.motogroups, motorecorders):
            spikeout(group[k_nrns], group[k_name], i, recorder)
        for group, recorder in zip(cpg_ex.motogroups, motorecorders_mem):
            spikeout(group[k_nrns], 'mem_{}'.format(group[k_name]), i, recorder)
        for group, recorder in zip(cpg_ex.affgroups, affrecorders):
            spikeout(group[k_nrns], group[k_name], i, recorder)
        for group, recorder in zip(cpg_ex.groups, recorders):
            spikeout(group[k_nrns], group[k_name], i, recorder)
        for group, recorder in zip(cpg_ex.musclegroups, force_recorders):
            spikeout(group[k_nrns], 'force_{}'.format(group[k_name]), i, recorder)

        logging.info("recorded")

    finish()
