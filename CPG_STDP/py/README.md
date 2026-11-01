We conducted computational experiments using the neurosimulator NEURON 7.8. We use Python3.7 instead of the standard Hoc interpreter and ParallelContext with MPI for faster calculation.

Firstly, please, install NEURON
Source and installation guide here: https://neuron.yale.edu/neuron/getstd.

Please, install with python(this key: --with-nrnpython) and mpi(key: --with-paranrn for parallel computing)

Before simulation compile mod files
go to directory /CPG_STDP/py

and paste the command:

nrnivmodl ./mod_files

After succesful compile you can see:

Successfully created x86_64/special

For complex activity simulation we recommend to use parallel computing, but even in this case calcutation takes from 1 hour to 12 hours depend on number of steps, speed and number of receptors. In addition, calculation time depends on computer performance and number of threads.

Before run parallel calculation, please, make the directory for recorded results: mkdir res_alina

To run parallel calculation, please, use this command:

mpiexec -n [number_of_threads] nrniv -mpi -python test_stdp_la_with_gid.py 

or mpiexec -n [number_of_threads] nrniv -python test_stdp_la_with_gid.py
