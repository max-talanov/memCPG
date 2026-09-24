#!/usr/bin/env python3
import sys
import os
import time
import gc

# Fix for macOS having PYTHONPATH pointing to system NEURON
if "darwin" in sys.platform:
    sys.path = [p for p in sys.path if "/Applications/NEURON/lib/python" not in p]
    # Also unset PYTHONPATH for subprocesses
    if "PYTHONPATH" in os.environ:
        del os.environ["PYTHONPATH"]

from constants import *
from utils_cpg import *
from leg import *
from recorg_cpg import *
from bs_command import WALK


def prun(speed, step_number):
    print(f"🚀 [rank {rank}] Starting simulation: speed={speed}, steps={step_number}")
    logging.info(f"prun start: speed={speed}, steps={step_number}, time_sim={time_sim}")

    try:
        print(f"   Setting timeout...")
        pc.timeout(0)

        h.tstop = time_sim
        sim_t0 = time.perf_counter()

        print(f"   Setting tstop={time_sim}")

        import platform
        is_macos = platform.system() == "Darwin"

        h.dt = 0.025
        h.cvode_active(0)
        if is_macos:
            pc.set_maxstep(1)
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

        sim_elapsed = time.perf_counter() - sim_t0
        logging.info(f"Simulation completed successfully in {sim_elapsed:.3f} s")
        return sim_elapsed

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
    logging.info(f"Rank {rank}/{nhost}, N={N}, Step number: {step_number}, speed={speed}, versions={versions}")

    if rank == 0:
        print(f"   ✅ Clean output directory: {file_name}")

    for i in range(versions):
        print(f"🔄 [rank {rank}] VERSION {i + 1}/{versions} START")
        logging.info(f"=== VERSION {i + 1} START ===")

        try:
            print(f"   Creating CPG network...")
            LEG_L = LEG(speed, bs_fr, 100, step_number, N, leg_l=True)
            LEG_R = LEG(speed, bs_fr, 100, step_number, N, leg_l=False)

            bs_cmd = create_connect_bs_command(LEG_L, LEG_R, mode=WALK)
            add_external_connections(LEG_L, LEG_R, mode=WALK)
            #create_connect_bs(LEG_R, LEG_L)
            #add_external_connections(LEG_R, LEG_L)
            print(f"   ✅ CPG network created successfully")
            logging.info("CPG created successfully")

            # Synchronize after network creation
            pc.barrier()

            print(f"   Setting up voltage recorders...")
            motorecorders_l = []
            motorecorders_mem_l = []
            musclerecorders_l = []
            muscle_units_recorders_l = []
            muscle_am_recorders_l = []
            force_recorders_l = []
            affrecorders_l = []
            recorders_l = []

            motorecorders_r = []
            motorecorders_mem_r = []
            musclerecorders_r = []
            muscle_units_recorders_r = []
            muscle_am_recorders_r = []
            force_recorders_r = []
            affrecorders_r = []
            recorders_r = []

            setup_recorders(LEG_L, motorecorders_mem_l, 'motogroups', 'motor')
            setup_recorders(LEG_R, motorecorders_mem_r, 'motogroups', 'motor')
            setup_recorders(LEG_L, affrecorders_l, 'affgroups', 'afferent')
            setup_recorders(LEG_R, affrecorders_r, 'affgroups', 'afferent')
            setup_recorders(LEG_L, recorders_l, 'intgroups', 'interneuron')
            setup_recorders(LEG_R, recorders_r, 'intgroups', 'interneuron')

            # print(f"      Setting up muscle recorders...")
            for group in getattr(LEG_L, "musclegroups"):
                musclerecorders_l.append(spike_record(group[k_nrns]))
                force_recorders_l.append(force_record(group[k_nrns]))
                muscle_units_recorders_l.append(spike_record(group[k_nrns], location='muscle'))
                muscle_am_recorders_l.append(spike_record(group[k_nrns], location='am'))
            for group in LEG_R.musclegroups:
                musclerecorders_r.append(spike_record(group[k_nrns]))
                force_recorders_r.append(force_record(group[k_nrns]))
                muscle_units_recorders_r.append(spike_record(group[k_nrns], location='muscle'))
                muscle_am_recorders_r.append(spike_record(group[k_nrns], location='am'))
            print(f"      ✅ {len(musclerecorders_l)} muscle recorder groups")

            # print(f"      Setting up Ia generator recorders...")
            # vel_vecs_recorders = velocity_record(cpg_ex.gener_Iagids, attr='_ref_vel')
            # v0_vecs_recorders = velocity_record(cpg_ex.gener_Iagids, attr='_ref_v0')
            # print(f"      ✅ {len(vel_vecs_recorders)} Ia generator recorders")

            print(f"   ✅ All recorders set up successfully")
            logging.info("Added recorders")

            # Synchronize before simulation
            pc.barrier()

            logging.info(f"[version {i + 1}] simulation start")
            sim_time_sec = prun(speed, step_number)
            logging.info(f"[version {i + 1}] simulation time: {sim_time_sec:.3f} s")

            logging.info("Simulation done")

            print(f"   💾 Saving results...")

            left_raw = []
            right_raw = []
            for group, recorder in zip(LEG_L.intgroups, recorders_l):
                trace = gather_population_average(group[k_nrns], recorder)
                if rank == 0 and trace is not None and trace.size:
                    left_raw.append((group[k_name], trace))
            for group, recorder in zip(LEG_R.intgroups, recorders_r):
                trace = gather_population_average(group[k_nrns], recorder)
                if rank == 0 and trace is not None and trace.size:
                    right_raw.append((group[k_name], trace))

            plot_series, left_count = build_overview_series(left_raw, right_raw)
            if rank == 0 and plot_series:
                overview_path = f'./{file_name}/overview_v{i}.png'
                draw_overview_plot(
                    t_values=np.arange(0.0, time_sim + h.dt * 0.5, h.dt, dtype=np.float32),
                    labeled_traces=plot_series,
                    output_path=overview_path,
                    title=f'Overview, speed={speed}, bs={bs_fr}, version={i}',
                    separator_after=left_count
                )
                print(f"      ✅ Overview saved: {overview_path}")

            if rank == 0:
                t_arr = np.arange(0.0, time_sim + h.dt * 0.5, h.dt, dtype=np.float32)
                with open(f'./{file_name}/time.txt', 'w') as time_file:
                    for time_val in t_arr:
                        time_file.write(str(time_val) + "\n")
                del t_arr

            for group, recorder in zip(LEG_L.musclegroups, musclerecorders_l):
                spikeout(group[k_nrns], group[k_name], i, recorder, "left")
            for group, recorder in zip(LEG_L.motogroups, motorecorders_mem_l):
                spikeout(group[k_nrns], 'mem_{}'.format(group[k_name]), i, recorder, "left")
            for group, recorder in zip(LEG_L.affgroups, affrecorders_l):
                spikeout(group[k_nrns], group[k_name], i, recorder, "left")
            for group, recorder in zip(LEG_L.intgroups, recorders_l):
                spikeout(group[k_nrns], group[k_name], i, recorder, "left")
            for group, recorder in zip(LEG_L.musclegroups, force_recorders_l):
                spikeout(group[k_nrns], 'force_{}'.format(group[k_name]), i, recorder, "left")
            for group, recorder in zip(LEG_L.musclegroups, muscle_units_recorders_l):
                spikeout(group[k_nrns], 'units_{}'.format(group[k_name]), i, recorder, "left")
            for group, recorder in zip(LEG_L.musclegroups, muscle_am_recorders_l):
                spikeout(group[k_nrns], 'am_{}'.format(group[k_name]), i, recorder, "left")

            for group, recorder in zip(LEG_R.musclegroups, musclerecorders_r):
                spikeout(group[k_nrns], group[k_name], i, recorder, "right")
            for group, recorder in zip(LEG_R.motogroups, motorecorders_mem_r):
                spikeout(group[k_nrns], 'mem_{}'.format(group[k_name]), i, recorder, "right")
            for group, recorder in zip(LEG_R.affgroups, affrecorders_r):
                spikeout(group[k_nrns], group[k_name], i, recorder, "right")
            for group, recorder in zip(LEG_R.intgroups, recorders_r):
                spikeout(group[k_nrns], group[k_name], i, recorder, "right")
            for group, recorder in zip(LEG_R.musclegroups, force_recorders_r):
                spikeout(group[k_nrns], 'force_{}'.format(group[k_name]), i, recorder, "right")
            for group, recorder in zip(LEG_R.musclegroups, muscle_units_recorders_r):
                spikeout(group[k_nrns], 'units_{}'.format(group[k_name]), i, recorder, "right")
            for group, recorder in zip(LEG_R.musclegroups, muscle_am_recorders_r):
                spikeout(group[k_nrns], 'am_{}'.format(group[k_name]), i, recorder, "right")

            # spikeout(cpg_ex.gener_Iagids, 'vel', i, vel_vecs_recorders)
            # spikeout(cpg_ex.gener_Iagids, 'v0', i, v0_vecs_recorders)
            # print(f"      ✅ Spike data saved")

            del (motorecorders_mem_l, motorecorders_mem_r,
                 affrecorders_l, affrecorders_r,
                 recorders_l, recorders_r,
                 musclerecorders_l, musclerecorders_r,
                 force_recorders_l, force_recorders_r,
                 muscle_units_recorders_l, muscle_units_recorders_r,
                 muscle_am_recorders_l, muscle_am_recorders_r)
            gc.collect()

            if rank == 0:
                logging.info(f"      Saving STDP weight changes...")
                stdp_dir = f'./{file_name}/stdp_1'
                if not os.path.exists(stdp_dir):
                    os.makedirs(stdp_dir)
                    print(f"      ✅ Created STDP directory: {stdp_dir}")

                stdp_count = 0
                for src_gid, post_gid, weight_vec in LEG_L.weight_changes_vectors:
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

                logging.info(f"      ✅ Saved {stdp_count} STDP weight change files")

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