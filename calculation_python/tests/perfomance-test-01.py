from multiprocessing import Pool, cpu_count
import subprocess
import os
import pathlib
import time


def worker(spawn_cmd, spawn_cwd):
    '''
    This function just spawn the process via spawn_cmd
    in spawn_cwd directory, but it cat be everithing
    you want with any positional arguments
    '''
    try:
        pid = os.getpid()

        print(f"'{spawn_cmd}' process id:", pid)
        p = subprocess.run(spawn_cmd,
                        capture_output=True,
                        shell=True,
                        text=True,
                        cwd=spawn_cwd,
                        check=True, # raise in exitcode != 0
                        executable='bash')

        print(f"CMD WITH PID: {pid}\n'{spawn_cmd}'\nFINISHED WITH EXIT CODE: {p.returncode}")
        if p.stdout:
            print("STDOUT:")
            print(p.stdout)

        if p.stderr:
            print("STDERR:")
            print(p.stderr)
    except Exception as e:
        # catch any exceptions
        print(f"Exception here: {e}")

def get_spawn_cmds_and_spawn_cwd():
    '''
    this function return cmds which we want to execute
    '''
    examples_rel_path = "../examples"
    files_to_run = ["custom_integrator.py",
                    "odeint.py"]

    cd = os.path.dirname(os.path.abspath(__file__))
    spawn_cwd = os.path.normpath(os.path.join(cd, examples_rel_path))

    spawn_cmds = [f"time poetry run python {f}" for f in files_to_run]

    return spawn_cmds, spawn_cwd

def error_callback(e):
    print(e)

def main():
    WORKERS_COUNT = cpu_count()
    spawn_cmds, spawn_cwd = get_spawn_cmds_and_spawn_cwd()

    print(f"There are {WORKERS_COUNT} workers")
    print(f"cwd is {spawn_cwd}")

    args_arr = [(spawn_cmd, spawn_cwd) for spawn_cmd in spawn_cmds]

    # RUN PARALLEL WITH {WORKERS_COUNT} WORKERS
    with Pool(WORKERS_COUNT) as p:
        p.starmap_async(worker, args_arr, error_callback=error_callback).wait()
        print("All done")

if __name__ == '__main__':
    main()

##########################################################
##  A MUCH MORE ABSTRACT EXAMPLE OF PARALLEL USAGE API
##########################################################

# with Pool(WORKERS_COUNT) as p:
#     tasks = [p.apply_async(worker, args=args) for args in args_arr]
#     [task.wait() for task in tasks]
#     p.close(); p.join()
#     print("all done")

# Note, what the map, async_map, starmap_async
# and etc will kill all processes
# if one of them failed, apply_async and etc
# NO, because they are not connected to each other.
# But in our case we actually fine with it, cause we
# catch any error into worker function
