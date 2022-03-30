import subprocess
import os
import pytest

def exec(cmd, dir):
    '''
    This function just spawn the process via 'cmd'
    in 'dir' directory, but it cat be everithing
    you want with any positional arguments
    '''
    
    print(f"We spawn {cmd}")
    p = subprocess.run(cmd,
                    capture_output=True,
                    shell=True,
                    text=True,
                    cwd=dir,
                    check=True, # raise in exitcode != 0
                    executable='bash')

    return p.stdout


def test_limit_cycle_one():
    cd = os.path.dirname(os.path.abspath(__file__))
    spawn_cwd = os.path.normpath(os.path.join(cd))
    
    stdout_file = f'{spawn_cwd}/stdout_file.txt'
    refer_stdout_file = f'{spawn_cwd}/refer_stdout.txt'

    # if os.path.isfile(stdout_file):
    exec(f'rm -f {stdout_file}', spawn_cwd)

    file_to_run = "limit_cycle.py"

    exec(f"poetry run python3 {file_to_run} > {stdout_file}", spawn_cwd)

    # check here
    try:
        exec(f'diff {stdout_file} {refer_stdout_file}', spawn_cwd)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"{e}")


    exec(f'rm {stdout_file}', spawn_cwd)
    print("OK")



if __name__ == '__main__':
    test_limit_cycle_one()
