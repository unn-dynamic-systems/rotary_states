import subprocess
import os

def exec(spawn_cmd, spawn_cwd):
    '''
    This function just spawn the process via spawn_cmd
    in spawn_cwd directory, but it cat be everithing
    you want with any positional arguments
    '''
    
    print(f"We spawn {spawn_cmd}")
    p = subprocess.run(spawn_cmd,
                    capture_output=True,
                    shell=True,
                    text=True,
                    cwd=spawn_cwd,
                    check=True, # raise in exitcode != 0
                    executable='bash')

    return p.stdout


def test_limit_cycle_one():
    
    file_to_run = "limit_cycle.py"
    cd = os.path.dirname(os.path.abspath(__file__))

    spawn_cwd = os.path.normpath(os.path.join(cd))
    spawn_cmd = f"poetry run python3 {file_to_run}"
    stdout = exec(spawn_cmd, spawn_cwd)

    with open(f'{spawn_cwd}/refer_stdout.txt') as f:
        assert stdout.strip() == f.read().strip()
        print("OK")


if __name__ == '__main__':
    test_limit_cycle_one()
