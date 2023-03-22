import os, sys
from pathlib import Path
import subprocess


def main():

    project_root = Path(__file__).resolve().absolute().parent.parent
    path_dir = project_root / "aequilibrae" / "paths"
    ipf_dir = project_root / "aequilibrae" / "distribution"

    def compile(dir, file):
        os.chdir(dir)
        compile_cmd = [sys.executable, f"{file}.py", "build_ext", "--inplace"]
        process = subprocess.run(compile_cmd, stdout=subprocess.PIPE, universal_newlines=True)
        print(process.stdout)

    compile(path_dir, 'setup_assignment')
    compile(ipf_dir, 'setup_ipf')


if __name__ == "__main__":
    main()
    