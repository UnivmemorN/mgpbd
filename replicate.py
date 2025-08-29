
import os,sys,shutil
import subprocess
import logging
import datetime


pythonExe = "python"
if sys.platform == "darwin":
    pythonExe = "python3"

# python engine/soft/soft3d.py -use_json=1 -json_path='data/scene/bunny_squash/bunny_squash.json'

allargs = [None]
casenames= {}
args = [
    "engine/soft/soft3d.py", "-use_json=1", "-json_path=data/scene/bunny_squash/bunny_squash.json"
]
allargs.append(args)


def log_args(args:list):
    args1 = " ".join(args) # 将ARGS转换为字符串
    print(f"\nArguments:\n{args1}\n")
    with open("last_run_case.txt", "w") as f:
        f.write(f"{args1}\n")


def run_case(case_num:int):
    if case_num < 1 or case_num >= len(allargs):
        print(f'Invalid case number {case_num}. Exiting...')
        sys.exit(1)
    
    args = allargs[case_num]

    args = [pythonExe, *args]
    log_args(args)

    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError as e:
        date = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
        logging.exception(f"Case {case_num} failed with error\nDate={date}\n\n")
    except KeyboardInterrupt:
        logging.exception(f"KeyboardInterrupt case {case_num}")

def build():
    # cd cpp/mgcg_cuda
    # cmake -B build
    # cmake --build build --config Release --target fastmg
    # cd ../..

    subprocess.check_call("buildcuda.bat",shell=True)
    

if __name__=='__main__':
    # build
    build()
    print(os.getcwd())
    # run
    run_case(1)