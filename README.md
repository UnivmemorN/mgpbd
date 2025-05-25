# mgpbd

The code for SIGGRAPH 2025 paper "MGPBD: A Multigrid Accelerated Global XPBD Solver".

It is a Multigrid solver to solve the stalling issue of XPBD under high stiffness and high resolution. 

This code use a python front-end mixed with C++/CUDA back-end. C++/CUDA code is compiled to a dll with CUDA 12, and called by ctypes in python.

Paper(arix): https://arxiv.org/abs/2505.13390


CAUTION: The code is still not well organized and not clean. If it has bugs(highly possible), please let me know. 

I only tested it on **Windows 10, VS 2022**.

# Pre-requisites
C++ end
- CUDA 12
- CMake

python end
```
pip install -r requirements.txt
```


# Build

```
buildcuda.bat
```

It will generate `cpp\mgcg_cuda\lib\fastmg.dll`, which is the C++ back-end.

# Run
```
python engine/soft/soft3d.py -use_json=1 -json_path='data/scene/bunny_squash/bunny_squash.json'
```

# Results
By default, they are in `result/latest`. You can aslo specify `-out_dir=xxx`. The output meshes are in ply format. 


# Specify the parameters
Two ways:
1. Just use python argparse, see `common_args.py`. It is a bit messy but I have to support it because of backward compatibility. 
2. Use a json file. `-use_json=1  -json_path=xxx`. I recommend this way. You can easily change the parameters in json file without changing the code because python can dynamically add new attributes.

# Experiments
For cleanness and smaller repo size, I delete most of models, but only keep the bunny squash and armadillo collision cases. If you want to test all experiments in the paper, please let me know (propose a issue).

Just change "solver_type": "AMG", to "XPBD" and set "maxiter" to 100000. You will see that the XPBD is much harder to converge than AMG under high stiffness (Specify by "mu", the 2nd lame parameter) high resolution. This is due to the stalling issue of XPBD(See paper, the last picture).


# Q&A

## CUDA version problem
In Windows, you should specify the dll directory manually.  I use CUDA 12.5, so cuda_dir is set to "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin". If you use other version, please change it.

Related code(in the common_args.py and init_extlib.py).
```
parser.add_argument("-cuda_dir", type=str, default="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin")
...
os.add_dll_directory(cuda_dir)
```

## How to add your own models
The models are in tetgen format(.ele and .node). If you use Houdini, there is a [hip](https://github.com/chunleili/houdini_output_tet) to generate the tetrahedron mesh. 

## How to debug
The python-end is easy to debug. 

For C++ end, I recommond to use `python c++ debugger`， which is a VSCode extension. 

## Pylance report Invalid TypeForm

It is becuase the pylance does not support the `taichi` data types, which is not a big trouble but just annoying. Adding the following code to your `./vscode/settings.json` to stop the warning.
```
"python.analysis.diagnosticSeverityOverrides": {
        "reportInvalidTypeForm": false
    }
```

## CUDA ARCHITECTURES error
In some cases, you should manually set CMAKE_CUDA_ARCHITECTURES based on your GPU.
Change set(CMAKE_CUDA_ARCHITECTURES 89) in cpp/mgcg_cuda/CMakeLists.txt
Run cmake from cpp/mgcg_cuda in **cmd**(Powershell does not work for `set(CMAKE_CUDA_ARCHITECTURES 89).`)

# Note

For cleanness, I delete most of messy codes in this repo. So some experiments may be missing. I put the old repo here: https://github.com/chunleili/tiPBD/tree/amg. 


If you have any questions, please feel free to propose a issue.
