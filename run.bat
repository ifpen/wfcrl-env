@REM mpiexec -n 1 ./example/bin/FAST.Farm_x64_OMP.exe example/TSinflow/TSinflow.fstf : -n 1 python test.py
@REM mpiexec -n 1 ./example/bin/FAST.Farm_x64_OMP.exe example/TSinflow/TSinflow.fstf : -n 1 python test_rl.py
mpiexec -n 1 ./simulators/fastfarm/bin/FAST.Farm_x64_OMP.exe simulators/fastfarm/inputs/FarmInputs/Case.fstf : -n 1 python run.py
@REM mpiexec -n 1 python run.py
