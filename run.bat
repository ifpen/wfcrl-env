@REM mpiexec -n 1 ./example/bin/FAST.Farm_x64_OMP.exe example/TSinflow/TSinflow.fstf : -n 1 python test.py
mpiexec -n 1 ./example/bin/FAST.Farm_x64_OMP.exe example/TSinflow/TSinflow.fstf : -n 1 python test_rl.py
