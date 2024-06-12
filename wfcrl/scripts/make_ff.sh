#!/usr/bin/env bash

conda install --yes -c conda-forge openfast==3.5.3

export I_MPI_SPAWN=on
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
cd ../simulators/fastfarm/src/DISCON
rm -f -r build && mkdir build && cd build
MPIF90_PATH=$(which mpif90)
cmake .. -DCMAKE_Fortran_COMPILER=${MPIF90_PATH}
make
make install
cd ../../
cp ../servo_dll/DISCON.dll ../servo_dll/DISCON_WT1.dll

cd SC_DLL
rm -f -r build && mkdir build && cd build
MPIF90_PATH=$(which mpif90)
cmake .. -DCMAKE_Fortran_COMPILER=${MPIF90_PATH}
make
make install
