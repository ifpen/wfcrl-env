import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print("PYTHON process, Process ", rank, "of size ", rank)

T = 400
n_turbines = 3
power = np.zeros(n_turbines, dtype=np.double)
windspeed = np.zeros(n_turbines, dtype=np.double)
for i in range(T):
    # comm.Barrier()
    print("PYTHON: Iter ", i)
    comm.Send(buf=np.radians(np.ones(n_turbines, dtype=np.double)*i), dest=0, tag=0)
    comm.Barrier()
    comm.Recv(power, source=0, tag=1)
    comm.Recv(windspeed, source=0, tag=2)
    comm.Barrier()
    with open("example/log.txt", "a") as fp:
        fp.write(f"Sent {i} - Received Power: {power}, Wind Speed: {windspeed}\n")
