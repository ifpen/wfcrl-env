import numpy as np
from mpi4py import MPI

from interface import MPI_Interface
from src.envs import FarmEnv


def test_routine(comm):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print("PYTHON process, Process ", rank, "of size ", size)
    T = 10
    n_turbines = 3
    power = np.zeros(n_turbines, dtype=np.double)
    windspeed = np.zeros(n_turbines, dtype=np.double)
    for i in range(T):
        print("PYTHON: Iter ", i)
        command = np.radians(np.ones(n_turbines, dtype=np.double) * i)
        comm.Send(buf=command, dest=0, tag=0)
        # comm.Barrier()
        comm.Recv(power, source=0, tag=1)
        comm.Recv(windspeed, source=0, tag=2)
        comm.Barrier()
        with open("example/log.txt", "a") as fp:
            fp.write(f"Sent {i} - Received Power: {power}" f"Wind Speed: {windspeed}\n")


def rl_routine(env, T=20):
    r = 0
    for i in range(T):
        observation, reward, done, info = env.step(np.array([0, 0, 0]))
        r += reward
    return


if __name__ == "__main__":
    env = FarmEnv(
        config="",
        interface=MPI_Interface,
        state_lb=np.array([-75, 0, 3]),
        state_ub=np.array([75, 360, 28]),
    )
