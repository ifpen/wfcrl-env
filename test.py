import numpy as np
from mpi4py import MPI

from wfcrl.interface import MPI_Interface


def test_routine():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print("PYTHON process, Process ", rank, "of size ", size)
    T = 100
    n_turbines = 3
    power = np.zeros(n_turbines, dtype=np.double)
    windspeed = np.zeros(n_turbines, dtype=np.double)
    winddir = np.zeros(n_turbines, dtype=np.double)
    for i in range(T):
        print("PYTHON: Iter ", i)
        command = np.zeros(n_turbines, dtype=np.double)
        if i > 20:
            command[0] = np.radians(30)
        comm.Send(buf=command, dest=0, tag=0)
        comm.Recv(power, source=0, tag=1)
        comm.Recv(windspeed, source=0, tag=2)
        comm.Recv(winddir, source=0, tag=3)
        comm.Barrier()
        with open("example/log.txt", "a") as fp:
            fp.write(
                f"Iter {i} - Sent yaws {np.degrees(command)} - "
                f" Received Power: {power}"
                f" Wind Speed: {windspeed}\n"
                f" Wind Direction: {np.degrees(winddir)}\n"
            )


def test_interface(interface):
    T = 100
    for i in range(T):
        print("PYTHON: Iter ", i)
        command = np.zeros(interface.num_turbines, dtype=np.double)
        if i > 20:
            command[0] = 30
        interface.update_command(command)
        with open("example/log_interface.txt", "a") as fp:
            fp.write(
                f"Iter {i} - Sent command YAW {interface.get_yaw_command()} - "
                f" PITCH {interface.get_pitch_command()}"
                f" TORQUE {interface.get_torque_command()} - "
                f" Received Power: {interface.get_turbine_powers()}"
                f" Wind : {interface.get_turbine_wind()}\n"
            )


if __name__ == "__main__":
    # test_routine()
    interface = MPI_Interface(measurement_window=10, buffer_size=50_000, num_turbines=3)
    test_interface(interface)
