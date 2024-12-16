!subroutine sc_init_obfuscator()
!
!   CALL RANDOM_SEED ( SIZE = 1 )
!   CALL RANDOM_SEED ( PUT=3459872 )
!
!end subroutine sc_init_obfuscator



!=======================================================================
!SUBROUTINE sc_init (  ) BIND (C, NAME='sc_init')
!subroutine sc_init ( nTurbines, nInpGlobal, NumCtrl2SC, NumParamGlobal, ParamGlobal, NumParamTurbine, &
!                        ParamTurbine, NumStatesGlobal, NumStatesTurbine, NumSC2CtrlGlob, &
!                        NumSC2Ctrl, errStat, errMsg )  bind (C, NAME='sc_init')
subroutine SC_Init ( nTurbines, nInpGlobal, NumCtrl2SC, NumParamGlobal,  NumParamTurbine, &
                         NumStatesGlobal, NumStatesTurbine, NumSC2CtrlGlob, &
                        NumSC2Ctrl, errStat, errMsg )  bind (C, NAME='sc_init')
!subroutine sc_init ( t, nTurbines, nInpGlobal, to_SCglob, NumCtrl2SC, to_SC, &
!                        nStatesGlobal, StatesGlob, nStatesTurbine, StatesTurbine, NumSC2CtrlGlob, from_SCglob, &
!                        NumSC2Ctrl, from_SC, errStat, errMsg )  bind (C, NAME='sc_calcOutputs')


   ! This DLL super controller is used to implement a ...

   ! Modified by B. Jonkman to conform to ISO C Bindings (standard Fortran 2003) and
   ! compile with either gfortran or Intel Visual Fortran (IVF)
   ! DO NOT REMOVE or MODIFY LINES starting with "!DEC$" or "!GCC$"
   ! !DEC$ specifies attributes for IVF and !GCC$ specifies attributes for gfortran
   !
   ! Note that gfortran v5.x on Mac produces compiler errors with the DLLEXPORT attribute,
   ! so I've added the compiler directive IMPLICIT_DLLEXPORT.

   use, intrinsic :: ISO_C_Binding

   implicit                        none
#ifndef IMPLICIT_DLLEXPORT
!DEC$ ATTRIBUTES DLLEXPORT :: sc_init
!GCC$ ATTRIBUTES DLLEXPORT :: sc_init
#endif
   integer(C_INT),            intent(in   ) :: nTurbines         !< number of turbines connected to this supercontroller
   integer(C_INT),            intent(  out) :: nInpGlobal          !< number of global inputs to supercontroller
   integer(C_INT),            intent(  out) :: NumCtrl2SC          !< number of turbine controller outputs [inputs to supercontroller]
   integer(C_INT),            intent(  out) :: NumParamGlobal      !< number of global parameters
   integer(C_INT),            intent(  out) :: NumParamTurbine     !< number of parameters per turbine
   integer(C_INT),            intent(  out) :: NumStatesGlobal       !< number of global states
   integer(C_INT),            intent(  out) :: NumStatesTurbine      !< number of states per turbine
   integer(C_INT),            intent(  out) :: NumSC2CtrlGlob      !< number of global controller inputs [from supercontroller]
   integer(C_INT),            intent(  out) :: NumSC2Ctrl          !< number of turbine specific controller inputs [output from supercontroller]
   integer(C_INT),            intent(  out) :: errStat             !< error status code (uses NWTC_Library error codes)
   character(kind=C_CHAR),    intent(inout) :: errMsg          (*) !< Error Message from DLL to simulation code

   !errMsg = TRANSFER( TRIM(avcMSG)//C_NULL_CHAR, avcMSG, SIZE(avcMSG) )
   errStat           = 0
   !errMsg            = ''

   nInpGlobal        = 0
   NumCtrl2SC        = 12
   NumParamGlobal    = 1
   NumParamTurbine   = 3
   NumStatesGlobal   = 0
   NumStatesTurbine  = 0
   NumSC2CtrlGlob    = 0
   NumSC2Ctrl        = 6


   return

 end subroutine SC_Init

   subroutine SC_GetInitData(nTurbines, NumParamGlobal, NumParamTurbine, ParamGlobal, ParamTurbine, &
        NumSC2CtrlGlob, from_SCglob, NumSC2Ctrl, from_SC,&
        & nStatesGlobal, StatesGlob, nStatesTurbine, StatesTurbine,&
        & errStat, errMsg )  bind (C, NAME='sc_getInitData')
use, intrinsic :: ISO_C_Binding

   implicit                        none
#ifndef IMPLICIT_DLLEXPORT
!DEC$ ATTRIBUTES DLLEXPORT :: sc_getInitData
!GCC$ ATTRIBUTES DLLEXPORT :: sc_getInitData
#endif
   integer(C_INT),            intent(in   ) :: nTurbines         !< number of turbines connected to this supercontroller
   integer(C_INT),            intent(in   ) :: NumParamGlobal      !< number of global parameters
   integer(C_INT),            intent(in   ) :: NumParamTurbine     !< number of parameters per turbine
   real(C_FLOAT),             intent(inout) :: ParamGlobal     (*) !< global parameters
   real(C_FLOAT),             intent(inout) :: ParamTurbine    (*) !< turbine-based parameters
   integer(C_INT),            intent(in   ) :: NumSC2CtrlGlob    !< number of global controller inputs [from supercontroller]
   real(C_FLOAT),             intent(inout) :: from_SCglob  (*)  !< global outputs of the super controller (to the turbine controller)
   integer(C_INT),            intent(in   ) :: NumSC2Ctrl        !< number of turbine specific controller inputs [output from supercontroller]
   real(C_FLOAT),             intent(inout) :: from_SC      (*)  !< turbine specific outputs of the super controller (to the turbine controller)
   integer(C_INT),         intent(in   ) :: nStatesGlobal     !< number of global states
   real(C_FLOAT),          intent(inout) :: StatesGlob   (*)  !< global states at time increment, n=0 (total of nStatesGlobal of these states)
   integer(C_INT),         intent(in   ) :: nStatesTurbine    !< number of states per turbine
   real(C_FLOAT),          intent(inout) :: StatesTurbine(*)  !< turbine-dependent states at time increment, n=0 (total of nTurbines*nStatesTurbine of these states)
   integer(C_INT),            intent(inout) :: errStat             !< error status code (uses NWTC_Library error codes)
   character(kind=C_CHAR),    intent(inout) :: errMsg          (*) !< Error Message from DLL to simulation code
   integer                                  :: i,j
   real(C_FLOAT), allocatable               :: mask1(:)
   integer                                  :: seedVal(1), nSeeds

       ! Add a data obfuscator for your proprietary Parameter data

   !nSeeds     = 1
   !seedVal(1) = 3459872
   !call random_seed ( size = nSeeds  )
   !call random_seed ( put  = seedVal )
   !allocate(mask1(NumParamGlobal), stat = errStat)
   !call random_number( mask1 )

   !Turbine parameters set the frequency of oscillation of demanded yaw rate for each turbine
   do j = 1, nTurbines
      ParamTurbine(j) = real((nTurbines-j)*.1, C_FLOAT)
   end do

   do j = 1, nTurbines
      do i = 1, NumSC2Ctrl
         from_SC((j-1)*NumSC2Ctrl+i) = real( 10.0 * sin(ParamTurbine(j)*0.0), C_FLOAT)
      end do
   end do

    end subroutine SC_GetInitData
!=======================================================================


    !SUBROUTINE sc_calcOutputs (  ) BIND (C, NAME='sc_calcOutputs')
subroutine SC_CalcOutputs ( t, nTurbines, nParamGlobal, paramGlobal, nParamTurbine, paramTurbine, nInpGlobal, to_SCglob, NumCtrl2SC, to_SC, &
                        nStatesGlobal, StatesGlob, nStatesTurbine, StatesTurbine, NumSC2CtrlGlob, from_SCglob, &
                        NumSC2Ctrl, from_SC, errStat, errMsg  )  bind (C, NAME='sc_calcOutputs')


   ! This DLL super controller is used to implement a ...

   ! Modified by B. Jonkman to conform to ISO C Bindings (standard Fortran 2003) and
   ! compile with either gfortran or Intel Visual Fortran (IVF)
   ! DO NOT REMOVE or MODIFY LINES starting with "!DEC$" or "!GCC$"
   ! !DEC$ specifies attributes for IVF and !GCC$ specifies attributes for gfortran
   !
   ! Note that gfortran v5.x on Mac produces compiler errors with the DLLEXPORT attribute,
   ! so I've added the compiler directive IMPLICIT_DLLEXPORT.

use, intrinsic :: ISO_C_Binding

!use typeConf

   implicit                        none
#ifndef IMPLICIT_DLLEXPORT
!DEC$ ATTRIBUTES DLLEXPORT :: sc_calcOutputs
!GCC$ ATTRIBUTES DLLEXPORT :: sc_calcOutputs
#endif
   include 'mpif.h'

   real(C_DOUBLE),         INTENT(IN   ) :: t                 !< time (s)
   integer(C_INT),         intent(in   ) :: nTurbines         !< number of turbines connected to this supercontroller
   integer(C_INT),         intent(in   ) :: nParamGlobal        !< number of global parameters for the supercontroller
   real(C_FLOAT),          intent(in   ) :: paramGlobal    (*)  !< global parameters for the supercontroller
   integer(C_INT),         intent(in   ) :: nParamTurbine        !< number of turbine-based parameters for supercontroller
   real(C_FLOAT),          intent(in   ) :: paramTurbine    (*)  !< turbine-based parameters for the supercontroller
   integer(C_INT),         intent(in   ) :: nInpGlobal        !< number of global inputs to supercontroller
   real(C_FLOAT),          intent(in   ) :: to_SCglob    (*)  !< global inputs to the supercontroller
   integer(C_INT),         intent(in   ) :: NumCtrl2SC        !< number of turbine controller outputs [inputs to supercontroller]
   real(C_FLOAT),          intent(in   ) :: to_SC        (*)  !< inputs to the super controller (from the turbine controller)
   integer(C_INT),         intent(in   ) :: nStatesGlobal     !< number of global states
   real(C_FLOAT),          intent(in   ) :: StatesGlob   (*)  !< global states at time increment, n (total of nStatesGlobal of these states)
   integer(C_INT),         intent(in   ) :: nStatesTurbine    !< number of states per turbine
   real(C_FLOAT),          intent(in   ) :: StatesTurbine(*)  !< turbine-dependent states at time increment, n (total of nTurbines*nStatesTurbine of these states)
   integer(C_INT),         intent(in   ) :: NumSC2CtrlGlob    !< number of global controller inputs [from supercontroller]
   real(C_FLOAT),          intent(inout) :: from_SCglob  (*)  !< global outputs of the super controller (to the turbine controller)
   integer(C_INT),         intent(in   ) :: NumSC2Ctrl        !< number of turbine specific controller inputs [output from supercontroller]
   real(C_FLOAT),          intent(inout) :: from_SC      (*)  !< turbine specific outputs of the super controller (to the turbine controller)
   integer(C_INT),         intent(inout) :: errStat           !< error status code (uses NWTC_Library error codes)
   character(kind=C_CHAR), intent(inout) :: errMsg       (*)  !< Error Message from DLL to simulation code
   integer                               :: i, j, c

   real(C_DOUBLE) :: ws(nTurbines)
   real(C_DOUBLE) :: wd(nTurbines)
   real(C_DOUBLE) :: power(nTurbines)
   REAL(C_DOUBLE) :: yawRef(nTurbines+1)
   REAL(C_DOUBLE) :: pitchRef(nTurbines+1)
   REAL(C_DOUBLE) :: torqueRef(nTurbines+1)
   REAL(C_DOUBLE) :: to_interface(nTurbines*NumCtrl2SC)

   ! For MPI communication
   integer this_process_rank, size_of_cluster, ierror
   !integer initialized
   logical initialized, finalized
   integer message
   integer yaw_tag, pitch_tag, torque_tag, com_tag, measures_tag
   integer, save :: parent_comm
   integer, save :: command_process_rank
   REAL(C_DOUBLE) :: max_iter_com
   integer, save :: max_iter
   integer, save :: num_iter = 0

   integer, dimension(MPI_STATUS_SIZE) :: statut


   initialized = .false.
   finalized = .false.

   ! MPI comm tags
   yaw_tag = 1
   pitch_tag = 2
   torque_tag = 3
   com_tag = 0
   measures_tag = 4


   call MPI_INITIALIZED(initialized, ierror)

   if (int(t)==0) then
      print *, 'Initialize MPI'
      call MPI_INIT(ierror)
      call MPI_Comm_get_parent(parent_comm, ierror)
      if (parent_comm == MPI_COMM_NULL) THEN
          print *, 'This is not a spawned process. Using MPI_COMM_WORLD'
          parent_comm = MPI_COMM_WORLD
          !call MPI_COMM_SIZE(parent_comm, size_of_cluster, ierror)
          call MPI_COMM_RANK(parent_comm, this_process_rank, ierror)
          ! Will send and receive via intra-communicator
          command_process_rank = 1 - this_process_rank
      else
          print *, 'Found parent process', parent_comm
          ! Will send and receive via inter-communicator to the process of rank 0
          command_process_rank = 0
      end if
      !print *, 'Hello World from process: ', this_process_rank, 'of ', size_of_cluster
      ! Send a message giving the number of measures given at every iteration
      call MPI_SEND(NumCtrl2SC, 1, MPI_INT, command_process_rank, com_tag, parent_comm, ierror)
      ! Receive the number of iterations
      call MPI_RECV(max_iter_com, 1, MPI_DOUBLE, command_process_rank, com_tag, parent_comm, statut, ierror)
      max_iter = INT(max_iter_com)
      print *, "Will receive MPI interface control for ", max_iter, " iterations"
   end if

   !Reception des variables venant du controleur DISCON
   !to_SC: [all_vars_for_turbine1, all_vars_for_turbine2, ..., all_vars_for_turbineM]
   do j = 1, nTurbines
      do i = 1, NumCtrl2SC
         to_interface((j-1)*NumCtrl2SC+i) = to_SC((j-1)*NumCtrl2SC+i)
         !print *, "interface value", to_interface((j-1)*NumCtrl2SC+i)
      end do
   end do

   if (initialized) then
      call MPI_RECV(yawRef, nTurbines+1, MPI_DOUBLE, command_process_rank, yaw_tag, parent_comm, statut, ierror)
      call MPI_RECV(pitchRef, nTurbines+1, MPI_DOUBLE, command_process_rank, pitch_tag, parent_comm, statut, ierror)
      call MPI_RECV(torqueRef, nTurbines+1  , MPI_DOUBLE, command_process_rank, torque_tag, parent_comm, statut, ierror)
      call MPI_SEND(to_interface, NumCtrl2SC*nTurbines, MPI_DOUBLE, command_process_rank, measures_tag, parent_comm, ierror)
      call MPI_BARRIER(parent_comm, ierror)
   end if

   ! Get control command from other MPI process
   !print *, 'Value of finalized', finalized
   if (num_iter == max_iter) then
      print *, 'Finalizing MPI Communication...'
      if (parent_comm .ne. MPI_COMM_WORLD) then
          ! Disconnect from intercommunicator
          call MPI_COMM_DISCONNECT(parent_comm, ierror)
      endif
      call MPI_FINALIZE(ierror)
   end if


   !Affectation des variables a communiquer au controleur DISCON
   !print * , "command activations", yawRef(1), pitchRef(1), torqueRef(1)
   do j = 1, nTurbines
      !print *, "command values", yawRef(j+1), pitchRef(j+1), torqueRef(j+1)
      from_SC((j-1)*NumSC2Ctrl+1) = yawRef(1) ! activateControl for yaw
      from_SC((j-1)*NumSC2Ctrl+2) = pitchRef(1) ! ... for pitch
      from_SC((j-1)*NumSC2Ctrl+3) = torqueRef(1) ! ... for torque
      from_SC((j-1)*NumSC2Ctrl+1+3) = yawRef(j+1)
      from_SC((j-1)*NumSC2Ctrl+2+3) = pitchRef(j+1)
      from_SC((j-1)*NumSC2Ctrl+3+3) = torqueRef(j+1)
   end do

   !Saturation en vitesse de yaw dans DISCON (inutilise pour le moment)
   from_SCglob(1) = 0.005235 !0.3deg/s
   num_iter = num_iter + 1


   return
end subroutine SC_CalcOutputs

!=======================================================================
!SUBROUTINE sc_updateStates (  ) BIND (C, NAME='sc_updateStates')
subroutine SC_UpdateStates ( t, nTurbines, nParamGlobal, paramGlobal, nParamTurbine, paramTurbine, nInpGlobal, to_SCglob, NumCtrl2SC, to_SC, &
                        nStatesGlobal, StatesGlob, nStatesTurbine, StatesTurbine, errStat, errMsg )  bind (C, NAME='sc_updateStates')


   ! This DLL super controller is used to implement a ...

   ! Modified by B. Jonkman to conform to ISO C Bindings (standard Fortran 2003) and
   ! compile with either gfortran or Intel Visual Fortran (IVF)
   ! DO NOT REMOVE or MODIFY LINES starting with "!DEC$" or "!GCC$"
   ! !DEC$ specifies attributes for IVF and !GCC$ specifies attributes for gfortran
   !
   ! Note that gfortran v5.x on Mac produces compiler errors with the DLLEXPORT attribute,
   ! so I've added the compiler directive IMPLICIT_DLLEXPORT.

   use, intrinsic :: ISO_C_Binding

   implicit                        none
#ifndef IMPLICIT_DLLEXPORT
!DEC$ ATTRIBUTES DLLEXPORT :: sc_updateStates
!GCC$ ATTRIBUTES DLLEXPORT :: sc_updateStates
#endif

   real(C_DOUBLE),         INTENT(IN   ) :: t                 !< time (s)
   integer(C_INT),         intent(in   ) :: nTurbines         !< number of turbines connected to this supercontroller
   integer(C_INT),         intent(in   ) :: nParamGlobal        !< number of global parameters for the supercontroller
   real(C_FLOAT),          intent(in   ) :: paramGlobal    (*)  !< global parameters for the supercontroller
   integer(C_INT),         intent(in   ) :: nParamTurbine        !< number of turbine-based parameters for supercontroller
   real(C_FLOAT),          intent(in   ) :: paramTurbine    (*)  !< turbine-based parameters for the supercontroller
   integer(C_INT),         intent(in   ) :: nInpGlobal        !< number of global inputs to supercontroller
   real(C_FLOAT),          intent(in   ) :: to_SCglob    (*)  !< global inputs to the supercontroller
   integer(C_INT),         intent(in   ) :: NumCtrl2SC        !< number of turbine controller outputs [inputs to supercontroller]
   real(C_FLOAT),          intent(in   ) :: to_SC        (*)  !< inputs to the super controller (from the turbine controller)
   integer(C_INT),         intent(in   ) :: nStatesGlobal     !< number of global states
   real(C_FLOAT),          intent(inout) :: StatesGlob   (*)  !< global states at time increment, n (total of nStatesGlobal of these states)
   integer(C_INT),         intent(in   ) :: nStatesTurbine    !< number of states per turbine
   real(C_FLOAT),          intent(inout) :: StatesTurbine(*)  !< turbine-dependent states at time increment, n (total of nTurbines*nStatesTurbine of these states)
   integer(C_INT),         intent(inout) :: errStat           !< error status code (uses NWTC_Library error codes)
   character(kind=C_CHAR), intent(inout) :: errMsg       (*)  !< Error Message from DLL to simulation code
   integer                               :: i
   real(C_FLOAT)                         :: sum
   ! Turbine-based inputs (one per turbine): to_SC
   ! 1 - GenTorque
   !
   ! Meaning of scOutputs
   ! 1 - Minimum Blade pitch

   ! No states

   return
end subroutine SC_UpdateStates

subroutine SC_End ( errStat, errMsg )  bind (C, NAME='sc_end')


   ! This DLL super controller is used to implement a...

   ! Modified by B. Jonkman to conform to ISO C Bindings (standard Fortran 2003) and
   ! compile with either gfortran or Intel Visual Fortran (IVF)
   ! DO NOT REMOVE or MODIFY LINES starting with "!DEC$" or "!GCC$"
   ! !DEC$ specifies attributes for IVF and !GCC$ specifies attributes for gfortran
   !
   ! Note that gfortran v5.x on Mac produces compiler errors with the DLLEXPORT attribute,
   ! so I've added the compiler directive IMPLICIT_DLLEXPORT.

   use, intrinsic :: ISO_C_Binding

   implicit                        none
#ifndef IMPLICIT_DLLEXPORT
!DEC$ ATTRIBUTES DLLEXPORT :: sc_end
!GCC$ ATTRIBUTES DLLEXPORT :: sc_end
#endif

   integer(C_INT),         intent(inout) :: errStat           !< error status code (uses NWTC_Library error codes)
   character(kind=C_CHAR), intent(inout) :: errMsg       (*)  !< Error Message from DLL to simulation code


   return
end subroutine SC_End
