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
   NumCtrl2SC        = 2
   NumParamGlobal    = 1
   NumParamTurbine   = 3
   NumStatesGlobal   = 0
   NumStatesTurbine  = 0
   NumSC2CtrlGlob    = 0
   NumSC2Ctrl        = 1


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
      from_SC(j) = real( 10.0 * sin(ParamTurbine(j)*0.0), C_FLOAT)
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
use algoCont !appel du module

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
   !real(C_FLOAT),          intent(inout) :: wind      (*)  !< turbine specific outputs of the super controller (to the turbine controller)
   real(C_DOUBLE)          :: wind_tmp        !< turbine specific outputs of the super controller (to the turbine controller)
   
   logical(C_BOOL) :: adaptive_tau
   logical(C_BOOL) :: zap_learning
   real(C_DOUBLE) :: n_turbines
   real(C_DOUBLE) :: time
   real(C_DOUBLE) :: ws(nTurbines) !3
   real(C_DOUBLE) :: power(nTurbines) !3
   real(C_DOUBLE) :: start_time
   real(C_DOUBLE) :: u_inf
   real(C_DOUBLE) :: dt
   real(C_DOUBLE) :: wait_time
   real(C_DOUBLE) :: yaw_init(nTurbines) !3
   REAL(C_DOUBLE) :: yawRef(nTurbines) !3
   REAL(C_DOUBLE) :: old_prod(nTurbines) !3
   REAL(C_DOUBLE) :: new_prod(nTurbines) !3
   
   ! For bobyqa
   real(C_DOUBLE) :: propagation_delay
   !CHARACTER(len=40) ::  com_file_yaws    = 'com_yaws.csv'//achar(0)
   !CHARACTER(len=40) ::  com_file_power    = 'com_power.csv'//achar(0)
   
   ! For anyalgo interface
   !CHARACTER(len=100) ::  com_file_yaws    = 'C:\\Users\\bizonmoc\\Documents\\rl-farmcontrol\\simul\\FF_dec_A2C\\com_files\\yaws.csv'//achar(0)
   !CHARACTER(len=100) ::  com_file_power    = 'C:\\Users\\bizonmoc\\Documents\\rl-farmcontrol\\simul\\FF_dec_A2C\\com_files\\power.csv'//achar(0)
   !CHARACTER(len=100) ::  com_file_wind    = 'C:\\Users\\bizonmoc\\Documents\\rl-farmcontrol\\simul\\FF_dec_A2C\\com_files\\wind.csv'//achar(0)
   CHARACTER(len=100) ::  com_file_yaws    = 'L:\\Users\\bizonmoc\\Documents\\rl-farmcontrol-fast\\simul\\temp2\\com_files\\yaws.csv'//achar(0)
   CHARACTER(len=100) ::  com_file_power    = 'L:\\Users\\bizonmoc\\Documents\\rl-farmcontrol-fast\\simul\\temp2\\com_files\\power.csv'//achar(0)
   CHARACTER(len=100) ::  com_file_wind    = 'L:\\Users\\bizonmoc\\Documents\\rl-farmcontrol-fast\\simul\\temp2\\com_files\\wind.csv'//achar(0)
   
   ! For MPI communication
   integer process_Rank, size_Of_Cluster, ierror
   integer initialized, finalized
   integer message
   integer command_process_rank
   integer yaw_tag, power_tag, wind_tag

   integer, dimension(MPI_STATUS_SIZE) :: statut
   
   
   initialized = 0
   finalized = 0
   yaw_tag = 0
   power_tag = 1
   wind_tag = 2
   command_process_rank = 1
   call MPI_INITIALIZED(initialized, ierror)  
   if (int(t)==0) then
      print *, 'Initialize MPI'
      call MPI_INIT(ierror)
   end if

   if (t==1200) then
      print *, 'Finalize MPI'
      call MPI_FINALIZE(ierror)
      finalized = 1
   end if
   
   ! Define config
   !type (config) :: algo_config

   !Prsent dans le code initial (ne sert  rien  priori)
   do j = 1, nTurbines
      do i = 1, NumSC2Ctrl
         from_SC((j-1)*NumSC2Ctrl+i) = 0 !real(0.05 * sin(paramTurbine(j) * t)) ! The yaw is the yaw rate times the time
      end do
   end do
   !print *, 'FORTRAN number of global controllers', NumSC2CtrlGlob, ' Number of turbine specfic controllers: ', NumSC2Ctrl

   !Rception des variables venant du controleur DISCON
   do j = 1, nTurbines
      power(j) = to_SC(2*j)
      ws(j) = to_SC(2*j - 1)
   end do   
   !Parametrage de la fonction   
   !n_turbines = nTurbines
   !time = t
   !start_time = 1800.0
   !u_inf = 8.0 
   !dt = 3 !
   !wait_time = 1041
   !do j = 1, nTurbines
   !   yaw_init(j) = 0 !en deg
   !end do
   ! FLORIS init ! [28. 27. -1.]
   ! FLORIS init - wind_shear = 0.0 [30, 23, 0]
   ! FLORIS init q learning [31, 24, -1]
   ! FLORIS init q learning polynom [30, 24, 0]
   ! FLORIS init q learning 6 turbines [29, 25, 0, 28, 24, 0]
   ! bobyqa 0 turb init [33, 33, 0]
   ! standard init : [0, 0, 0]
   !yaw_init(1) = 0 !en deg
   !yaw_init(2) = 0! 27 !23 !27 ! 0
   !yaw_init(3) = 0 !0 !-1
   !yaw_init(4) = 28 ! 0 !en deg
   !yaw_init(5) = 24 ! 0
   !yaw_init(6) = 0
   !zap_learning = .false.
   !adaptive_tau = .false.
    ! For bobyqa
   !propagation_delay = 2500
   ! Just for test cases
   !propagation_delay = 200
   !algo_config = config(0)
   
   !Appel de l'algo de controle
   ! With control:
   !call fctControl(zap_learning,adaptive_tau,n_turbines,time,ws,power,start_time,u_inf,dt,wait_time,yaw_init,yawRef,old_prod,new_prod) !q_learning
   !call bobyqaControl(com_file_yaws,com_file_power,power,time,yaw_init,propagation_delay,yawRef)
   !call anyAlgoControl(n_turbines, power, ws, time, com_file_yaws, com_file_power, com_file_wind, yawRef)
   !call lutControl(ws, yawRef)
   
   ! Get control command from other MPI process
   if (initialized) then
      !call MPI_COMM_SIZE(MPI_COMM_WORLD, size_Of_Cluster, ierror)
      !call MPI_COMM_RANK(MPI_COMM_WORLD, process_Rank, ierror)
      !print *, 'FORTRAN process at time', int(t), 'Process', process_Rank, 'of ', size_Of_Cluster
      call MPI_RECV(yawRef, nTurbines, MPI_DOUBLE, command_process_rank, yaw_tag, MPI_COMM_WORLD, statut, ierror)
      call MPI_BARRIER(MPI_COMM_WORLD, ierror)
      !print *, '***** Received output, yaw ', yawRef(1), 'error: ', ierror
      call MPI_SEND(power, nTurbines, MPI_DOUBLE, command_process_rank, power_tag, MPI_COMM_WORLD, ierror)
      !print *, '***** Sent output, power: ', power(1), 'error: ', ierror
      call MPI_SEND(ws, nTurbines, MPI_DOUBLE, command_process_rank, wind_tag, MPI_COMM_WORLD, ierror)
      !print *, '***** Sent output, ws: ', ws(1), 'error: ', ierror
      call MPI_BARRIER(MPI_COMM_WORLD, ierror)
   end if

   
   ! Without control:
   !do j = 1, nTurbines
   !   yawRef(j) = yaw_init(j) / 180.0 * asin(1.0) * 2.0 !en rad
   !end do
   
   
   !!Affectation des variables  communiquer au contrleur DISCON
   !print *, 'Sending variables to DISCON'
   do j = 1, nTurbines
      from_SC(j) = yawRef(j)
   end do

   !Saturation en vitesse de yaw dans DISCON (inutilise pour le moment) 
   from_SCglob(1) = 0.005235 !0.3deg/s 
   
   !Ecriture d'un fichier debug
   !OPEN  (456, FILE='windSC.dat', STATUS='old',position="append")      
   !!WRITE (456,*)(yawRef(i),achar(9),i=1,3) !boucle pour crire les 3 variables
   !WRITE (456,*) t
   !WRITE (456,*) yawRef(1),yawRef(2),yawRef(3)
   !WRITE (456,*) power(1),power(2),power(3)
   !WRITE (456,*) ws(1),ws(2),ws(3)
   !WRITE (456,*) old_prod(1),old_prod(2),old_prod(3)
   !WRITE (456,*) new_prod(1),new_prod(2),new_prod(3)
   !close (456) 
     
   !from_SC(1) = 0
   !from_SC(2) = 0
   !from_SC(3) = 0
   
   !if ( t >= 2500 ) then
   !    from_SC(1) = 0.523598775598299 !30
   !    from_SC(2) = 0
   !    from_SC(3) = 0
   !end if    
      
   !opti FLORIS Ti=8% 4D    	
   !from_SC(1) = 0.521228292097520
   !from_SC(2) = 0.400952864159124
   !from_SC(3) = 0
   
   !opti FLORIS Ti=8% 8D  
   !from_SC(1) = 0.377538614254492
   !from_SC(2) = 0.193676490560116
   !from_SC(3) = 0
 

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
