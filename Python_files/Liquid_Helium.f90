

module Liquid_Helium_potential
        implicit none 
        private 
        !public :: sample_gaussian
        !public :: HFD_B2
        public ::scalar_dpotential
        public ::scalar_ddpotential
        public ::dpotential
        public ::vector_ddpotential
        public ::identity
        public ::ddpotential
        public ::assign_pot_params
        real(kind=8) :: m,D,A_s,beta_s,R_m,C_6,C_8,C_10,alp_s,eps  
        
        common /potential_parameters/  m,D,A_s,beta_s,R_m,C_6,C_8,C_10,alp_s,eps  
        
        contains
               
               subroutine assign_pot_params()
                        real(kind=8) ::  m,D,A_s,beta_s,R_m,C_6,C_8,C_10,alp_s,eps  
                        common /potential_parameters/  m,D,A_s,beta_s,R_m,C_6,C_8,C_10,alp_s,eps  
                      
                        m=4.0  !Has to be changed; It is very risky to define it as a global variable.
                        D=1.4135
                        A_s = 1.9221529e5 
                        beta_s = -1.89296514
                        R_m = 0.2970*18.897259885789
                        C_6= 1.34920045
                        C_8= 0.41365922
                        C_10= 0.17078164
                        alp_s = 10.73520708
                        eps = 10.94*0.31668115634e-5   !k_B = 0.31668115634e-5 a.u./K   # 1/k_B = 315775.02481 in atomic units (a.u./K)
               
               end subroutine assign_pot_params

               function identity(n_dim) result(id)
                       integer, intent(in) :: n_dim
                       real(kind=8), dimension(n_dim,n_dim) :: id
                       integer :: i,j
                       id=0.0
                       do i=1,n_dim
                        do j=1,n_dim
                                if(i .eq. j) then
                                        id(i,j)=1.0
                                end if
                        end do
                       end do
               end function identity

               subroutine scalar_dpotential(R,sdpot)
                       real(kind=8), intent(in) :: R
                       !f2py real(kind=8),intent(out,copy) :: sdpot
                       real(kind=8),intent(out) :: sdpot
                       real(kind=8) ::  m,D,A_s,beta_s,R_m,C_6,C_8,C_10,alp_s,eps  
                       common /potential_parameters/  m,D,A_s,beta_s,R_m,C_6,C_8,C_10,alp_s,eps  
                       !print*, 'm',m
                       !sdpot = 2*R*exp(-R**2)
                              
                        if(R.le.D) then
                            sdpot = eps*(A_s*(2*R*beta_s/R_m**2 - alp_s/R_m)*exp(R**2*beta_s/R_m**2 - R*alp_s/R_m)&
                                    + 2*D*R_m*(D*R_m/R -1)*(-C_10*R_m**10/R**10 &
                                    - C_6*R_m**6/R**6 - C_8*R_m**8/R**8)*exp(-(D*R_m/R - 1)**2)/R**2&
                                    + (10*C_10*R_m**10/R**11 + 6*C_6*R_m**6/R**7 + 8*C_8*R_m**8/R**9)*exp(-(D*R_m/R - 1)**2))
                            
                        else
                            sdpot = eps*(A_s*(2*R*beta_s/R_m**2 - alp_s/R_m)*exp(R**2*beta_s/R_m**2 - R*alp_s/R_m) &
                                    + 10*C_10*R_m**10/R**11 + 6*C_6*R_m**6/R**7 + 8*C_8*R_m**8/R**9)
                        end if
                end subroutine scalar_dpotential

                subroutine scalar_ddpotential(R,sddpot)

                       real(kind=8), intent(in) :: R
                       !f2py real(kind=8),intent(out,copy) :: sddpot
                       real(kind=8),intent(out) :: sddpot
                       real(kind=8) ::  m,D,A_s,beta_s,R_m,C_6,C_8,C_10,alp_s,eps  
                       common /potential_parameters/  m,D,A_s,beta_s,R_m,C_6,C_8,C_10,alp_s,eps  
                       !sddpot = (2-4*R**2)*exp(-R**2)

                        if(R .le. D) then
                            sddpot= eps*(2*A_s*beta_s*exp(R*(R*beta_s/R_m - alp_s)/R_m)/R_m**2 + &
                                    A_s*(2*R*beta_s/R_m -alp_s)**2*exp(R*(R*beta_s/R_m - alp_s)/R_m)/R_m**2 - &
                                    4*D**2*R_m**8*(D*R_m/R - 1)**2*(C_10*R_m**4/R**4 + C_6 + C_8*R_m**2/R**2)*exp(-(D*R_m/R -&
                            1)**2)/R**10 + 2*D**2*R_m**8*(C_10*R_m**4/R**4 + &
                                    C_6 + C_8*R_m**2/R**2)*exp(-(D*R_m/R - 1)**2)/R**10 + 4*D*R_m**7*(D*R_m/R - 1)& 
                                    *(C_10*R_m**4/R**4 + C_6 + C_8*R_m**2/R**2)*exp(-(D*R_m/R - 1)**2)/R**9 + &
                            8*D*R_m**7*(D*R_m/R - 1)*(5*C_10*R_m**4/R**4 + 3*C_6 + 4*C_8*R_m**2/R**2)*exp(-(D*R_m/R - 1)**2)/R**9 -&
                    2*R_m**6*(55*C_10*R_m**4/R**4 + 21*C_6 + 36*C_8*R_m**2/R**2)*exp(-(D*R_m/R - 1)**2)/R**8)
                            
                        else
                            sddpot= -eps*(-2*A_s*beta_s*exp(R*(R*beta_s/R_m - alp_s)/R_m)/R_m**2 -&
                                    A_s*(2*R*beta_s/R_m - alp_s)**2*exp(R*(R*beta_s/R_m - alp_s)/R_m)/R_m**2 &
                                    + 110*C_10*R_m**10/R**12 + 42*C_6*R_m**6/R**8 + 72*C_8*R_m**8/R**10)
                        end if
                end subroutine scalar_ddpotential

                subroutine dpotential(q,lenq1,lenq2,dpot)
                      integer, intent(in) ::lenq1, lenq2
                      real(kind=8), dimension(lenq1,lenq2), intent(in) ::q
                      !f2py real(kind=8), dimension(lenq1,lenq2), intent(out,copy) ::dpot
                      real(kind=8), dimension(lenq1,lenq2), intent(out) :: dpot
                      integer :: i,j
                      real(kind=8) ::R,sdpot
                      real(kind=8), dimension(lenq2) :: unitv
                      !print*, 'Here'
                      dpot = 0.0*dpot
                      do i=1,lenq1
                        do j=1,lenq1
                                if(i .NE. j) then
                                        R = sum( (q(i,1:lenq2)-q(j,1:lenq2))**2 )**0.5
                                        !print*,'R=',R
                                        unitv = (q(i,1:lenq2)-q(j,1:lenq2))/R
                                        sdpot = 0.0
                                        call scalar_dpotential(R,sdpot)
                                        dpot(i,1:lenq2) = dpot(i,1:lenq2) + sdpot*unitv
                                end if
                        end do
                      end do                                
                end subroutine dpotential

                subroutine vector_ddpotential(R_vector,n_dim,R,vectddpot)
                       integer, intent(in) :: n_dim
                       real(kind=8), dimension(n_dim), intent(in) :: R_vector
                       real(kind=8), intent(in):: R
                       !f2py real(kind=8), dimension(n_dim,n_dim), intent(out,copy) ::vectddpot
                       real(kind=8), dimension(n_dim,n_dim), intent(out) :: vectddpot
                   
                       vectddpot = 0.0*vectddpot
                       vectddpot = R**2*identity(n_dim) - matmul(reshape(R_vector,(/n_dim,1/)) , reshape(R_vector,(/1,n_dim/)))   ! matmul(R_vector,R_vector)
                       !print*,'outer product', matmul(reshape(R_vector,(/n_dim,1/)) , reshape(R_vector,(/1,n_dim/))) 
                       vectddpot = vectddpot/R**3

                end subroutine vector_ddpotential

                subroutine ddpotential(q,n_particles,n_dim,ddpot)
                        integer, intent(in) :: n_particles, n_dim
                        real(kind=8), dimension(n_particles,n_dim), intent(in) :: q
                        !f2py real(kind=8),dimension(n_particles,n_dim,n_dim), intent(out,copy) :: ddpot
                        real(kind=8),dimension(n_particles,n_dim,n_dim), intent(out) :: ddpot
                        integer :: i,j
                        real(kind=8), dimension(n_dim) :: R_vector, unitv
                        real(kind=8) :: R,sddpot, sdpot
                        real(kind=8), dimension(n_dim,n_dim) :: vectddpot
                        ddpot = 0.0
                        do i =1,n_particles
                                do j=1,n_particles
                                    if(i .ne. j) then 
                                        R_vector = q(i,1:n_dim)-q(j,1:n_dim)
                                        R = sum(R_vector**2)**0.5
                                        unitv = R_vector/R
                                        sddpot = 0.0
                                        call scalar_ddpotential(R,sddpot)
                                        sdpot = 0.0
                                        call scalar_dpotential(R,sdpot) 
                                        vectddpot = 0.0
                                        call vector_ddpotential(R_vector,n_dim,R,vectddpot)
                                        ddpot(i,1:n_dim,1:n_dim) =  ddpot(i,1:n_dim,1:n_dim) +&
      sddpot*matmul(reshape(unitv,(/n_dim,1/)) , reshape(unitv,(/1,n_dim/))) + sdpot*vectddpot 
                                    end if
                                end do
                        end do
                end subroutine ddpotential

end module Liquid_Helium_potential 
