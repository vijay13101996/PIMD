
module stadium_billiards	
	implicit none
	!integer, parameter :: dp = kind(1.0d0)
	!integer, parameter :: dp = selected_real_kind(15, 307)
	private
	public ::potential
	public :: potential_assignment
	
	contains
		subroutine potential(x,y,Le,r_c,pot)
			real(kind=8), intent(in) :: x,y,Le,r_c ! input
			real(kind=8), intent(inout) :: pot ! output
		
		if (.false.) then	
			if(x**2 + y**2 <= 1.0) then
				pot = x**2 + y**2
			else
				pot = 1.0
			end if
		end if
			
		if (.true.) then
			if(abs(x)<Le/2 .and. abs(y)<r_c) then 
				pot = 0.0 
			else if(abs(x)<(Le/2+r_c) .and. abs(y)<r_c) then
				if(x>0) then
			    		if(((x-Le/2)**2+y**2)<r_c**2) then
						pot = 0.0
			     		else 
						pot = 1e5
					end if
				else
			    		if(((x+Le/2)**2+y**2)<r_c**2) then
						pot = 0.0
			    		else
						pot = 1e5
					end if
				end if
			else
				pot= 1e5
			end if
		end if		
		end subroutine potential

		subroutine potential_assignment(X,lenx,Y,leny,Le,r_c,POT)
			integer :: i,j
			integer :: lenx,leny
			real(kind=8), dimension(lenx), intent(in) :: X
			real(kind=8), dimension(leny), intent(in) :: Y
			real(kind=8), intent(in) :: Le,r_c
			real(kind=8), dimension(lenx, leny), intent(inout) :: POT
			!f2py real(kind=8), dimension(lenx, leny), intent(in,out,copy) :: POT
			!lenx = size(X)
			!leny = size(Y)
			do i = 1,lenx
				do j = 1,leny
					call potential(X(i),Y(j),Le,r_c,POT(j,i))
				end do
			end do
		end subroutine potential_assignment
end module stadium_billiards



