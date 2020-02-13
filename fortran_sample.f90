
module stadium_billiards
	implicit none
	private
	public ::potential
	public :: potential_assignment
	contains
		subroutine potential(x,y,Le,r_c,pot)
			real, intent(in) :: x ! input
			real, intent(in) :: y ! input
			real, intent(in) :: Le ! input
			real, intent(in) :: r_c ! input
			real, intent(inout) :: pot ! output
		
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
				!print*, "Hello"
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
				!print*, "Hi hi,pot", pot
			end if
		end if		
		end subroutine potential

		subroutine potential_assignment(X,lenx,Y,leny,Le,r_c,POT)
			integer :: i,j,lenx,leny			
			real, dimension(lenx), intent(in) :: X
			real, dimension(leny), intent(in) :: Y
			real, dimension(lenx,leny), intent(inout) :: POT
			real :: Le
			real :: r_c
			
			do i = 1,lenx
				do j = 1,leny
					call potential(X(i),Y(j),Le,r_c,POT(i,j))
				end do
			end do
		end subroutine potential_assignment
end module stadium_billiards



