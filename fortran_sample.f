
 
function potential(x,y,Le,r_c,pot)
	real, intent(in) :: x ! input
	real, intent(in) :: y ! input
	real, intent(in) :: Le ! input
	real, intent(in) :: r_c ! input
!	real, intent(out) :: pot ! output
!	if(x**2 + y**2 <= 1.0) then
!		pot = x**2 + y**2
!	else
!		pot = 1.0
!	end if

	print*, "Here"
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
end function potential

program xfunc
	implicit none
	integer :: switch
	real :: x
	real :: y
	real :: Le
	real :: r_c
	real :: pot
	real :: potential
	
    
	switch =1
	if(switch == 1) then
		x = 5
		y = 5
		Le = 1.0
		r_c = 1.0
		pot = 0.0
		pot = potential(x,y,Le,r_c,pot)
		print*, "potential =", pot !potential(x,y,pot)
        end if
end program xfunc



