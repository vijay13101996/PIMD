
program xfunc
	use stadium_billiards, only: potential,potential_assignment	
	use array_tools, only: linspace
	implicit none
	integer :: switch,i,lenx=100, leny=100
	integer, parameter :: dp = kind(1.d0)
	real(dp) :: x,y,Le,r_c,pot
	real(dp), dimension(10000,10000) :: pot_mat
	real(dp), dimension(10000) :: x_arr
	real(dp), dimension(10000) :: y_arr
	switch =1
	if(switch == 1) then
		x = 5
		y = 5
		Le = 1.0
		r_c = 1.0
		pot = 0.0
		x_arr = linspace(-2.0,2.0,10000)
		y_arr = linspace(-2.0,2.0,10000)
		!print*, "x_arr = ", x_arr
		call potential(x,y,Le,r_c,pot)
		call potential_assignment(x_arr,10000,y_arr,10000,Le,r_c,pot_mat)
		
		print*, "potential =", pot_mat(50,50) !potential(x,y,pot)
        end if
end program xfunc
