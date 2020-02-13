module array_tools
	implicit none
	private
	public :: linspace
	contains
		function linspace(init,fin,num)
			real :: space
			real, intent(in) :: init,fin
			integer,intent(in) :: num
			real, dimension(num) :: linspace
			integer :: i
		
			space = (fin-init)/(num-1)
			do i=1,num
				linspace(i) = init + space*(i-1)
			end do
		end function linspace

end module array_tools		
