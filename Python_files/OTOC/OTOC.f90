


module position_matrix	
        implicit none
        private
        public :: pos_matrix_elts 

        contains

                subroutine pos_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n,k,pos_mat_elt)
                        integer, intent(in) :: n,k , len1vecs,len2vecs,lenx			
			real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
			real(kind=8), dimension(lenx), intent(in) :: x_arr
			real(kind=8), intent(in) ::  dx,dy
                        integer :: i,j
			real(kind=8), intent(inout) :: pos_mat_elt

                        do i = 1,len1vecs
                                do j = 1,len2vecs
                                        pos_mat_elt = pos_mat_elt + vecs(i,n+1)*vecs(i,k+1)*x_arr(i)*dx*dy
                                end do
                        end do

                end subroutine pos_matrix_elts

end module position_matrix
