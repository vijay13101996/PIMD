
C FILE: FIB1.F
      SUBROUTINE FIB(A,N)
C
C     CALCULATE FIRST N FIBONACCI NUMBERS
C
      INTEGER N
      REAL*8 A(N)
      DO I=1,N
         IF (I.EQ.1) THEN
            A(I) = 0.0D0
         ELSEIF (I.EQ.2) THEN
            A(I) = 1.0D0
         ELSE 
            A(I) = A(I-1) + A(I-2)
         ENDIF
      ENDDO
      END
C END FILE FIB1.F


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