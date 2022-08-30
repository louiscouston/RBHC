PROGRAM genhpts 
    IMPLICIT NONE
    REAL(kind=8) :: xmax,ymax,xp,yp,xmin,ymin,eps,pi,dx
    INTEGER :: ip,nx,nf,no,i,n,nz,nb 
    CHARACTER*10 :: char

    pi=4.0d0*atan(1.0d0)

    xmax=4.0d0
    ymax=1.0d0

! xmin and ymin added by me    
    xmin=-4.0d0 
    ymin=0.0d0

! Thickness of the boundary probes region
    dx=0.000001d0

! Small correction to ensure that the probe is inside the domain
    eps=1.0e-8
   
! Resolution along the horizontal
    nx = 81
! Resolution along the vertical
    nz = 51

! Number of normal points for boundary probes
    nb = 3

! nx*nz gives the total number of history points' 
    write(66,*) nx*nz+3*nx*2+3*nz*2 

! Bulk probes
    ip = 0    
    do n=1,nx  
     xp = xmin+eps+dble(n-1)*(xmax-xmin-2.0d0*eps)/dble(nx-1)    
! Mapping to concentrate probes close to the boundaries
    ! xp = xmax*sin(pi*xp/(2.0d0*xmax))
     do i=1,nz
      yp = ymin+eps+dble(i-1)*(ymax-ymin-2.0d0*eps)/dble(nz-1)
! Mapping to concentrate probes close to the boundaries
     ! yp = ymax*0.5d0*(1.0d0-cos(pi*yp/(ymax)))
      ip = ip+1
      write(66,*) xp,yp
     enddo
    enddo

! Boundary probes
    do n=1,nx
     xp = xmin+eps+dble(n-1)*(xmax-xmin-2.0d0*eps)/dble(nx-1)    
! Mapping to concentrate probes close to the boundaries
    ! xp = xmax*sin(pi*xp/(2.0d0*xmax))
     do i=1,nb
      yp = ymin+eps+dble(i-1)*(dx-eps)/dble(nb-1)
      ip = ip+1
      write(66,*) xp,yp
     enddo
     do i=1,nb
      yp = ymax-eps-dble(i-1)*(dx+eps)/dble(nb-1)
      ip = ip+1
      write(66,*) xp,yp
     enddo
    enddo
   do i=1,nz
     yp = ymin+eps+dble(i-1)*(ymax-ymin-2.0d0*eps)/dble(nz-1)
! Mapping to concentrate probes close to the boundaries
    ! yp = ymax*0.5d0*(1.0d0-cos(pi*yp/(ymax)))
     do n=1,nb
      xp = xmin+eps+dble(n-1)*(dx-eps)/dble(nb-1)
      ip = ip+1
      write(66,*) xp,yp
     enddo
     do n=1,nb
      xp = xmax-eps-dble(n-1)*(dx+eps)/dble(nb-1)
      ip = ip+1
      write(66,*) xp,yp
     enddo
    enddo

   if (ip.ne.nx*nz+3*nx*2+3*nz*2) then
     STOP 'Wrong number of probes'
   else
     write(*,*) ' OK!'
   endif

END PROGRAM genhpts
