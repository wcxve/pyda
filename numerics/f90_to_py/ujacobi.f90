!*************************************************************
!* This subroutine computes all eigenvalues and eigenvectors *
!* of a real symmetric square matrix A(N,N). On output, ele- *
!* ments of A above the diagonal are destroyed. D(N) returns *
!* the eigenvalues of matrix A. V(N,N) contains, on output,  *
!* the eigenvectors of A by columns. THe normalization to    *
!* unity is made by main program before printing results.    *
!* NROT returns the number of Jacobi matrix rotations which  *
!* were required.                                            *
!* --------------------------------------------------------- *
!* Ref.:"NUMERICAL RECIPES, Cambridge University Press, 1986,*
!*       chap. 11, pages 346-348" [BIBLI 08].                *
!*************************************************************
Subroutine Jacobi(A,N,D,V,NROT)
integer N,NROT
real*8  A(1:N,1:N),D(1:N),V(1:N,1:N)
real*8, pointer :: B(:), Z(:)
real*8  c,g,h,s,sm,t,tau,theta,tresh

allocate(B(1:100),stat=ialloc)
allocate(Z(1:100),stat=ialloc)

  do ip=1, N    !initialize V to identity matrix
    do iq=1, N
      V(ip,iq)=0.d0 
    end do
      V(ip,ip)=1.d0
  end do  
  do ip=1, N
    B(ip)=A(ip,ip)
    D(ip)=B(ip)
    Z(ip)=0.d0    
  end do
  NROT=0
  do i=1, 50
    sm=0.d0
    do ip=1, N-1     !sum off-diagonal elements
      do iq=ip+1, N
        sm=sm+DABS(A(ip,iq))
      end do
    end do
    if(sm==0.d0) return  !normal return
    if(i.lt.4) then
      tresh=0.2d0*sm**2
    else
      tresh=0.d0
    end if
    do ip=1, N-1
      do iq=ip+1, N
        g=100.d0*DABS(A(ip,iq))
! after 4 sweeps, skip the rotation if the off-diagonal element is small
        if((i.gt.4).and.(DABS(D(ip))+g.eq.DABS(D(ip))) &
          .and.(DABS(D(iq))+g.eq.DABS(D(iq)))) then
          A(ip,iq)=0.d0
        else if(DABS(A(ip,iq)).gt.tresh) then
          h=D(iq)-D(ip)
          if(DABS(h)+g.eq.DABS(h)) then
            t=A(ip,iq)/h
          else
            theta=0.5d0*h/A(ip,iq)  
            t=1.d0/(DABS(theta)+DSQRT(1.d0+theta**2))
            if(theta.lt.0.d0) t=-t
          end if
          c=1.d0/DSQRT(1.d0+t**2)
          s=t*c
          tau=s/(1.d0+c)
          h=t*A(ip,iq)
          Z(ip)=Z(ip)-h
          Z(iq)=Z(iq)+h
          D(ip)=D(ip)-h
          D(iq)=D(iq)+h
          A(ip,iq)=0.d0
          do j=1, ip-1
            g=A(j,ip)
            h=A(j,iq)
            A(j,ip)=g-s*(h+g*tau)
            A(j,iq)=h+s*(g-h*tau)
          end do
          do j=ip+1, iq-1
            g=A(ip,j)
            h=A(j,iq)
            A(ip,j)=g-s*(h+g*tau)
            A(j,iq)=h+s*(g-h*tau)
          end do              
          do j=iq+1, N
            g=A(ip,j)
            h=A(iq,j)
            A(ip,j)=g-s*(h+g*tau)
            A(iq,j)=h+s*(g-h*tau)
          end do          
          do j=1, N
            g=V(j,ip)
            h=V(j,iq)
            V(j,ip)=g-s*(h+g*tau)
            V(j,iq)=h+s*(g-h*tau)
          end do          
          NROT=NROT+1
        end if !if ((i.gt.4)...
      end do !main iq loop
    end do !main ip loop
    do ip=1, N
      B(ip)=B(ip)+Z(ip)
      D(ip)=B(ip)
      Z(ip)=0.d0
    end do
  end do !main i loop
  pause ' 50 iterations !'
  return
END    

! end of file ujacobi.f90

