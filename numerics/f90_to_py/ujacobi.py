#*************************************************************
#* This subroutine computes all eigenvalues and eigenvectors *
#* of a real symmetric square matrix A(N,N). On output, ele- *
#* ments of A above the diagonal are destroyed. D(N) returns *
#* the eigenvalues of matrix A. V(N,N) contains, on output,  *
#* the eigenvectors of A by columns. THe normalization to    *
#* unity is made by main program before printing results.    *
#* NROT returns the number of Jacobi matrix rotations which  *
#* were required.                                            *
#* --------------------------------------------------------- *
#* Ref.:"NUMERICAL RECIPES, Cambridge University Press, 1986,*
#*       chap. 11, pages 346-348" [BIBLI 08].                *
#*************************************************************
def Jacobi(A,N,D,V,NROT):
#integer N,NROT
#real*8  A(1:N,1:N),D(1:N),V(1:N,1:N)
#real*8, pointer :: B(:), Z(:)
#real*8  c,g,h,s,sm,t,tau,theta,tresh

#allocate(B(1:100),stat=ialloc)
#allocate(Z(1:100),stat=ialloc)

  for ip in range(0, N): #initialize V to identity matrix
    for iq in range(0, N):
      V[ip,iq]=0.0 
    
      V[ip,ip]=1.0
    
  for ip in range(0, N):
    B[ip]=A[ip,ip]
    D[ip]=B[ip]
    Z[ip]=0.0    
  
  NROT=0
  for i in range(0, 50):
    sm=0.0
    for ip in range(0, N-1): #sum off-diagonal elements
      for iq in range(ip+1, N):
        sm=sm+abs(A[ip,iq])
      
    
    if(sm==0.0) return  #normal return
    if(i < 4) :
      tresh=0.20*sm**2
    else:
      tresh=0.0
    ## end if
    for ip in range(0, N-1):
      for iq in range(ip+1, N):
        g=100.0*abs(A[ip,iq])
# after 4 sweeps, skip the rotation if the off-diagonal element is small
        if((i > 4) and (abs(D[ip])+g == abs(D[ip])) \
           and (abs(D[iq])+g == abs(D[iq]))) :
          A[ip,iq]=0.0
        elif(abs(A[ip,iq]) > tresh) :
          h=D[iq]-D[ip]
          if(abs(h)+g == abs(h)) :
            t=A[ip,iq]/h
          else:
            theta=0.50*h/A[ip,iq]  
            t=1.0/(abs(theta)+math.sqrt(1.0+theta**2))
            if(theta < 0.0) t=-t
          ## end if
          c=1.0/math.sqrt(1.0+t**2)
          s=t*c
          tau=s/(1.0+c)
          h=t*A[ip,iq]
          Z[ip]=Z[ip]-h
          Z[iq]=Z[iq]+h
          D[ip]=D[ip]-h
          D[iq]=D[iq]+h
          A[ip,iq]=0.0
          for j in range(0, ip-1):
            g=A[j,ip]
            h=A[j,iq]
            A[j,ip]=g-s*(h+g*tau)
            A[j,iq]=h+s*(g-h*tau)
          
          for j in range(ip+1, iq-1):
            g=A[ip,j]
            h=A[j,iq]
            A[ip,j]=g-s*(h+g*tau)
            A[j,iq]=h+s*(g-h*tau)
                        
          for j in range(iq+1, N):
            g=A[ip,j]
            h=A[iq,j]
            A[ip,j]=g-s*(h+g*tau)
            A[iq,j]=h+s*(g-h*tau)
                    
          for j in range(0, N):
            g=V[j,ip]
            h=V[j,iq]
            V[j,ip]=g-s*(h+g*tau)
            V[j,iq]=h+s*(g-h*tau)
                    
          NROT=NROT+1
        ## end if #if ((i > 4)...
       #main iq loop
     #main ip loop
    for ip in range(0, N):
      B[ip]=B[ip]+Z[ip]
      D[ip]=B[ip]
      Z[ip]=0.0
    
   #main i loop
  pause ' 50 iterations #'
  return
# end    

# end of file ujacobi.f90

