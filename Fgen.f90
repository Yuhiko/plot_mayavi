!!! CAUTION !!!!
!! Fortran doesn't have capital letter. So, don't use Capital letter in the function name


! program testrun
!   implicit none
!   integer(4),parameter::im=8,jm=10,km=12,  nxx=im
!   integer(4):: ii,jj,kk
!   real(8),dimension(km,jm,im)::vec,dvdx
!   integer(4),dimension(km,jm,im)::reflev
!   real(8)::xx(nxx)
  

!   reflev = 0
!   do ii = 1,im
!      do jj = 1,jm
!         do kk = 1,km
!            vec(kk,jj,ii) = kk + jj*2 + ii*3
!            if (kk>km/3 .and. kk<km/3*2)reflev(kk,jj,ii)= 1
!         enddo
!         print*, jj,ii,vec(:,jj,ii)
!      enddo
!   enddo
  
!   do kk = 1,nxx
!      xx(kk) = kk
!   enddo
!   call diff_f90(km,jm,im,nxx,3,maxval(reflev),vec,xx,2**reflev,.false.,dvdx)

!   print*
!   do ii = 1,im
!      do jj = 1,jm
!         print*, jj,ii,dvdx(:,jj,ii)
!      enddo
!   enddo
  
!   stop
! end program testrun



!!! the input vector should be transposed in the python
!!! The transpose procedure doens't change the memory allocation, only change the pointer
subroutine diff_f90(val,reflev,xx,dim,lmax, flg_period,nx3,nx2,nx1,nxx, dvdx)
  implicit none
  real(8),parameter:: PI2 = 3.14159265358979323846d0 * 2.
  integer(4),intent(IN):: nx1,nx2,nx3,dim,nxx,lmax
  real(8),intent(IN),dimension(nx1,nx2,nx3):: val
  real(8),intent(IN),dimension(nxx)::  xx
  integer(4),intent(IN),dimension(nx1,nx2,nx3):: reflev
  logical,intent(IN)::flg_period
  
  real(8),intent(OUT),dimension(nx1,nx2,nx3):: dvdx
  integer(4):: ml, ii,jj,kk, indexp, indexm
  !integer(4):: item, tt
  integer(4),dimension(nx1,nx2,nx3):: steps
  !real(8),dimension(nx1,nx2,nx3):: xuse, vuse
  !real(8)::xtem, vtem
  
  ! integer(4),dimension(nx1,nx2,nx3):: isteps=0, jsteps=0, ksteps=0
  ! real(8),dimension(nx1,nx2,nx3):: dx
  ! integer(4)::is=1,js=1,ks=1, ie=nx1,je=nx2,ke=nx3, it1,it2,jt1,jt2,kt1,kt2
  ml    = 2**lmax
  steps = 2**(lmax-reflev)
  dvdx  = 0d0
  
  
  if (dim == 1) then
     !! xuse procedure
     ! kk=1
     ! do jj=1,nx2
     !    do ii=1,nx1
     !       do while (kk<nx3)
     !          xtem=0d0
     !          vtem=0d0
     !          item=steps(ii,jj,kk) -1
     !          do tt = 0, item
     !             xtem = xtem + xx(kk+tt)
     !             vtem = vtem + val(ii,jj,kk+tt)
     !          enddo
     !          xtem = xtem / steps(ii,jj,kk)
     !          vtem = vtem / steps(ii,jj,kk)              
     !          do tt = 0, item
     !             xuse(ii,jj,kk+tt) = xtem
     !             vuse(ii,jj,kk+tt) = vtem
     !          enddo
     !          kk       = kk+steps(ii,jj,kk)
     !       enddo
     !       ! ! no x use
     !       ! do kk = 1,nx3
     !       !    xuse(ii,jj,kk) = xx(kk)
     !       ! enddo
     !    enddo
     ! enddo
     ! !! END xuse procedure
     do kk = 1+ml,nx3-ml
        do jj = 1,nx2
           do ii = 1,nx1
              indexp = kk + steps(ii,jj,kk)
              indexm = kk - steps(ii,jj,kk)                 
              ! dvdx(ii,jj,kk) = (vuse(ii,jj,indexp)-vuse(ii,jj,indexm)) &
              !      / (xuse(ii,jj,indexp)-xuse(ii,jj,indexm))
              dvdx(ii,jj,kk) = (val(ii,jj,indexp)-val(ii,jj,indexm)) &
                   / (xx(indexp)-xx(indexm))
           enddo
        enddo
     enddo
!!! START periodic phi
!! assuming no level boundary in phi-direction and uniform grid
     if(flg_period) then
        do kk = 1, ml
           do jj = 1,nx2           
              do ii = 1,nx1
                 indexp = kk + steps(ii,jj,kk)
                 indexm = kk - steps(ii,jj,kk)
                 if (indexm < 1) then
                    indexm = indexm + nx3
                    dvdx(ii,jj,kk) = (val(ii,jj,indexp)-val(ii,jj,indexm)) &
                         / (xx(indexp)-xx(indexm) + PI2)
                 else
                    dvdx(ii,jj,kk) = (val(ii,jj,indexp)-val(ii,jj,indexm)) &
                         / (xx(indexp)-xx(indexm))
                 endif
              enddo
           enddo
        enddo
        do kk = nx3-ml+1, nx3
           do jj = 1,nx2           
              do ii = 1,nx1           
                 indexp = kk + steps(ii,jj,kk)
                 indexm = kk - steps(ii,jj,kk)
                 if (indexp > nx3)then
                    indexp = indexp - nx3
                    dvdx(ii,jj,kk) = (val(ii,jj,indexp)-val(ii,jj,indexm)) &
                         / (xx(indexp)-xx(indexm) + PI2)
                 else
                    dvdx(ii,jj,kk) = (val(ii,jj,indexp)-val(ii,jj,indexm)) &
                         / (xx(indexp)-xx(indexm))
                 endif                    
              enddo
           enddo
        enddo
     endif
!!! END periodic phi
  else if (dim == 2) then
     !! xuse
     ! jj=1
     ! do kk=1,nx3
     !    do ii=1,nx1
     !       do while (jj<nx2)
     !          xtem=0d0
     !          vtem=0d0
     !          item=steps(ii,jj,kk) -1
     !          do tt = 0, item
     !             xtem = xtem + xx(jj+tt)
     !             vtem = vtem + val(ii,jj+tt,kk)
     !          enddo
     !          xtem  = xtem / steps(ii,jj,kk)
     !          vtem  = vtem / steps(ii,jj,kk)              
     !          do tt = 0, item
     !             xuse(ii,jj+tt,kk) = xtem
     !             vuse(ii,jj+tt,kk) = vtem
     !          enddo
     !          jj    = jj+steps(ii,jj,kk)
     !       enddo
     !    enddo
     ! enddo
     ! !! END xuse
     do kk = 1,nx3
        do jj = ml+1, nx2-ml
           do ii = 1,nx1              
              indexp = jj + steps(ii,jj,kk)
              indexm = jj - steps(ii,jj,kk)                 
              ! dvdx(ii,jj,kk) = (vuse(ii,indexp,kk)-vuse(ii,indexm,kk)) &
              !      / (xuse(ii,indexp,kk)-xuse(ii,indexm,kk))
              dvdx(ii,jj,kk) = (val(ii,indexp,kk)-val(ii,indexm,kk)) &
                   / (xx(indexp)-xx(indexm))
           enddo
        enddo
     enddo
  else if (dim == 3) then
     ! !! xuse procedure
     ! ii=1
     ! do kk=1,nx3
     !    do jj=1,nx2
     !       do while (ii<nx1)
     !          xtem=0d0
     !          vtem=0d0
     !          item=steps(ii,jj,kk) -1
     !          do tt = 0, item
     !             xtem = xtem + xx(ii+tt)
     !             vtem = vtem + val(ii+tt,jj,kk)
     !          enddo
     !          xtem  = xtem / steps(ii,jj,kk)
     !          vtem  = vtem / steps(ii,jj,kk)              
     !          do tt = 0, item
     !             xuse(ii+tt,jj,kk) = xtem
     !             vuse(ii+tt,jj,kk) = vtem
     !          enddo
     !          ii    = ii+steps(ii,jj,kk)
     !       enddo
     !    enddo
     ! enddo
     ! !! END xuse procedure
     ! xuse = xx
     ! vuse = val     
     do kk = 1,nx3
        do jj = 1,nx2
           do ii = ml+1,nx1-ml
              indexp = ii + steps(ii,jj,kk)
              indexm = ii - steps(ii,jj,kk)                 
              ! dvdx(ii,jj,kk) = (vuse(indexp,jj,kk)-vuse(indexm,jj,kk)) &
              !      / (xuse(indexp,jj,kk)-xuse(indexm,jj,kk))
              dvdx(ii,jj,kk) = (val(indexp,jj,kk)-val(indexm,jj,kk)) &
                   / (xx(indexp)-xx(indexm))

           enddo
        enddo
     enddo
  endif
  return
end subroutine diff_f90

subroutine average(vals, rr, avR, nr, nval, vals_av)
  implicit none
  integer(4),intent(IN):: nr, nval
  real(8),intent(IN),dimension(nval,nr):: vals
  real(8),intent(IN),dimension(nr):: rr, avR
  real(8),intent(OUT),dimension(nval,nr):: vals_av
  integer(4)::Iav, lcount=1, rcount=1, ii !! this initialization is problematic when this subroutine is multiple-times called
  real(8),dimension(nval)::Svals !, Svuse
  real(8),dimension(nval,nr)::vdr
  real(8),dimension(nr):: rrig, rlef, dr
  real(8),dimension(nr+1):: rmid
  lcount = 1
  rcount = 1
  rrig(1:nr) = rr(1:nr) + avR(1:nr)
  rlef(1:nr) = rr(1:nr) - avR(1:nr)
  dr(1:nr)   = rrig(1:nr) - rlef(1:nr)
  rmid(1)    = rr(1)
  rmid(2:nr) = (rr(2:nr) + rr(1:nr-1))*0.5
  rmid(nr)   = rr(nr) !! dummy. This should not be used excepting the while rmid(rcount)  
  do ii = 1,nval
     vdr(ii,1:nr) = vals(ii,1:nr) * (rmid(2:nr+1) - rmid(1:nr))
  enddo
  ! print*,'F90',nr, nval
  ! print*, vals(1,21:50)
  ! print*, vdr(1,21:50)
  
  Svals(1:nval) = -vdr(1:nval,1)  !! will be cancelled out with the first cumsum
  !print*, Svals
  do Iav = 1,nr
     !! P1: Here, only the cells whose whole region is included is calculated
     do while (rcount < nr .and. rrig(Iav) > rmid(rcount+1) ) !! right boundary. 
        Svals(1:nval) = Svals(1:nval) + vdr(1:nval,rcount)        
        rcount        = rcount +1
     enddo
     !! now, rmid(rc-1) < rrig < rmid(rc)
     do while (rlef(Iav) > rmid(lcount+1) ) !! left boundary.
        Svals(1:nval) = Svals(1:nval) - vdr(1:nval,lcount+1)
        lcount        = lcount +1
     enddo
     !! now, rmid(lc) < rlef < rmid(lc+1)
     !! END P1
     vals_av(1:nval,Iav) = (Svals(1:nval) &
          + (rrig(Iav) - rmid(rcount)) *vals(1:nval,rcount) & !! for rrig > rr(nr), extrapolated
          + (rmid(lcount+1)- rlef(Iav))*vals(1:nval,lcount))& !! for rlef < rr(1), extrapolated
          / dr(Iav)
  enddo
  return
end subroutine average

!! val is already the summation of the both hemisphere int theta
subroutine zinteg_f90(val,rr,rf2,tan_mid, itmax,itmin, HoR_max, nval,nx3,JC,nx1, val_out)
  implicit none
  real(8),parameter:: hPI = 3.14159265358979323846d0 * 0.5
  integer(4),intent(IN):: nx1,JC,nx3,nval, itmax, itmin
  real(8),intent(IN):: HoR_max
  real(8),intent(IN),dimension(nx1,JC,nx3,nval):: val
  real(8),intent(IN),dimension(nx1)::  rr
  real(8),intent(IN),dimension(nx1+1)::rf2  !! Square of rface
  real(8),intent(IN),dimension(JC+1):: tan_mid !!! tan theta that is mesured from the midplane
  
  real(8),intent(OUT),dimension(nx1,nx3,nval):: val_out
  integer(4):: ii,jj, ir_loop
  real(8):: r_aim, ra2, zbot,zup,dz,zAC
  integer(4),parameter:: irmax=10
  
  val_out = 0d0
  
  do ii = 1,nx1-irmax
     !! initialization
     r_aim   = rr(ii)
     ra2     = r_aim*r_aim
     ir_loop = ii+1
     zbot    = r_aim * tan_mid(itmin+1) !! +1 corresponds to the lower boundary
     zAC     = dsqrt(rf2(ir_loop) - ra2)
     do while (zbot > zAC)
        ir_loop = ir_loop+1
        if (ir_loop > nx1-irmax) return
        zAC     = dsqrt(rf2(ir_loop) - ra2)
     enddo
     !print*, zbot/r_aim , tan_mid(itmax)
     do jj = itmin, itmax, -1  !! from midplane to top
        zup = r_aim * min( tan_mid(jj), HoR_max)  !!z at the upper boundary        
        dz = zup - zbot
        do while ( zup > zAC )
           !!! int dz from zbot to zAC
           dz   = zAC - zbot
           call kl_sum(val_out,val, nx1,JC,nx3,nval,dz,ii,jj)
           !!! renewing parameters
           dz   = zup - zAC
           zbot = zAC
           ir_loop = ir_loop+1
           if(ir_loop > nx1 - irmax) exit
           zAC  = dsqrt(rf2(ir_loop) - ra2)
        enddo
        if(ir_loop > nx1 - irmax) exit
        !! integ dz from zbot to zup, if zAC didn't come. If came, from zAC_old to zup
        call kl_sum(val_out,val, nx1,JC,nx3,nval,dz,ii,jj)
        zbot = zup
     enddo     
  enddo
  return
  
contains
  subroutine kl_sum(val_out,val, nx1,JC,nx3,nval,dz, ii,jj)
    implicit none
    integer(4),intent(IN):: nx1,JC,nx3,nval,ii,jj
    real(8),dimension(nx1,nx3,nval),intent(INOUT):: val_out
    real(8),dimension(nx1,JC,nx3,nval),intent(IN):: val
    real(8),intent(IN):: dz
    
    !integer(4)::kk,ll
    ! do kk = 1,nx3
    !    do ll = 1,nval
    val_out(ii,1:nx3,1:nval) = val_out(ii,1:nx3,1:nval) + &
         dz * val(ii,jj,1:nx3,1:nval)
    !    enddo
    ! enddo
    return
  end subroutine kl_sum
end subroutine ZInteg_f90

!! integration is done in zlim rather than theta lim
subroutine zinteg_zlim_f90(val,rr,rf2,tan_mid, zmin, zmax, nval,nx3,JC,nx1, val_out)
  implicit none
  real(8),parameter:: hPI = 3.14159265358979323846d0 * 0.5
  integer(4),intent(IN):: nx1,JC,nx3,nval
  real(8),intent(IN)::zmin, zmax
  real(8),intent(IN),dimension(nx1,JC,nx3,nval):: val
  real(8),intent(IN),dimension(nx1)::  rr
  real(8),intent(IN),dimension(nx1+1)::rf2  !! Square of rface
  real(8),intent(IN),dimension(JC+1):: tan_mid !!! tan theta that is mesured from the midplane
  
  real(8),intent(OUT),dimension(nx1,nx3,nval):: val_out
  integer(4):: ii,jj, ir_loop
  integer(4):: js,jt
  real(8):: r_aim, ra2, zbot,zup,dz,zAC
  integer(4),parameter:: irmax=10
  
  val_out = 0d0
  
  do ii = 1,nx1-irmax
     !! initialization
     r_aim   = rr(ii)
     ra2     = r_aim*r_aim
     ir_loop = ii+1
     zbot    = zmin
     zAC     = dsqrt(rf2(ir_loop) - ra2)
     do while (zAC < zbot)
        ir_loop = ir_loop+1
        zAC     = dsqrt(rf2(ir_loop) - ra2)
     enddo
     
     js      = JC
     zup = r_aim * tan_mid(js)  !!z at the upper boundary     
     do while (zup < zbot)
        js  = js-1
        zup = r_aim * tan_mid(js)  !!z at the upper boundary
     enddo
     !print*, js, JC, zup, zAC, zbot, dsqrt(rf2(ir_loop-1) - ra2), ir_loop
     
     do jj = js, 2, -1  !! from midplane to top
        if (zup > zmax) then
           dz   = zmax - zbot
           if(dz<0) then
              jt=jj+1
           else
              jt=jj
           endif
           call kl_sum(val_out,val, nx1,JC,nx3,nval,dz,ii,jt)
           !print*, dz, val_out(ii,1,1), 'last'
           exit
        endif
        zup = r_aim * tan_mid(jj)  !!z at the upper boundary
        dz = zup - zbot        
        do while ( zup > zAC )
           !1!! integral dz from zbot to zAC
           dz   = zAC - zbot
           call kl_sum(val_out,val, nx1,JC,nx3,nval,dz,ii,jj)
           !print*, dz, val_out(ii,1,1), 'AC', ir_loop, nx1-irmax, rf2(ir_loop)
           !1!! renewing parameters
           zbot = zAC
           dz   = zup - zbot
           ir_loop = ir_loop+1
           !if(ir_loop > nx1 - irmax) exit
           zAC  = dsqrt(rf2(ir_loop) - ra2)
        enddo
        if(ir_loop > nx1 - irmax) exit
        !! integ dz from zbot to zup, if zAC didn't come. If came, from zAC_old to zup
        call kl_sum(val_out,val, nx1,JC,nx3,nval,dz,ii,jj)
        !print*, dz, val_out(ii,1,1), 'UP', jj
        zbot = zup
     enddo
     !if (val_out(ii,1,1) < 10) stop
  enddo  
  return
  
contains
  subroutine kl_sum(val_out,val, nx1,JC,nx3,nval,dz, ii,jj)
    implicit none
    integer(4),intent(IN):: nx1,JC,nx3,nval,ii,jj
    real(8),dimension(nx1,nx3,nval),intent(INOUT):: val_out
    real(8),dimension(nx1,JC,nx3,nval),intent(IN):: val
    real(8),intent(IN):: dz
    
    integer(4)::ll !,kk
    do ll = 1,nval
       val_out(ii,:,ll) = val_out(ii,:,ll) + &
            dz * val(ii,jj,:,ll)
    enddo
    return
  end subroutine kl_sum
end subroutine ZInteg_zlim_f90


!! properties other than rho such as velocity square is fine. 
subroutine F90_shock_detect(rho, rr, phi, iRSP, nphi, nr, I_preshock)
  implicit none
  integer(4),intent(IN):: nr, nphi, iRSP
  real(8),intent(IN),dimension(nr,nphi)::rho
  real(8),intent(IN),dimension(nphi)::   phi
  real(8),intent(IN),dimension(nr)::     rr
  integer(4),intent(OUT),dimension(nr)::I_preshock
  !real(8),intent(OUT),dimension(nr)::shock_cos, shock_sin  
  integer(4):: ii
  real(8),dimension(nphi):: drho
  !real(8),dimension(nr):: dr
  !real(8)::rdphi, d_inv
  integer(4),dimension(1):: temI
  !! shock detect
  do ii = 1,iRSP
     drho(1:nphi-1) = rho(ii,2:nphi)
     drho(nphi)     = rho(ii,1)
     drho(:)        = drho(:) - rho(ii,:)     
     temI= maxloc(abs(drho(:)))
     I_preshock(ii) = temI(1)
  enddo
  do ii = iRSP+1 , nr
     drho(2:nphi) = rho(ii,1:nphi-1) - rho(ii,2:nphi)
     drho(1) = rho(ii,nphi) - rho(ii,1)
     temI= maxloc(abs(drho(:)))
     I_preshock(ii) = temI(1)
  enddo

  
  !! shock angle
  ! dr(2:nr-1) = rr(3:nr) - rr(1:nr-2)
  ! !! initialization
  ! shock_cos(1)  = 0d0
  ! shock_cos(nr) = 0d0
  ! shock_sin(1)  = 0d0
  ! shock_sin(nr) = 0d0
  ! do ii = 2,nr-1
  !    rdphi         = (phi(I_preshock(ii+1)) - phi(I_preshock(ii-1)) ) * rr(ii)
  !    d_inv         = 1d0 / sqrt(rdphi**2 + dr(ii)**2)
  !    shock_cos(ii) = dr(ii) * d_inv
  !    shock_sin(ii) = rdphi  * d_inv
  ! enddo
  return
end subroutine F90_shock_detect


subroutine f90_zconst(param, rface, rtarget, theta_face, theta_target, nx3,nx2,nx1, pz)
  implicit none
  integer(4),intent(IN):: nx1,nx2,nx3 !! nx2 is half of original one
  real(8),intent(IN),dimension(nx1,nx2,nx3):: param
  real(8),intent(IN),dimension(nx2):: theta_face
  real(8),intent(IN),dimension(nx1+1):: rface
  real(8),intent(IN),dimension(nx1):: rtarget, theta_target
  real(8),intent(OUT),dimension(nx1,nx3):: pz
  
  integer(4)::iz ,jj, iR !!iR: i for targetting R, iz: i for spherical r for calculating z
  !real(8)::

  jj=1
  iz=1
  do iR=1,nx1
     do while (theta_face(jj+1) < theta_target(iR) )
        jj = jj+1
     enddo
     do while (rface(iz+1) < rtarget(iR) )
        iz = iz+1
     enddo
     pz(iR,:) = param(iz,jj,:)
     !print*, theta_face(jj+1), theta_target(iR), rface(iz+1), rtarget(IR), pz(IR,1), rface(iz+1) * cos(theta_face(jj+1))
  enddo
  return
end subroutine f90_zconst

     

subroutine rad_level(reflev, rr, theta, lmax, nx2,nx1, rad)
  implicit none
  real(8),parameter:: PI2 = 3.14159265358979323846d0 * 2.
  integer(4),intent(IN):: nx1,nx2, lmax
  integer(4),intent(IN),dimension(nx1,nx2):: reflev
  real(8),intent(IN),dimension(nx1):: rr
  real(8),intent(IN),dimension(nx2):: theta
  real(8),intent(OUT),dimension(nx1,nx2):: rad

  integer(4)::ii, jj, tem
  integer(4),dimension(nx1,nx2):: dlev  
  real(8),dimension(nx1,nx2):: dlev_inv
  real(8):: tem_rad
  
  dlev     = 2** (lmax - reflev(:,:))
  dlev_inv = 1d0/ dble(dlev(:,:))
  
  do jj = 1,nx2
     ii=1
     do while(ii < nx1)
        if (dlev(ii,jj) == 1) then
           rad(ii,jj) = rr(ii) * sin(theta(jj))
           
        else if( mod(jj, dlev(ii,jj) ) == 1) then 
           tem     = dlev(ii,jj)-1
           tem_rad = sum(rr(ii:ii+tem)) *dlev_inv(ii,jj)  * &
                sin(sum(theta(jj: jj+tem)) * dlev_inv(ii,jj) )
           rad(ii:ii+tem, jj:jj+tem) = tem_rad
        endif
        ii = ii + dlev(ii,jj)
     enddo
  enddo
  
  return
end subroutine rad_level
  
