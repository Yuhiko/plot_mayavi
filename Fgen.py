## Functions that are generally useful
import sys
import math
import numpy as np
import athena_read
import FgenF90
import ctypes
import multiprocessing as mulp

from  mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import copy
from sklearn import preprocessing


import time as mod_time

def find_cross(fx):
    index = np.argwhere(np.diff(np.sign(fx))).flatten()
    return index

def Integral_SphZ_2D(props,sphr,r_face,theta_face,nprop,theta_max):
## now, props is in 2D[index, theta, r]
## theta_max is mesured from the midplane
    theta_num= len(theta_face) ## theta should be even, theta_face should be odd
    IC = round( ( theta_num -1 ) * 0.5 ) -1  ### -1 means it dosn't include the midplane.
    r_num = len(sphr)
    IZP = np.zeros((nprop,r_num))
    theta_Mid = (math.pi*0.5 - theta_face[0:IC]) ## theta from the mid plane
    #print(math.pi*0.5 - theta_face[IC]) ### This should be 0, but -1e-8, which probably comes fromthe accuracy problem
    tm = max( 16, int(find_cross(theta_Mid - theta_max)[0]) )
    if nprop == 1:
        tem_props = np.zeros((1,IC+1,r_num))
        tem_props[0,0:IC,:]  = props[0:IC,:]
        tem_props[0,0:IC,:] += props[-1:IC+1:-1,:]
    else:    
        tem_props = copy.deepcopy(props[:,0:IC,:])
        tem_props += props[:,-1:IC+1:-1,:]
    ###########################################
    for ii in range(r_num):
        r_aim    = sphr[ii]
        kk       = ii
        jj       = IC
        zbot     = 0.0
        theta_AC = math.acos(r_aim/r_face[ii+1]) ## arc cross
        ex_flg   = False
        while jj>tm: ## neglecting near the boundary
            jj     -= 1
            ztop_j  = r_aim * math.tan(theta_Mid[jj])
            while (theta_AC < theta_Mid[jj]):
                ztop_AC  = r_aim * math.tan(theta_AC)
                IZP[:,ii] += (tem_props[:,jj,kk]) * (ztop_AC - zbot) 
                kk        += 1
                zbot       = ztop_AC
                if (kk >= r_num-1 ):
                    ex_flg = True
                    break
                ### end if
                theta_AC = math.acos(r_aim/r_face[kk+1]) ## arc cross                           
            ### end while
            if(ex_flg):
                break            
            IZP[:,ii] += (tem_props[:,jj,kk]) * (ztop_j - zbot)                   
            zbot       = ztop_j
        ### end while jj
    ### end for ii
    if nprop == 1:
        return IZP[0,:]
    else:
        return IZP

def Integral_SphZ_3D(props,sphr,r_face,theta_face,nx3,nprop,trans): ## now, prop is in 3D[phi, theta, r]
    ### trans[index_r] is not degree, but the height
    theta_num = len(theta_face)
    IC        = round( ( theta_num -1 ) * 0.5 ) -1
    IZP       = np.zeros( (nprop,nx3, len(sphr)) )
    IZP_disk  = np.zeros( (nprop,nx3, len(sphr)) )
    r_num     = len(sphr)
    theta_Mid = math.pi*0.5 - theta_face[0:IC] ## theta from the mid plane. minus in the lower half
    if (nprop==1):
        #tem_props = np.zeros(( 1, nx3, len(theta_face)//2-1, len(sphr) ))
        tem_props = np.zeros(( 1, nx3, IC, len(sphr) ))        
        tem_props[0,:,:,:] = props[:,0:IC,:] + props[:,-1:IC+1:-1,:]
    else:
        tem_props = copy.deepcopy(props[:,:,0:IC,:])
        tem_props += props[:,:,-1:IC+1:-1,:]
        
    for ii in range(r_num):
        r_aim    = sphr[ii]
        kk       = ii
        zbot     = 0.0
        theta_AC = math.acos(r_aim/r_face[kk+1]) ## arc cross. When theta>theta_AC, vertical line goes to the outer shell denoted by r[ii+1]        
        jj       = IC    ## upper half
        ex_flg   = False
        disk_flg = True
        while jj>8:
            jj     -= 1
            ztop_j  = r_aim * math.tan(theta_Mid[jj])
            if (disk_flg and trans[ii] < ztop_j ):
                dz = (trans[ii] - zbot)
                if(dz<0):
                    print("WARNING in integral sph 3D1")
                    print(trans[ii], zbot)
                IZP_disk[:,:,ii] = copy.deepcopy(IZP[:,:,ii])
                IZP_disk[:,:,ii] += tem_props[:,:,jj,kk] * dz
                disk_flg = False
            while (theta_AC < theta_Mid[jj]):
                ztop_AC  = r_aim * math.tan(theta_AC)
                dz =(ztop_AC - zbot)
                if(dz<0):
                    print("WARNING in integral sph 3D2")
                    print(ztop_AC, zbot)
                zbot     = ztop_AC  ## remember the ztop_AC < z_top                
                IZP[:,:,ii] += tem_props[:,:,jj,kk] * dz
                kk      += 1
                if (kk >= r_num-1):
                    ex_flg = True
                    break
                ### end if
                theta_AC = math.acos(r_aim/r_face[kk+1]) ## arc cross                           
            ### end while
            if(ex_flg ):
                #print("break",ii,jj,kk)
                break
            dz =(ztop_j - zbot)
            if(dz<0):
                print("WARNING in integral sph 3D 3")
                print(ztop_j, zbot)
            IZP[:,:,ii] += tem_props[:,:,jj,kk] * dz
            zbot = ztop_j
        ### end while jj
        if(disk_flg):
            #print('boundary never comes',ii , jj, r_aim)
            IZP_disk[:,:,ii] = copy.deepcopy(IZP[:,:,ii])
    ### end for ii
    if(nprop == 1):
        return IZP[0,:,:], IZP_disk[0,:,:]
    else:        
        return IZP, IZP_disk

## phi index indicating where skip the integration to avoid the too strong planetary effects.
def skip_phi(phi, RSP, distance):
    nx3 = len(phi)
    if (RSP*phi[1] > distance):
        index1=0
        index2=nx3
    else:
        tem= find_cross(phi * RSP  - distance)
        index1=tem[0]
        index2=nx3-index1
    return index1, index2

class ExHill:
    def __init__(self,sph,RSP,Hill):
        if sph.nx3==1: ## cylindrical 2D
            d2   = np.einsum( 'i,j->ji',np.square(sph.rr),np.ones(sph.nx2))
        else:
            unit_kj = np.ones((sph.nx3,sph.nx2))
            tem1 = np.square(sph.rr) + RSP**2 ## r^2 = (rcos)^2 + (rsin)^2            
            d2   = np.einsum('i, kj->kji', tem1,unit_kj) \
                - 2*RSP* np.einsum('ji,k -> kji', sph.rad_ji, sph.cosp )
        ################################
        self.mask_Hill = (d2 < Hill**2)
    ###################
    def mean3D(self,parameter):
        if  parameter.shape[0] ==1:
            masked_param = np.ma.array( parameter[0,:,:], mask=self.mask_Hill )
        else:
            masked_param = np.ma.array( parameter, mask=self.mask_Hill )
        ##########
        return masked_param.mean(axis=0)

def midplane(props, z, flg_cyl):
    zlen     = len(z)
    zlen_2   = zlen//2
    flg_3D   = False
    flg_1D   = False
    if props.ndim ==3:
        flg_3D=True
    elif props.ndim ==1:
        flg_1D=True
    if (zlen %2 == 1):       
        if flg_3D:
            if flg_cyl:       #z     phi r
                output = copy.deepcopy(props[zlen_2,:, :])
            else:
                output = copy.deepcopy(props[:,zlen_2, :])
                              #phi,theta,r
        elif flg_1D:
            output = props[zlen_2]                
        else:
            output = copy.deepcopy(props[zlen_2, :])
    else:
        if flg_3D:
            if flg_cyl:
                output = np.mean(props[zlen_2 -1:zlen_2, :,:],axis=0)
            else:
                output = np.mean(props[:,zlen_2 -1:zlen_2, :],axis=1)
        elif flg_1D:
            output = np.mean(props[zlen_2 -1:zlen_2],axis=0)
        else:
            output = np.mean(props[zlen_2 -1:zlen_2, :],axis=0)
    return output   

def Fplot(ax, xx ,yy,r_max,flg_log,color): ## Function plot
    if flg_log:
        ax.set_xlabel(r'$\log_{10}(r)\ x / r$')
    else:
        ax.set_xlabel(r'$x$')
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax.set_xlim(0,r_max)
    plot=ax.plot(xx,yy,color=color)
    return plot

def Fplot_dash(ax, xx ,yy,r_max,flg_log,color,ltype,lname): ## Function plot
    if flg_log:
        ax.set_xlabel(r'$\log_{10}(r)\ x / r$')
    else:
        ax.set_xlabel(r'$x$')
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax.set_xlim(0,r_max)
    plot=ax.plot(xx,yy,color=color, linestyle=ltype,label=lname)
    return plot

def Fzplot_dash(ax, xx ,yy,r_max,flg_log,color,ltype,lname): ## Function plot
    if flg_log:
        ax.set_xlabel(r'$\log_{10}(r)\ z / r$')
    else:
        ax.set_xlabel(r'$z$')
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax.set_xlim(-r_max,r_max)
    plot=ax.plot(xx,yy,color=color, linestyle=ltype,label=lname)
    return plot

def rzplot(ax, xx ,yy, xr ,flg_log,color): ## Function plot
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax.set_xlim(-xr,xr)
    plot=ax.plot(xx,yy,color=color)
    return plot

def Cplot(ax,xx,yy,cc,r_max,c_range,flg_log,flg_mid): ## color plot
    if flg_log:
        norm = colors.LogNorm(vmin=c_range[0],vmax=c_range[1])
    else:
        norm = colors.Normalize(vmin=c_range[0],vmax=c_range[1])
    im = ax.pcolormesh(xx, yy, cc, cmap='plasma', norm=norm)
    ax.set_aspect('equal')
    if flg_mid:
        ax.set_xlim((-r_max, r_max))
    else:
        ax.set_xlim((0,r_max))
    ax.set_ylim((-r_max, r_max))
    im.set_clim(c_range)
    driver = make_axes_locatable(ax)
    cax    = driver.append_axes("right",size="5%",pad=0.05)
    imc    = plt.colorbar(im,cax=cax)
    if not flg_log :
        imc.formatter.set_powerlimits((0,0)) 
    return

def Cplot2(ax,xx,yy,cc,r_max,c_range,flg_log,flg_mid,cmap='plasma'): ## color plot. Aspect ratio is not 1
    if flg_log:
        norm = colors.LogNorm(vmin=c_range[0],vmax=c_range[1])
    else:
        norm = colors.Normalize(vmin=c_range[0],vmax=c_range[1])
    im = ax.pcolormesh(xx, yy, cc, cmap=cmap, norm=norm)
    if flg_mid:
        ax.set_xlim((-r_max, r_max))
    else:
        ax.set_xlim((0,r_max))
    im.set_clim(c_range)
    driver = make_axes_locatable(ax)
    cax    = driver.append_axes("right",size="5%",pad=0.05)
    imc    = plt.colorbar(im,cax=cax)
    if not flg_log :
        imc.formatter.set_powerlimits((0,0)) 
    return

def Cplot_planet(ax,xx,yy,cc,r_range,c_range,flg_log,RSP): ## color plot, centering the RSP
    if flg_log:
        norm = colors.LogNorm(vmin=c_range[0],vmax=c_range[1])
    else:
        norm = colors.Normalize(vmin=c_range[0],vmax=c_range[1])
    im = ax.pcolormesh(xx, yy, cc, cmap='plasma', norm=norm)
    ax.set_aspect('equal')
    ax.set_xlim((RSP-r_range, RSP+r_range))
    ax.set_ylim((-r_range, r_range))
    im.set_clim(c_range)
    driver = make_axes_locatable(ax)
    cax    = driver.append_axes("right",size="5%",pad=0.05)
    imc    = plt.colorbar(im,cax=cax)
    if not flg_log :
        imc.formatter.set_powerlimits((0,0)) 
    return

# Define grid compression in theta-direction
def dtheta_func(xmin,xmax, hr2_user, nf):
    #ath = athena_read.athdf(kwargs)
    #print(athdf.level)
    x2_vals = np.linspace(xmin, xmax, nf)
    hr2=hr2_user  #abs(hr2_user) ** (1.0 / 2.0 ** self.level)
    hr2_down=1.0/hr2
    nf_2=(nf-1)//2
    #print("x2rat",hr2,hr2_down)
    PI_2=math.pi/2.0
    dx=xmax-xmin
    sigmaN=1.- hr2**(nf-1)
    sigmaN_down= 1.-hr2_down**(nf-1)
    dx_sN=dx/sigmaN
    dx_sNd=dx/sigmaN_down
    if (xmin<PI_2):
        if (xmax>PI_2):
            dx=dx*0.5
            dx_sN=dx/(1.0- hr2**(nf_2-1) )
            dx_sNd=dx/(1.0-hr2_down**(nf_2) )
            x2_vals[0:nf_2]= xmin + (1.0 - hr2**np.arange(nf_2) ) * dx_sN
            x2_vals[nf_2:nf]= PI_2 + (1.0 - hr2_down**np.arange(nf_2+1) ) * dx_sNd
        else:
            x2_vals[0:nf]= xmin + (1.0 - hr2**np.arange(nf) ) * dx_sN
    else:
        x2_vals[0:nf]= xmin + (1.0 - hr2_down**np.arange(nf) ) * dx_sNd
    return x2_vals


def read_data(**kwargs):
    # Determine refinement level to use
    if kwargs['level'] is not None:
        level = kwargs['level']
    else:
        level = None

    # Determine if vector quantities should be read
    if kwargs['UOV'] is not None:        
        if kwargs['ray'] is not None:
            qMHD = ['user_out_var4','user_out_var6']
        else:
            qMHD = ['user_out_var0','user_out_var2']
        #####
    if kwargs['MHD']:
        quantities = ['Levels','rho','vel1','vel2','vel3','press','Bcc1','Bcc2','Bcc3']
    else:
        quantities = ['Levels','rho','vel1','vel2','vel3','press']
    ########
    #fname = kwargs['UOV']
    nn    = kwargs['output_num']
    fname = kwargs['data_path'].strip()+f"all.{nn:05d}.athdf"
    uname = kwargs['data_path'].strip()+f"uov.{nn:05d}.athdf"
    dataUSER=None
    if kwargs['com2'] is not None:
        if level == 0:
            print("dtheta level0-mode level=",level)
            data = athena_read.athdf(fname, quantities=quantities,
                                     level=level, face_func_2=dtheta_func,
                                     fast_restrict=True)
            
            if fname is not None :
                dataUSER = athena_read.athdf(uname, quantities=qMHD,
                                             level=level, face_func_2=dtheta_func,
                                             fast_restrict=True) 
        elif quantities[0] == 'Levels':
            #print("dtheta levels level=",level)
            data = athena_read.athdf(fname, quantities=quantities[1:],
                                     level=level, return_levels=True,
                                     face_func_2=dtheta_func)
            if fname is not None :
                dataUSER = athena_read.athdf(uname, quantities=qMHD,
                                             level=level,
                                             face_func_2=dtheta_func)
        else:
            print("dtheta else")
            data = athena_read.athdf(fname, quantities=quantities,
                                     level=level, face_func_2=dtheta_func)
            if fname  is not None :
                dataUSER = athena_read.athdf(uname, quantities=qMHD,
                                             level=level, face_func_2=dtheta_func)
            ####
    else:
        if quantities[0] == 'Levels':
            data = athena_read.athdf(fname, quantities=quantities[1:],
                                     level=level, return_levels=True)
            if fname is not None :
                dataUSER = athena_read.athdf(uname, quantities=quantities,
                                     level=level)
        else:
            data = athena_read.athdf(fname, quantities=quantities,
                                     level=level)
            if kwargs['UOV'] is not None :
                dataUSER = athena_read.athdf(uname, quantities=qMHD,
                                         level=level) 
        #####
    ##############
    return data, dataUSER


#### zstep should be small enough. Now, it's not volume average nor mass average. All points have the same weight. It's not a good way.
def rz_avr(nval,a_rz_in, rr, rmin,rmax,theta, zstep,zmax):  ## zmax: 7*HoR,  zstep = 0.1*HoR
    if (nval==1):
        n1=len(a_rz_in[:,0])
        n2=len(a_rz_in[0,:])
        a_rz=np.zeros((n1,n2,1))
        a_rz[:,:,0]=a_rz_in
    else:
        a_rz=a_rz_in
    ###########################
    rslice = (rmin + rmax)*0.5
    zmax_r = zmax*rslice   ### 7H
    zSQR   = (zmax_r )**2
    r_inc  = math.sqrt(rmin**2 + zSQR)
    r_outc = math.sqrt(rmax**2 + zSQR)    
    index_rmin = find_cross(rr - rmin)[0] + 1 # +0 is just outside of the zone
    index_rmax = find_cross(rr - rmax)[0] + 1  # additional +1 is not requried because it's used as the second index of range()
    index_rinc = find_cross(rr - r_inc)[0] + 1
    index_routc= find_cross(rr - r_outc)[0] + 1 # additional +1 is not requried because it's used as the second index of range()
    nz_out= int(round(2 * zmax/zstep))    
    # global RZAVR_out
    # global RZAVR_count
    RZAVR_out = np.zeros((nz_out,nval))
    RZAVR_count = np.zeros((nz_out))
    sint = np.sin(theta)
    cost = np.cos(theta)

    def loop_theta(tin,tout,r_it,rr_pos,RZAVR_out, RZAVR_count): # rr_pos is normalized by rslice, because zmax and zstep are normalized by rslice.
        for t_it in range(tin, tout):
            index_t = math.floor( (cost[t_it] * rr_pos + zmax)/ zstep )
            #print( (cost[t_it] * rr_pos + zmax)/ zstep , index_t)
            # When z = -zmax, index = 0. When z=zmax, index = 2 zmax / zstep
            RZAVR_out[index_t,:] += a_rz[t_it, r_it,:]
            RZAVR_count[index_t] += 1
            ###################        
        return RZAVR_out, RZAVR_count

    
    def theta_r(r_it, r_cross): ## crossing at r=const. line
        tlist      = find_cross(sint[:]*rr[r_it] - r_cross )
        index_tmin = tlist[0]+1
        index_tmax = tlist[1]+1  ## not 0 but 1, to reduce '+1' later.
        nx2 = len(theta)//2
        # if (not nx2 - index_tmin == index_tmax-nx2):
        #     print('tr',index_tmin, index_tmax, len(theta), sint[index_tmin]*rr[r_it], sint[index_tmax]*rr[r_it],r_cross)
        return index_tmin, index_tmax    

    
    def theta_z(r_it):  ## crossing at z=const. line
        index_tmin = find_cross(cost[:]*rr[r_it] - zmax_r)[0]+1
        index_tmax = find_cross(cost[:]*rr[r_it] + zmax_r)[0]+1
        nx2 = len(theta)//2
        # if (not nx2 - index_tmin == index_tmax-nx2):
        #     print('tz',index_tmin, index_tmax, len(theta), cost[index_tmin]*rr[r_it], cost[index_tmax]*rr[r_it], zmax_r)
        return index_tmin, index_tmax                
    
    
    def in_in(index_out, RZAVR_out, RZAVR_count): # The case the arc is crossing the inner boundary at rmin
        for r_it in range(index_rmin, index_out):
            rr_pos = rr[r_it] / rslice
            tin,tout = theta_r(r_it,rmin)
            RZAVR_out, RZAVR_count = loop_theta(tin,tout,r_it,rr_pos,RZAVR_out, RZAVR_count)
        return RZAVR_out, RZAVR_count
    
        
#    print(r_inc,r_outc,rmin,rmax)
    if (r_inc < rmax):
        RZAVR_out, RZAVR_count=in_in(index_rinc,RZAVR_out, RZAVR_count)
        ####
        for r_it in range(index_rinc,index_rmax):
            rr_pos  = rr[r_it] / rslice
            t1,t2   = theta_z(r_it)
            RZAVR_out, RZAVR_count=loop_theta(t1, t2,r_it,rr_pos,RZAVR_out, RZAVR_count)
        ####
        for r_it in range(index_rmax, index_routc):
            rr_pos  = rr[r_it] / rslice
            t1,t2   = theta_z(r_it)
            to1,to2 = theta_r(r_it,rmax)
            RZAVR_out, RZAVR_count= loop_theta(t1 , to1,r_it,rr_pos,RZAVR_out, RZAVR_count)
            RZAVR_out, RZAVR_count= loop_theta(to2, t2,r_it,rr_pos,RZAVR_out, RZAVR_count)
        ####
    else:
        RZAVR_out, RZAVR_count=in_in(index_rmax,RZAVR_out, RZAVR_count)
        ####
        for r_it in range(index_rmax,index_rinc):
            rr_pos  = rr[r_it] / rslice
            t1,t2   = theta_r(r_it,rmin)
            to1,to2 = theta_r(r_it,rmax)
            RZAVR_out, RZAVR_count=loop_theta(t1 , to1,r_it,rr_pos,RZAVR_out, RZAVR_count)
            RZAVR_out, RZAVR_count=loop_theta(to2, t2,r_it,rr_pos,RZAVR_out, RZAVR_count)
        ####
        for r_it in range(index_rinc, index_routc):
            rr_pos  = rr[r_it] / rslice
            t1,t2   = theta_z(r_it)
            to1,to2 = theta_r(r_it,rmax)
            RZAVR_out, RZAVR_count=loop_theta(t1 , to1,r_it,rr_pos,RZAVR_out, RZAVR_count)
            RZAVR_out, RZAVR_count=loop_theta(to2, t2,r_it,rr_pos,RZAVR_out, RZAVR_count)
        ####    
    return RZAVR_out, RZAVR_count



def rz_slice(rr, cost, sint, param,rslice , zmax): ## param is in (theta, r)
    nx1 = len(rr)
    nx2 = len(cost)
    nmid = round(nx2/2)
    
    def int_val(ii,jj):
        out = (param[jj, ii-1] * (rr[ii]-rslice) + param[jj, ii] * (rslice-rr[ii-1]) )\
            / (rr[ii]-rr[ii-1])
        return out

    
    for ii in range(nx1):
        if (rr[ii] > rslice):
            out_z = np.array([rslice*cost[nmid-1], rslice*cost[nmid]])
            val1 = int_val(ii,nmid-1)
            val2 = int_val(ii,nmid)
            out_param = np.array([val1,val2])
            inum = ii
            break
    ######################
    for jj in range(nmid):
        if (rr[inum]*sint[nmid+jj] < rslice):
            inum += 1
            if (rr[inum]*sint[nmid+jj] < rslice):
                print("WARNING")
        j1=nmid -1-jj
        j2=nmid   +jj
        if (rr[inum]*cost[j1] >zmax):
            break
        out_z = np.append(rr[inum]*cost[j1], out_z)
        out_z = np.append(out_z , rr[inum]*cost[j2])
        val1 = int_val(inum,j1)
        val2 = int_val(inum,j2)
        out_param = np.append(val1, out_param)
        out_param = np.append(out_param, val2)
    return out_z, out_param




### find the height where 'prop' is equal to the aimed value
### Now, this surveys only the upper hemisphere (theta < pi/2)
# flg_dec: if True, prop is expected to decrease from the midplane
def contour_surf(prop, rr, r_face, theta_face, phi, aimed, flg_dec=True):
    nx1 = len(rr)
    nx2 = (len(theta_face)-1)//2 
    nx3 = len(phi)    
    out = np.ones((nx3,nx1))
    out *= -1
    ofi = 10 ## doesn't use some grids close to the (outer) boundary
    irmax = nx1-ofi
    theta_mid = theta_face[0:nx2+1] - math.pi
    costL = np.cos(theta_face)
    for ir in range(irmax):
        irtem = ir
        arc_theta = math.atan(r_face[irtem+1]/ rr[ir])
        for it in range(nx2-ofi):            
            ITC = nx2 - it # theta index counting from the midplane    
            cost = costL[ITC]
            while (arc_theta < theta_mid[ITC]):
                irtem += 1
                arc_theta = math.atan(r_face[irtem+1]/ rr[ir])
            height = cost * rr[irtem]
            temp = prop[:,ITC,irtem]
            out[:,ir]=np.where( prop[:,ITC,irtem] > aimed, height, out[:,ir])
            ####################
            if np.max(temp) < aimed  or irtem == irmax :
                break
            ####################            
        ########################
    return out
            
            
            

def d2(val,dim,ddim):
    if dim==1:
        return val[2:] - val[0:-2]
    if dim==2:
        if ddim == 1:
            return val[2:,:] - val[0:-2,:]
        else:
            return val[:,2:] - val[:,0:-2]
    if dim==3:
        if ddim == 1:
            return val[2:,:,:] - val[0:-2,:,:]
        elif ddim==2:
            return val[:,2:,:] - val[:,0:-2,:]
        else:
            return val[:,:,2:] - val[:,:,0:-2]
    ##############################################


def diff(val,xx,dim,levels,lmax ,flg_period=False):
    #start = mod_time.time()
    nx     = len(xx)
    ddim   = val.ndim
    #print(ddim)
    if ddim == 1:
        val_ex        = np.zeros((val.shape[0],1,1))
        val_ex[:,0,0] = val
        lev_ex        = np.zeros((val.shape[0],1,1))
        lev_ex[:,0,0] = levels
    elif ddim ==2:
        val_ex = np.zeros((val.shape[0],val.shape[1],1))
        val_ex[:,:,0] = val
        lev_ex = np.zeros((val.shape[0],val.shape[1],1))
        lev_ex[:,:,0] = levels
    else:
        val_ex = val
        lev_ex = levels
    #####################################        
    nx1     = val_ex.shape[2]
    nx2     = val_ex.shape[1]
    nx3     = val_ex.shape[0]
    ndim    = val_ex.shape[dim-1]
    # print(val_ex.T.shape)
    # print(val_ex.shape)
    # print(lev_ex.T.shape)
    # print(lev_ex.shape)
    out  = FgenF90.diff_f90(val_ex.T, lev_ex.T, xx,dim, lmax ,flg_period, nx3,nx2,nx1,ndim)
    dvdx = out.T
    # print(out.T.shape)
    # print(out.shape)
    # elapsed_time = mod_time.time() - start
    # print("elapsed time in diff:{0} [sec]".format(elapsed_time))        
    if ddim==1:
        return dvdx[:,0,0]
    elif ddim==2:
        return dvdx[:,:,0]
    else:
        return dvdx

####################################################

def Rav(vals, rr, scale, nval):  ## considering 1D val
    nr  = len(rr)
    if nval ==1:
        val_ex      = np.zeros((nr,nval))
        val_ex[:,0] = vals
    else:
        val_ex      = vals
    ##############################
    out    = FgenF90.average(val_ex.T, rr, scale, nr, nval)
    if nval==1:
        return out[0,:]
    else:
        return out.T
#############################
    



def interp_stream(vals, rr, theta, Rlim=[0, 30], zlim=[-15, 15], n=301, flg_theta=True):
    from scipy import interpolate
    R = np.linspace(Rlim[0], Rlim[1], n)
    z = np.linspace(zlim[0], zlim[1], n)
    RR, zz = np.meshgrid(R,z)
    if flg_theta:
        newgrid = np.stack([np.arctan2(RR, zz), np.sqrt(RR ** 2 + zz ** 2)], axis=-1) ## measuring from y axis
    else:
        phi = np.arctan2(zz, RR)
        phi[phi<0] += 2*math.pi
        newgrid = np.stack([phi, np.sqrt(RR ** 2 + zz ** 2)], axis=-1) ## measuring from x axis
    #######################
    intps = []
    for val in vals: 
        intp = interpolate.interpn(
            (theta, rr),
            val,
            newgrid,
            bounds_error=False,
            fill_value=None,
        ) 
        intps.append(intp)
    return (RR, zz, *intps)

def interp_stream_pz(vals, phi, zz, plim=[0, 2*math.pi], zlim=[-15, 15], n=301):
    from scipy import interpolate
    Lphi = np.linspace(plim[0], plim[1], n)
    Lz   = np.linspace(zlim[0], zlim[1], n)
    Gphi, Gz= np.meshgrid(Lphi,Lz) ## order where one want to use
    newgrid = np.stack([Gphi, Gz], axis=-1)
    #######################
    intps = []
    for val in vals:
        intp = interpolate.interpn(
            (phi, zz),
            val,
            newgrid,
            bounds_error=False,
            fill_value=None,
        ) 
        intps.append(intp)
    return (Gphi, Gz, *intps)

def interp_stream_sph(vals, rr, phi, Rlim=[0, 30], philim=[0, 2*math.pi], n=301):
    from scipy import interpolate
    Rt     = np.linspace(Rlim[0], Rlim[1], n)
    pt     = np.linspace(philim[0], philim[1], n)
    RR, pp = np.meshgrid(Rt,pt)
    newgrid = np.stack([pp, RR], axis=-1)
    #######################
    intps = []
    #print(phi.shape, rr.shape, vals[0].shape, vals[1].shape)
    for val in vals: 
        intp = interpolate.interpn(
            (phi, rr),
            val,
            newgrid,
            bounds_error=False,
            fill_value=None,
        ) 
        intps.append(intp)
    return (RR, pp, *intps)

def gen_cmap(ctem):
    #ctem   = np.array([ [0,'black'],[1.0,'springgreen']])
    cindex = ctem[:,0].astype(np.float32)
    cindex = preprocessing.minmax_scale(cindex)
    colornorm=[]
    for no, cnorm in enumerate(cindex):
        colornorm.append([cnorm,ctem[no,1]])
    cmap =matplotlib.colors.LinearSegmentedColormap.from_list('a',colornorm,N=6000)
    cmap.set_bad('black')
    return cmap

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def ColDenSurf(rho_rz, rr, theta, dtheta, col_aim):
    nx1 = len(rr)
    n2mid = len(theta)//2
    ColD = np.zeros((nx1))
    theta_list = np.zeros((nx1))
    theta_list[:] = n2mid
    #for jj in range(n2mid,0,-1): from the midplane
    for jj in range(n2mid): ## from the pole
        ColD[:] += dtheta[jj] * rr[:] * rho_rz[jj,:]
        #print(ColD[0])
        theta_list[ColD[:] < col_aim] = jj ## updating the position. After reaching to the aimed column density, this stops
    return theta_list


def val2nda(val):
    return np.ctypeslib.as_array(val)  ## shallow copy?

def nda2val(nda):
    nx = nda.ndim
    if nx == 3:
        nx3,nx2,nx1 = nda.shape
    else:
        sys.exit('Not yet made.')
    ################################
    val = mulp.Value(((ctypes.c_float * nx1) *nx2)*nx3, lock=False)
    val2nda(val)[:] = nda   ## shallow copy?
    return val

def MP_nda(nx3,nx2=1,nx1=1):
    val = mulp.Value(((ctypes.c_float * nx1) *nx2)*nx3, lock=False)  ## must be False
    return val

def shock_detect(CorS, rho,vR,vphi):
    ###################################################
    nphi = len(CorS.phi)
    nr   = len(CorS.rr)
    vSQ  = np.square(vR) + np.square(vphi)
    ## detecting shock shock
    #I_preshock  = FgenF90.f90_shock_detect(v2.T, CorS.rr, CorS.phi, CorS.iRSP, nphi,nr)#by vv
    ## by rho
    I_preshock  = FgenF90.f90_shock_detect(rho.T, CorS.rr, CorS.phi, CorS.iRSP, nphi,nr) ## postshock
    I_preshock -= 1 ### for donversion from f90 to c++. No transpose because of 1D
    ## post
    I_postshock = np.zeros_like(I_preshock)
    I_postshock[0:CorS.iRSP]    = I_preshock[0:CorS.iRSP] +1
    I_postshock[CorS.iRSP:]     = I_preshock[CorS.iRSP:]  -1
    I_postshock[I_postshock==nphi] = 0
    ## found
    ## measuring the angle
    phi_preshock = np.zeros((nr))
    phi_preshock[:] = CorS.phi[I_preshock[:]]
    phi_preshock_periodic = np.zeros((nr))
    temi = 0
    pi2  = 2*math.pi
    for ii in range (CorS.iRSP, nr):
        if phi_preshock[ii] > phi_preshock[ii-1]:
            temi -= 1
        phi_preshock_periodic[ii] = phi_preshock[ii] + temi * pi2        
    ####################
    temi  = 0
    for ii in range (CorS.iRSP, 0,-1):
        if phi_preshock[ii+1] > phi_preshock[ii]:
            temi += 1
        phi_preshock_periodic[ii] = phi_preshock[ii] + temi * pi2
    ####################

    #print(phi_preshock_periodic[CorS.iRSP: CorS.iRSP+150])
    ii=CorS.iRSP+90
    nn=15
    print(f'r={CorS.rr[ii]}')
    print(rho[I_postshock[ii],ii])
    print(rho[I_preshock[ii],ii])
    print(rho[I_preshock[ii]-nn:I_preshock[ii]+nn,ii])
    #ii=CorS.iRSP-40
    print('velocity')
    print(vSQ[I_postshock[ii],ii])
    print(vSQ[I_preshock[ii],ii])
    print(vSQ[I_preshock[ii]-nn:I_preshock[ii]+nn,ii])
    #print(rho[I_postshock[ii]-5:I_postshock[ii]+5,ii])    
    out = diff(phi_preshock_periodic, CorS.rr, 1, CorS.levels[0,CorS.nx2//2,:], CorS.lmax)
    rdphi     = out * CorS.rr  ## no Transpose because it's 1D
    phi_shock = np.arctan2(rdphi, 1)

    vel_perp    = np.zeros((nr))
    eta         = np.zeros((nr))
    shock_cos   = np.cos(phi_shock[:])
    shock_sin   = np.sin(phi_shock[:])
    for ii in range(nr):
        vel_perp[ii] = abs(vphi[I_preshock[ii],ii] * shock_cos[ii]\
            - vR[I_preshock[ii], ii] * shock_sin[ii] )
        #eta[ii]      = rho[I_postshock[ii],ii] / rho[I_preshock[ii],ii]
    ###########################################
    eta  = np.max(rho,axis=0)/ np.min(rho,axis=0)
    eta -= 1
    return I_preshock, vel_perp, eta, shock_cos, shock_sin


#def shock_Vperp(SorC,vel, I_preshock):


class param:
    def __init__(self,kwargs):
        r0 = 1
        #self.gamma = param.gamma
        print(kwargs['MHD'])
        if kwargs['MHD']:
            import param_MHD as param
            self.beta0  = param.beta0
            self.addB   = param.taddBp
        else:
            import param_2D as param
            self.Valpha  = param.valpha
        self.GM          = param.GM
        self.omega_frame = 0.0
        self.alpha       = - param.alpha  ### density 
        self.Tslope      = - param.qT     ### temperature slope 
        self.rho0        = param.rho0
        self.hor0        = param.HoR0
        self.SD0         = param.rho0 * (r0* self.hor0 * math.sqrt(2*math.pi))
        self.Sigma_slope = (self.Tslope*0.5 +1.5 + self.alpha)
        OmegaP      = param.Omega0
        self.omega_frame = OmegaP
        self.RSP    = (self.GM/OmegaP**2)**(1./3.)
        ####
        # else:  ## case with 2D hydro? need to modify
        #     import param
        #     self.GM  = param.GM
        #     self.RSP = param.RSP
        #     self.omega_frame = 1.0 #(GM/RSP**3)**0.5     #(1.0/6**3)**0.5 = 1.0/6**1.5 = 0.068
        #     self.q_MP        = param.q_MP
        #     self.Tslope      = 2.* param.cs_slope
        #     self.SD0         = param.sigma0
        #     self.hor0        = param.cs0
        #     self.Sigma_slope = param.sigma_slope
        #     self.rho0        = self.SD0 / (self.hor0 * math.sqrt(2*math.pi))
        #     self.alpha       = sefl.Sigma_slope - ( self.Tslope * 0.5 +1.5 )
        #######################################################################################
        self.HoR_slope = (self.Tslope + 3)*0.5 -1
        if kwargs['noP'] is None:
            self.t_planet= param.start_planet
            self.tgrow   = param.TauGrow
            self.fac_PM  = param.fac_PM   ## planetary mass normalized by thermal mass
            self.HP = (self.hor0 * self.RSP**(self.HoR_slope+1)  )  ## scaleheight at the planet
            GMth    = (self.HP/self.RSP)**3 * self.GM * 3.
            self.q_MP    = GMth * self.fac_PM
            self.GMP  = self.GM * self.q_MP
            #self.Hill = self.HP*self.q_MP
            self.Hill = (self.q_MP/3.)**(1./3.) * self.RSP
            print("Hill", self.HP*(self.fac_PM)**(1./3.), self.Hill, self.q_MP)
            print(f'Omega_frame={self.omega_frame:.2e}',self.RSP,self.GMP)
        ##################################
        self.ztrans  = 3.5 ## start changing around 3, due to the smoothing function
        #self.ztrans  = 1.0 ## for test
        #ztrans = 3.5 ## Xuenin's comment. It should be slightly smaller than the transition value (4 here), but should not be too small, to avoid counting the turbulent disk
        return
    ##########################################    

    
class Spherical:
    def __init__(self,param,data):
        #coordinates = data['Coordinates']
        self.time   = data['Time']
        self.rr     = data['x1v']
        self.theta  = data['x2v']
        self.phi    = data['x3v']
        self.r_face     = data['x1f']
        self.theta_face = data['x2f']
        self.phi_face   = data['x3f']
        self.dr     = np.diff(self.r_face)
        self.dtheta = np.diff(self.theta_face)
        self.dphi   = np.diff(self.phi_face)        
        self.nx1    = len(self.rr)
        self.nx2    = len(self.theta)
        self.nx3    = len(self.phi)
        self.sint   = np.sin(self.theta)
        self.cost   = np.cos(self.theta)        
        self.sinp   = np.sin(self.phi)
        self.cosp   = np.cos(self.phi)
        self.rad_ji = np.einsum("i,j->ji" , self.rr,self.sint)
        #self.z_ji   = np.einsum("i,j->ji" , self.rr,self.cost)
        
        self.sintdt = self.sint * self.dtheta
        self.rSQR   = np.square(self.rr)
        
        self.rad_inv = 1./self.rad_ji
        self.r_inv   = 1./self.rr

        self.omega_frame = param.omega_frame
        self.tauP    = (2*math.pi)/ self.omega_frame
        self.Ptime   = self.time/ self.tauP

        #dummy       = data['rho'] ## needed to initialize Levels
        self.levels = data['Levels']
        self.lmax   = data['MaxLevel']

        ### Disk parameter
        self.HoR0_r     = param.hor0   * self.rr**(param.HoR_slope)
        self.scaleH0    = self.HoR0_r  * self.rr
        self.transH     = self.scaleH0 * param.ztrans
        self.hor_trans  = param.ztrans * param.hor0
        ### inital properties
        self.rho0_mid_r = param.rho0 * self.rr**(param.alpha)
        self.rho0_mid_rad = param.rho0 * self.rad_ji**(param.alpha)        
        
        self.SD0_r      = (param.SD0 * self.rr** param.Sigma_slope)
        self.Bz0_mid_r  = np.sqrt( (2.0*param.GM/param.beta0)
                                   *self.r_inv* self.rho0_mid_r) * self.HoR0_r
        self.Bz0_mid_rad= np.sqrt( (2.0*param.GM/param.beta0)
                                   *self.rad_inv* self.rho0_mid_rad) \
                                   * param.hor0 * self.rad_ji**(param.HoR_slope) ## HoR rad
        self.vK         = np.sqrt(param.GM * self.r_inv)
        self.vK_rad     = np.sqrt(param.GM * self.rad_inv)
        self.OmegaK     = self.vK * self.r_inv
        self.cs0        = self.scaleH0 * self.OmegaK
        # self.P0         = self.rho0_mid_r * self.rad_ji**(param.HoR_slope*2 - 1)  \
        #     * np.square(param.hor0) * param.GM  check it again before using
        self.iRSP   = find_cross(self.rr - param.RSP)[0]
        ### surface
        theta_surf = math.atan(self.hor_trans) ## from midplane
        self.i_usurf  = find_cross(self.theta - (math.pi*0.5 - theta_surf) )[0] ## just before the criteria
        self.i_lsurf  = find_cross(self.theta - (math.pi*0.5 + theta_surf) )[0]+1 ## just after the criteria
        #print("isurf",self.i_usurf,self.i_lsurf, self.nx2, self.nx2-self.i_usurf)
        Lt  = self.cost[self.i_usurf] - self.cost[self.i_lsurf] ## 2pi is not needed in current setting * 2*math.pi
        self.Lt_inv = 1./Lt

        #### save data
        self.v3F    = np.zeros((self.nx3, self.nx2, self.nx1))
    ######################################
    def renew_time(self,data):
        self.Ptime   = data['Time'] / self.tauP
    #####################################
    
    def Rot_dvphi_dtheta(self,val): ## d(v sin theta)/(R dtheta)
        vndim = val.ndim
        if vndim ==3:
            sval  = np.einsum('kji,j->kji',val[:,:,:],self.sint[:])
            d2phi = diff(sval, self.theta, 2, self.levels,self.lmax) * self.rad_inv
        elif vndim==2:
            sval  = np.einsum('ji,j->ji',val[:,:],self.sint[:])
            d2phi = diff(sval, self.theta, 1, self.levels[0,:,:],self.lmax) * self.rad_inv
        #######################################################
        return d2phi
    def Rot_dvr_dtheta(self,val): ## dv/ (r dtheta), dt2_inv= 1./( d2(theta) )
        if val.ndim==3:
            d2r = diff(val, self.theta, 2, self.levels,self.lmax) * self.r_inv
        else:
            d2r = diff(val, self.theta, 1, self.levels[0,:,:],self.lmax) * self.r_inv
        return d2r

    def Rot_dv_dr(self,val, flg_nophi=False):  ## nabla_r in spherical or cylindrical 3D (1/r d(rA)/dr)        
        rval  = val * self.rr
        if val.ndim==3:
            d1ret = diff(rval, self.rr, 3, self.levels,self.lmax) *self.r_inv
        else:
            if flg_nophi:
                d1ret = diff(rval, self.rr, 2, self.levels[0,:,:],self.lmax) *self.r_inv
            else:
                d1ret = diff(rval, self.rr, 2, self.levels[:,self.nx2//2,:],self.lmax) *self.r_inv
        return d1ret
    def Rot_dv_dphi(self,val): ## dp2_inv = 1./[phi(i+1) - phi(i-1)] ## dv/ (R dphi)
        if val.ndim==3:
            d3ret = diff(val, self.phi, 1, self.levels,self.lmax,True) * self.rad_inv
        else:
            d3ret = diff(val, self.phi, 1, self.levels[:,self.nx2//2,:],self.lmax,True) * self.r_inv
        return d3ret
    #########################
    ## for multiprocess in self.rotation
    def MRot_dvphi_dtheta(self,val,rval):
        vndim = val.ndim
        if vndim ==3:
            sval  = np.einsum('kji,j->kji',val[:,:,:],self.sint[:])
            val2nda(rval)[:] = diff(sval, self.theta, 2, self.levels,self.lmax) * self.rad_inv
        elif vndim==2:
            sval  = np.einsum('ji,j->ji',val[:,:],self.sint[:])
            val2nda(rval)[:] = diff(sval, self.theta, 1, self.levels,self.lmax) * self.rad_inv
        #######################################################
        return
    def MRot_dvr_dtheta(self,val,rval):
        vndim = val.ndim
        if vndim == 3:
            val2nda(rval)[:] = diff(val, self.theta, 2, self.levels,self.lmax) * self.r_inv
        elif vndim == 2:
            val2nda(rval)[:] = diff(val, self.theta, 1, self.levels,self.lmax) * self.r_inv
        return

    def MRot_dv_dr(self,val,rval): 
        tem   = val * self.rr
        vndim = val.ndim
        if vndim == 3:
            val2nda(rval)[:] = diff(tem, self.rr, 3, self.levels,self.lmax) *self.r_inv
        elif vndim == 2:
            val2nda(rval)[:] = diff(tem, self.rr, 2, self.levels,self.lmax) *self.r_inv            
        return
    def MRot_dv_dphi(self,val,rval):
        # vndim = val.ndim
        # if vndim==3:
        ## Anyway, phi corresponds to 1
        val2nda(rval)[:] = diff(val, self.phi, 1, self.levels,self.lmax,True) * self.rad_inv
        return

    #########################
    def rotation(self,e1,e2,e3):
        #print(e1.dtype)
        d1r3 = MP_nda(self.nx3,self.nx2,self.nx1)
        d3r1 = MP_nda(self.nx3,self.nx2,self.nx1)
        d2r1 = MP_nda(self.nx3,self.nx2,self.nx1)
        d1r2 = MP_nda(self.nx3,self.nx2,self.nx1)
        d3r2 = MP_nda(self.nx3,self.nx2,self.nx1)

        p1 = mulp.Process(target = self.MRot_dv_dphi     , args=(e1,d1r3))
        p2 = mulp.Process(target = self.MRot_dv_dr       , args=(e3,d3r1))
        p3 = mulp.Process(target = self.MRot_dv_dr       , args=(e2,d2r1))
        p4 = mulp.Process(target = self.MRot_dvr_dtheta  , args=(e1,d1r2))
        p5 = mulp.Process(target = self.MRot_dvphi_dtheta, args=(e3,d3r2))
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        d2r3 = self.Rot_dv_dphi(e2)   ## main process
        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        rot_r = val2nda(d3r2)[:] - d2r3
        rot_t = val2nda(d1r3)[:] - val2nda(d3r1)[:]
        rot_p = val2nda(d2r1)[:] - val2nda(d1r2)[:]
        
        return rot_r, rot_t, rot_p    
    #######################################################
    def z(self,v_r, v_t):
        ndim = v_r.ndim
        if ndim==3:
            out = (np.einsum( "kji,j -> kji", v_r , self.cost) - 
                   np.einsum( "kji,j -> kji", v_t , self.sint)
            )
        elif ndim==2:
            out = (np.einsum( "ji,j -> ji", v_r , self.cost) - 
                   np.einsum( "ji,j -> ji", v_t , self.sint)
            )
        else:
            print('dimension problem in sph.z')
            sys.exit()
        ############################
        return out
    def rad(self,v_r, v_t):
        ndim = v_r.ndim
        if ndim==3:
            out = (np.einsum( "kji,j -> kji", v_r , self.sint) +
                   np.einsum( "kji,j -> kji", v_t , self.cost)
            )            
        elif ndim==2:
            out = (np.einsum( "ji,j -> ji", v_r , self.sint) +
                   np.einsum( "ji,j -> ji", v_t , self.cost)
            )
        else:
            print('dimension problem in sph.rad')
            sys.exit()
        ############################            
        return out
    
    def Bline(self,B1_rz, B2_rz ): ### It should be used for phi-averaged B
        offset_r = 5 ### index rather than the code unit length
        B1_use = B1_rz * self.rad_ji * 2*math.pi
        B2_use = B2_rz * self.rad_ji * 2*math.pi
        dr_ji=np.tile(self.dr,(self.nx2,1))
        #### assuming PHI(theta=0,r=rin)=0
        ### Now, tracing the PHI=constant line. The line traces (represents) B_r e_r + B_phi e_phi. Thus, PHI increase in the perpendicular direction.
        tem     = B1_use[:,offset_r]
        B1R     = -( tem*self.dtheta )*self.rr[offset_r]
        B1RI    = np.cumsum(B1R,axis=0)  ### Integrating in theta direction. Now, the PHI increase in theta direction is set to each cell.
        tem     = np.tile(B1RI,(self.nx1,1))   #### Broadcast
        B1RI_ji = tem.T        
        B2dr    = B2_use*dr_ji
        B2I_ji  = np.cumsum(B2dr,axis=1)  ## integrating in r direction. This term corresponds to the increase of PHI in r direction.
        PHI     = B1RI_ji + B2I_ji ## as a memo, either of + and - is fine
        return PHI

    ### output: value of 'props' at the disk surface (theta=theta_surf)
    ### theta_surf is mesured from the pole
    def disk_surf(self,props, nprops, theta_surf):
        pshape = props.shape
        if nprops==1:
            pdim   = 0
        else:
            pdim   = 1
        if len(pshape) ==3+pdim: ## 
            out = np.zeros((nprops, pshape[0], pshape[2])) ## phi and r
            if (nprops == 1):
                prop_in = np.zeros((nprops, pshape[0],pshape[1], pshape[2]))
                prop_in[0,:,:,:] = props
            else:
                prop_in = props
        else:
            out = np.zeros((nprops,1, pshape[1])) ## onlyl r
            prop_in = np.zeros((nprops, 1, pshape[0],pshape[1]))
            if (nprops == 1):
                prop_in[0,0,:,:] = props
            else:
                prop_in[:,0,:,:] = props
            
        it  = find_cross(self.theta - theta_surf)[0]
        CylR = self.rad_ji[it,:]
        maxr = CylR[-1]
        for ir in range(self.nx1):
            if(self.rr[ir] > maxr) :
                break
            irCyl = find_cross(self.rr[ir] - CylR)[0]        
            out[:,:,ir] = (  prop_in[:,:,it,irCyl]   *(CylR[irCyl+1] - self.rr[ir])
                         + prop_in[:,:,it,irCyl+1] *(self.rr[ir] - CylR[irCyl])
                        ) / (CylR[irCyl+1] - CylR[irCyl])
        if pdim == 0:
            if len(pshape) == 3:
                return out[0,:,:]
            else:
                return out[0,0,:]
        else:
            if len(pshape) == 3:
                return out[:,:,:]
            else:
                return out[:,0,:]
    ################################################

    
    def Planet_Gravity_Torque(self, rho, param):
        r_cut = 0.8 * param.Hill  ## Kley+2009

        rsinp = np.einsum( "i,k->ki" , self.rr, self.sinp)
        rcosp = np.einsum( "i,k->ki" , self.rr, self.cosp)
        zz    = np.einsum( "i,j->ji" , self.rr, self.cost)
        unit  = np.ones_like(rho)
        ###
        dist2_mid  = np.square(param.RSP-rcosp) + np.square(rsinp)
        dist2_tile = np.einsum("ki,kji->kji",dist2_mid,unit)
        z2         = np.square(zz)
        z2_tile    = unit * z2
        ####
        dist   = np.sqrt(dist2_tile + z2_tile)
        fd_inv = np.exp( -10.0*(dist/r_cut-1) ) + 1.0    ### Kley+2009
        RSP_d3 = param.RSP/( np.power(dist,3) * fd_inv)
        ###
        tem1        = rho * RSP_d3       ## [k,j,i]
        tem2        = np.einsum("k,kji->ji",self.sinp* self.dphi,tem1)
        ### this is the torque acting on planet.
        ltor_rtheta = param.GMP *  (np.square(self.rad_ji) * tem2)    ### For phi-direction integration, not r but R_cyl
        ### GMP/d^2 * Rsinp/d     * RSP             * R dphi
        # (gravity) * (direction) * (torque length) * (phiintegration)
        ## 2 pi is included when it's integrated
        ####
        torque      = np.einsum("i,j,ji->",self.rr*self.dr,self.dtheta,ltor_rtheta)
        return ltor_rtheta


    def ZIntegral(self, props, nprop, HoR_max = -10.0, HoR_min=-10.0, flg_zlim=False):
        ## now, props is in 2D[index, theta, r]
        ## HoR_max is set to the disk boundary, without input parameter
        ##
        ## Don't cross the midplane
        JC   = self.nx2//2
        ndim = props.ndim

        npdim = 1
        if nprop == 1:
            npdim=0
        
        if ( ndim == 3+npdim): ## 3D                    
            nphi = self.nx3
        elif(ndim == 2+npdim): ##2D
            nphi = 1
        ######################
        tem_props = np.zeros((nprop,nphi,JC,self.nx1))
        Double = 1
        upS    = 1
        itmin  = JC
        if flg_zlim:
            if (HoR_min < 0.0): 
                upS     = 0
                Double  = 1
                tem     = -HoR_min
                HoR_min = -HoR_max
                HoR_max = tem
                print('TAKE CARE THE SIGN OF PARAMETERS')
            else:
                upS    = 1
                Double = 0
        elif HoR_min > -9.0:
            Double  = 0
            theta_trans = math.atan2( 1.0, HoR_min ) ## measuring from the pole
            itmin       = int(find_cross(self.theta - theta_trans)[0])+1 ## because fortran count from 1
            #print(theta_trans/math.pi,itmin,self.theta[itmin]/math.pi,self.theta[itmin+1]/math.pi, self.theta[JC]/math.pi)
        ################
        if min(nprop, nphi) ==1:            
            if nprop+nphi == 2: ## both is 1
                tem_props[0,0,0:JC,:]  = props[0:JC,:] *upS
                tem_props[0,0,0:JC,:] += props[-1:JC-1:-1,:] * Double
            elif nprop==1:
                tem_props[0,:,0:JC,:]  = props[:,0:JC,:] * upS
                tem_props[0,:,0:JC,:] += props[:,-1:JC-1:-1,:]* Double
            else:
                tem_props[:,0,0:JC,:]  = props[:,0:JC,:] * upS
                tem_props[:,0,0:JC,:] += props[:,-1:JC-1:-1,:]* Double
            #############
        else:    
            tem_props[:,:,:,:]  = props[:,:,0:JC,:] * upS
            tem_props[:,:,:,:] += props[:,:,-1:JC-1:-1,:]* Double
        #################    
        #IZP = np.zeros((nprop, nphi, self.nx1))
        ###################################################
        if ((not flg_zlim) and HoR_max < -9.0):
            HoR_max = self.hor_trans
        #################
        theta_trans = math.atan2( 1.0, HoR_max ) ## measuring from the pole
        itmax       = max( 16, int(find_cross(self.theta - theta_trans)[0])+1 ) ##because fortran counts from 1
        ###################################################
        rf2     = np.square(self.r_face)
        tan_mid = np.tan(math.pi*0.5 - self.theta_face[0:JC+1])
        #print('tan',tan_mid)
        #print(itmax, itmin, JC)
        ###############
        if (flg_zlim):
            out = FgenF90.zinteg_zlim_f90(tem_props.T,self.rr, rf2,tan_mid, HoR_min, HoR_max,
                                          nprop,nphi,JC,self.nx1)
        else:
            out = FgenF90.zinteg_f90(     tem_props.T,self.rr, rf2,tan_mid, itmax, itmin, HoR_max,
                                          nprop,nphi,JC,self.nx1)
        ##################
        IZP = out.T

        if nphi == 1:
            if nprop==1:
                return IZP[0,0,:]
            else:
                return IZP[:,0,:]            
        else:
            if nprop==1:
                return IZP[0,:,:]
            else:
                return IZP[:,:,:]
    ###################################################################################
    
    def v3_frame(self):
        out  = FgenF90.rad_level(self.levels[0,:,:].T, self.rr, self.theta, self.lmax,
                                self.nx2, self.nx1)
        v3_frame = out.T * self.omega_frame
        # if (ndim==2):
        #     return v3_frame
        # elif (ndim==3):
        unit     = np.ones((self.nx3,self.nx2,self.nx1))
        self.v3F = np.einsum('ji,kji->kji', v3_frame, unit)
        return
    ###########################################

    

    def Isintdt(self,val, it_bound, it_Lbound=-1,Fsin=True):
        if Fsin:
            dI = self.sintdt
        else:
            dI = self.dtheta
        if (it_Lbound <0):
            it_Lbound = self.nx2-it_bound
        if(it_bound > it_Lbound):
            tem=it_Lbound
            it_Lbound = it_bound
            it_bound  = tem
        #print(it_bound, it_Lbound)
        if val.ndim==3:
            tem = np.einsum('kji,j->kji',val, dI)
            out = np.sum(tem[:,it_bound:it_Lbound,:], axis=1)
        elif val.ndim==2:
            tem = np.einsum('ji,j->ji',val, dI)
            out = np.sum(tem[it_bound:it_Lbound,:], axis=0)
        #######################################
        return out

    def Zconst(self, param, zface):
        ndim = param.ndim
        if ndim==2:
            TH   = param.shape[0]//2
            n3   = 1
            puse = np.zeros((n3, TH, param.shape[1]))
            if zface > 0:                
                puse[0,:,:] = param[:TH,:]
            else:
                puse[0,:,:] = param[TH:,:]
                print('TAKE CARE THE SIGN OF PARAMETERS')
        else:
            TH   = param.shape[1]//2
            n3   = param.shape[0]
            puse = np.zeros((n3, TH, param.shape[2]))
            if zface > 0:                
                puse[:,:,:] = param[:,:TH,:]
            else:
                puse[:,:,:] = param[:,TH:,:]
                print('TAKE CARE THE SIGN OF PARAMETERS')
        #####################
        theta_target = np.arctan(self.rr/ zface)
        rtarget  = np.sqrt(np.square(self.rr ) + zface**2)
        out      = FgenF90.f90_zconst(puse.T,self.r_face, rtarget,
                                      self.theta_face[:TH], theta_target,
                                      n3,TH,self.nx1)
        if ndim==2:
            return out.T[0,:]
        else:
            return out.T
    ###########################

    
class Cylindrical:
    def __init__(self,param,data):
        #coordinates = data['Coordinates']
        self.time   = data['Time']
        self.rr     = data['x1v']
        self.phi    = data['x2v']        
        self.zz     = data['x3v']
        self.r_face     = data['x1f']
        self.phi_face   = data['x2f']
        self.z_face     = data['x3f']
        self.dr     = np.diff(self.r_face)
        self.dphi   = np.diff(self.phi_face)
        self.dz     = np.diff(self.z_face)
        self.nx1    = len(self.rr)
        self.nx2    = len(self.phi)
        self.nx3    = len(self.zz)
        self.sinp   = np.sin(self.phi)
        self.cosp   = np.cos(self.phi)
        
        self.r_inv   = 1./self.rr
        self.omega_frame = param.omega_frame
        
        self.tauP    = (2*math.pi)/ self.omega_frame
        self.Ptime   = self.time/ self.tauP

        #dummy       = data['rho'] ## needed to initialize Levels
        self.levels = data['Levels']
        self.lmax   = data['MaxLevel']

        ### Disk parameter
        self.HoR0_r     = param.hor0   * self.rr**(param.HoR_slope)
        self.scaleH0    = self.HoR0_r  * self.rr
        self.transH     = self.scaleH0 * param.ztrans
        self.hor_trans  = param.ztrans * param.hor0   ## scaler
        ### inital properties
        # self.rho0_mid_r = param.rho0 * self.rr**(param.alpha)
        # self.rho0_mid_rad = param.rho0 * self.rad_ji**(param.alpha)        
        
        self.SD0_r      = (param.SD0 * self.rr** param.Sigma_slope)

        self.vK         = np.sqrt(param.GM * self.r_inv)
        self.OmegaK     = self.vK * self.r_inv
        self.cs0        = self.scaleH0 * self.OmegaK
        self.nu         = param.Valpha * self.cs0 * self.scaleH0         #here, 0 means not r=r0 but t=0
        #self.nuslope    = (param.HoR_slope+1) * 2 + 1.5  ## H^2 Omega
        # self.P0         = self.rho0_mid_r * self.rad_ji**(param.HoR_slope*2 - 1)  \
        #     * np.square(param.hor0) * param.GM  check it again before using
    ######################################
    def renew_time(self,data):
        self.Ptime   = data['Time'] / self.tauP
    #####################################
    def Planet_Gravity_Torque(self, rho, param):
        r_cut = 0.8 * param.Hill  ## Kley+2009

        rsinp = np.einsum( "i,k->ki" , self.rr, self.sinp)
        rcosp = np.einsum( "i,k->ki" , self.rr, self.cosp)
        ###
        dist  = np.sqrt(np.square(param.RSP-rcosp) + np.square(rsinp))
        #print(self.sinp)
        ####
        fd_inv = np.exp( -10.0*(dist/r_cut-1) ) + 1.0    ### Kley+2009
        RSP_d3 = param.RSP/( np.power(dist,3) * fd_inv)
        ###
        tem1        = rho * RSP_d3       ## [k,i]
        tem2        = np.einsum("k,ki->i",self.sinp* self.dphi, tem1)
        ### this is the torque acting on planet.
        ltor_r      = param.GMP *  (np.square(self.rr) * tem2)    ### For phi-direction integration, not r but R_cyl
        ### GMP/d^2 * Rsinp/d * RSP, (gravity) * (direction) * (torque length)
        ## 2 pi is included when it's integrated
        self.iRSP   = find_cross(self.rr - param.RSP)[0]
        ####
        return ltor_r
    def Rot_dv_dphi(self,val): ## dp2_inv = 1./[phi(i+1) - phi(i-1)] ## dv/ (R dphi)
        if val.ndim==3:
            d3ret = diff(val, self.phi, 1, self.levels,self.lmax,True) * self.r_inv
        else:
            d3ret = diff(val, self.phi, 1, self.levels[0,:,:],self.lmax,True) * self.r_inv
        return d3ret
    def Rot_dvphi_dr(self,val): ## dp2_inv = 1./[phi(i+1) - phi(i-1)] ## dv/ (R dphi)
        if val.ndim==3:
            d3ret = diff(val*self.rr, self.rr, 3, self.levels,self.lmax,True) * self.r_inv
        else:
            d3ret = diff(val*self.rr, self.rr, 2, self.levels[0,:,:],self.lmax,True) * self.r_inv
        return d3ret
