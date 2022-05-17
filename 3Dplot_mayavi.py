import argparse
import copy
import math
import numpy as np
import time as mod_time

import athena_read
import Fgen

from tvtk.api import tvtk

import multiprocessing as mulp


flg_offscreen = True ## seems not working with pdf. Fine with png
flg_stream    = False
flg_MHD       = True

def loop_in(kwargs, nn, quantities, param,sph, ExHill):
    start = mod_time.time()
    fname = kwargs['data_path'].strip()+f"all.{nn:05d}.athdf"
    data = athena_read.athdf(fname, quantities=quantities[1:],
                             face_func_2=Fgen.dtheta_func,
                             return_levels=True)

    rho  = data['rho'][:,:,:]    
    if sph is None:  ## must be after loading at least one quantity
        sph    = Fgen.Spherical(param,data)
        Exfac  = 1.0
        ExHill = Fgen.ExHill(sph, param.RSP, param.Hill*Exfac)
        elapsed_time = mod_time.time() - start
        sph.v3_frame()
        print(f"elapsed time after the coordinate initialization:{elapsed_time:.3e} [sec]")
    ###################################
    v3_rot   = data['vel3'][:,:,:] 
    ######################
    #####
    B1  = data['Bcc1'][:,:,:]
    B2  = data['Bcc2'][:,:,:]
    B3   = data['Bcc3'][:,:,:]    
    v3   = v3_rot + sph.v3F

    ncum    = 2
    each_av = np.zeros((sph.nx2,sph.nx1,ncum))
    each_av[:,:,0] = np.mean(B1,axis=0)
    each_av[:,:,1] = np.mean(B2,axis=0)
    
    ##########
    if flg_stream:
        ncum3D  = 5
        each_3D = np.zeros((sph.nx3,sph.nx2,sph.nx1,ncum3D))
        each_3D[:,:,:,0]  = rho
        each_3D[:,:,:,1]  = v3
        each_3D[:,:,:,2]  = B3
        each_3D[:,:,:,3]  = B1
        each_3D[:,:,:,4]  = B2
    else:
        ncum3D  = 3
        each_3D = np.zeros((sph.nx3,sph.nx2,sph.nx1,ncum3D))
        each_3D[:,:,:,0]  = rho
        each_3D[:,:,:,1]  = v3
        each_3D[:,:,:,2]  = B3

    elapsed_time = mod_time.time() - start
    print(f"elapsed time of {nn:d}th loop:{elapsed_time:.3e} [sec]")

    sph.renew_time(data)
    return each_av,each_3D, sph, ExHill
#######################################################################


def main(**kwargs):
    start = mod_time.time()
    PI    = math.pi
    param=Fgen.param(kwargs)    

    lstep  = kwargs['step']
    lstop  = kwargs['output_num'] + lstep ## it will be stopping at "output num"
    lstart = lstop - lstep * kwargs['lnum']
    if kwargs['MHD']:
        quantities = ['Levels', 'rho','vel1','vel2','vel3','press','Bcc1','Bcc2','Bcc3']
    else:
        quantities = ['Levels', 'rho','vel1','vel2','vel3','press']
    #######################################################################    
    ## first loop
    count_loop = 1
    each_av, each_3D, sph, ExHill = loop_in(kwargs, lstart, quantities, param, sph=None, ExHill=None)
    cum_av  = copy.deepcopy(each_av)
    cum_3D  = copy.deepcopy(each_3D)
    ## main loop
    for nn in range(lstart+1, lstop,lstep):
        count_loop += 1
        each_av, each_3D , sph, ExHill = loop_in(kwargs, nn, quantities, param, sph, ExHill)
        cum_av += each_av
        cum_3D += each_3D
    ###### end loop
    cum_av /= float(count_loop)
    cum_3D /= float(count_loop)
    elapsed_time = mod_time.time() - start
    print(f"elapsed time after the loop :{elapsed_time:.3e} [sec]")   


    ##################
    ## Visualization #
    ##################    
    ## grid
    rmax  = 30
    hormax= 0.3 #sph.hor_trans
    ########
    ipmin = 0
    ipmax = sph.nx3
    idp   = 1
    ipmid = (ipmax-ipmin)/idp //2
    ipquad= ipmid//2
    phi_loop = np.append( sph.phi[ipmin:ipmax:idp], sph.phi[ipmin])
    ###########
    xmax  = rmax / (math.sqrt(2)*1.1 )
    zmax  = xmax * hormax
    #############
    irmax = Fgen.find_cross(sph.rr-rmax )[0]+1
    itmid = sph.nx2//2
    # theta_max = math.atan( hormax )
    # itmin     = Fgen.find_cross(sph.theta - (math.pi*0.5 - theta_max) )[0] -1
    # itmax     = Fgen.find_cross(sph.theta - (math.pi*0.5 + theta_max) )[0] +1
    itmin = 0
    itmax = sph.nx3

    phi_grid, theta_grid, r_grid = np.meshgrid(phi_loop , sph.theta[itmin:itmax], sph.rr[:irmax], indexing='ij')
    x_grid  = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
    y_grid  = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
    z_grid  = r_grid * np.cos(theta_grid) 


    def add_loop(val):
        if val.ndim==1:
            val = np.append(val, val[ipmin])
        else:
            tem = np.reshape( val[ipmin,:,:], (1, val.shape[1], val.shape[2] ) )
            val = np.append(val, tem, axis=0 )
        return val
    
    rho_3d = np.log10(cum_3D[ipmin:ipmax:idp,itmin:itmax,0:irmax,0])
    rho_3d = add_loop(rho_3d)

    v3_3d  = cum_3D[ipmin:ipmax:idp,itmin:itmax,0:irmax,1] / sph.vK_rad[itmin:itmax, 0:irmax]
    v3_3d  = add_loop(v3_3d)


    if flg_MHD:
        B3_3d  = cum_3D[ipmin:ipmax:idp,itmin:itmax,0:irmax,2] 
        B3_3d  = add_loop(B3_3d)
        ##
        B3_nom = B3_3d / sph.Bz0_mid_rad[itmin:itmax, 0:irmax]
        ##############
        if flg_stream:
            B1_3d  = cum_3D[ipmin:ipmax:idp,itmin:itmax,0:irmax,3] 
            B1_3d  = add_loop(B1_3d)
            ##
            B2_3d  = cum_3D[ipmin:ipmax:idp,itmin:itmax,0:irmax,4]
            B2_3d  = add_loop(B2_3d)
            ##
            tem1= sph.rad(B1_3d, B2_3d)
            tems=add_loop(sph.sinp[ipmin:ipmax:idp])
            temc=add_loop(sph.cosp[ipmin:ipmax:idp])
            Bx  = np.einsum('kji,k->kji',tem1 , temc) -  np.einsum('kji,k->kji',B3_3d , tems)
            By  = np.einsum('kji,k->kji',tem1 , tems) +  np.einsum('kji,k->kji',B3_3d , temc)
            Bz  = sph.z(B1_3d, B2_3d)
            del tem1, tems, temc
        ########################
        del cum_3D

        B_PHI  = sph.Bline(cum_av[:,:,0], cum_av[:,:,1])[itmin:itmid,0:irmax]
        PHI_max= np.max(B_PHI)
        PHI_min= np.min(B_PHI)
        numBL  = 30
        dPHI = (PHI_max - PHI_min)/ float(numBL)
        LPHI = np.arange(PHI_min, PHI_max, dPHI).tolist()
        del dPHI
    ##################
    del cum_av

    
    if flg_offscreen :
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1280,1024))
        display.start()
    #########################
    import mayavi
    from mayavi import mlab
    
    if flg_offscreen :
        mlab.options.offscreen=True

    ################################    
    mlab.figure(size=(600,600))
    
    rhomin = -3
    rhomax = 0
    rhoiso = -2.5
    #### mayavi
    coord = np.empty(x_grid.shape + (3,))
    coord[..., 0] = x_grid
    coord[..., 1] = y_grid
    coord[..., 2] = z_grid
    coord = coord.transpose(2,1,0,3).copy()  ## copy to change the memory allocation order
    coord.shape = coord.size//3, 3

    #### grid
    SGrho= tvtk.StructuredGrid( dimensions=x_grid.shape , points=coord)
    SGrho.point_data.scalars      = np.ravel(rho_3d.T)
    SGrho.point_data.scalars.name = "density"
    src_rho                       = mlab.pipeline.add_dataset(SGrho)
    
    ## B vec
    if flg_stream:
        Bvec = np.empty(x_grid.shape+ (3,))
        Bvec[...,0] = Bx
        Bvec[...,1] = By
        Bvec[...,2] = Bz
        Bvec        = Bvec.transpose(2,1,0,3).copy()
        Bvec.shape  = Bvec.size//3 ,3
        ###
        SGB = tvtk.StructuredGrid( dimensions=x_grid.shape , points=coord)
        SGB.point_data.vectors      = Bvec
        SGB.point_data.vectors.name = 'B'
        src_B                       = mlab.pipeline.add_dataset(SGB)
    #################
    
    
    del rho_3d
    ########### rho_iso
    srcISO_rhov    = mlab.pipeline.extract_grid(src_rho)
    srcISO_rhov.trait_set(x_max=x_grid.shape[0]//4) ## x=phi, which ranges from 0 to x_grid.shape[0]-1.
    isosurface   = mlab.pipeline.iso_surface(srcISO_rhov, contours=[rhoiso], opacity=1,
                                             vmin=rhomin ,vmax=rhomax)
    ##
    rho_midplane = mlab.pipeline.scalar_cut_plane(src_rho,
                                                  plane_orientation='z_axes',
                                                  view_controls=False,
                                                  vmin=rhomin ,vmax=rhomax
    )
    rho_midplane.implicit_plane.origin=(0,0,0)
    ##
    src_rhov    = mlab.pipeline.extract_grid(src_rho)
    src_rhov.trait_set(x_min=x_grid.shape[0]-2) ## x=phi, which ranges from 0 to x_grid.shape[0]-1. 
    src_rhov    = mlab.pipeline.threshold(src_rhov, low=rhoiso)    

    rho_xz  = mlab.pipeline.scalar_cut_plane(src_rhov,
                                             plane_orientation='y_axes',                                             
                                             view_controls=False,
                                             vmin=rhomin ,vmax=rhomax
    )
    rho_xz.implicit_plane.origin=(0,0,0)

    ##### v3
    v3min = 0.9
    v3max = 1.1
    ###########
    SGv3 = tvtk.StructuredGrid( dimensions=x_grid.shape , points=coord)
    SGv3.point_data.scalars      = np.ravel(v3_3d.T)
    SGv3.point_data.scalars.name = "v3"    
    src_v3                       = mlab.pipeline.add_dataset(SGv3)

    srcE_v3 = mlab.pipeline.extract_grid(src_v3)    
    srcE_v3.trait_set(x_min=ipmid-2, x_max=ipmid+2)
    ## x=phi, which ranges from 0 to x_grid.shape[0]-1.    
    v3_xz   = mlab.pipeline.scalar_cut_plane(srcE_v3,
                                             plane_orientation='y_axes',
                                             view_controls=False,
                                             vmin=v3min ,vmax=v3max,
                                             colormap='bwr'
    )
    v3_xz.implicit_plane.origin=(0,0,0)

    ##### B3
    if flg_MHD:
        B3min    =-15
        B3max    = 0
        ###########
        SGB3 = tvtk.StructuredGrid( dimensions=x_grid.shape , points=coord)
        SGB3.point_data.scalars      = np.ravel(B3_nom.T)
        SGB3.point_data.scalars.name = "B3"
        src_B3                       = mlab.pipeline.add_dataset(SGB3)
        
        srcE_B3 = mlab.pipeline.extract_grid(src_B3)
        srcE_B3.trait_set( x_min = (ipquad-2) , x_max=(ipquad+2) )
        B3_yz   = mlab.pipeline.scalar_cut_plane(srcE_B3,
                                                 plane_orientation='x_axes',
                                                 view_controls=False,
                                                vmin=B3min ,vmax=B3max,
                                                 colormap='OrRd'
        )
        B3_yz.implicit_plane.origin=(0,0,0)
        B3_yz.module_manager.scalar_lut_manager.reverse_lut = True
        ###############
        #### Bline
        tg2, rg2 = np.meshgrid( sph.theta[itmin:itmid], sph.rr[:irmax], indexing='ij')
        xg2        = np.empty((1,)+tg2.shape )
        zg2        = np.empty( xg2.shape )  
        xg2[0,:,:] = rg2 * np.sin(tg2)
        zg2[0,:,:] = rg2 * np.cos(tg2) 
        yzcoord    = np.empty( xg2.shape + (3,))
        yzcoord[..., 0] = 0.0
        yzcoord[..., 1] = xg2
        yzcoord[..., 2] = zg2
        yzcoord = yzcoord.transpose(2,1,0,3).copy()  ## copy to change the memory allocation order
        yzcoord.shape = yzcoord.size//3, 3

        SGBline   = tvtk.StructuredGrid( dimensions=xg2.shape , points=yzcoord)
        SGBline.point_data.scalars = np.ravel(B_PHI.T)
        SGBline.point_data.scalars.name = 'B Phi'
        src_Bline = mlab.pipeline.add_dataset(SGBline)

        Bline = mlab.pipeline.contour_surface(src_Bline, colormap='bone',
                                              contours=LPHI
        )    
        ### all black
        lut = Bline.module_manager.scalar_lut_manager.lut.table.to_array()
        lut[:,0:3] = 0
        Bline.module_manager.scalar_lut_manager.lut.table = lut
    ####

        if flg_stream:
            ### Bstream
            Bfield_line = mlab.pipeline.streamline(SGB, seedtype='line',
                                                   integration_direction='both',
                                                   colormap='Vega10',vmin=PHI_min, vmax=PHI_max
            )
            Bfield_line.stream_tracer.maximum_propagation = 300.
            temz = 10.
            temy = 20.
            Bfield_line.seed.widget.point1 = [0.,0.,temz]
            Bfield_line.seed.widget.point2 = [0.,temy,temz]
            Bfield_line.seed.widget.resolution      = 50
            #Bfield_line.seed.widget.enabled         = False
            Bfield_line.seed.widget.clamp_to_bounds = False
            Bfield_line.seed.widget.align = 'y_axis'
            Bfield_line.stream_tracer.integration_direction = 'backward'
        #########
    ###########################################
    

    z1 = 0.04
    dz = 0.12
    ### decolation
    mlab.view(azimuth=330 ,elevation= 70, distance=35)
    cb1=mlab.colorbar(object    = rho_midplane, title='log10 Density', orientation='horizontal',
                      nb_labels =(rhomax - rhomin + 1),label_fmt='%.0f' )
    cb2=mlab.colorbar(object    = v3_xz, title='v3/vK', orientation='horizontal',
                      nb_labels =(int( (v3max-v3min)*10) + 1), label_fmt='%.1f' )
    if flg_MHD:
        cb3=mlab.colorbar(object    = B3_yz, title='B3/B{3,0}', orientation='horizontal',
                          nb_labels =(B3max - B3min)//5+1,label_fmt='%.0f' )
        cb3.scalar_bar_representation.position  = [0.67,z1]
        cb3.scalar_bar_representation.position2 = [0.33,dz]
    ##############
    cb1.scalar_bar_representation.position  = [0.01,z1]
    cb1.scalar_bar_representation.position2 = [0.33,dz] ## width?
    ##
    cb2.scalar_bar_representation.position  = [0.34,z1]
    cb2.scalar_bar_representation.position2 = [0.33,dz]
    
    
    
    if flg_offscreen:
        mlab.savefig("3D.png")
    else:
        if flg_stream:
            mlab.show()
        else:
            mlab.savefig(f"3D_3Mth_{kwargs['output_num']:05d}.pdf")
    #################################    
    
    elapsed_time = mod_time.time() - start
    print("Total elapsed time:{0} [sec]".format(elapsed_time))
    #######################

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',
                        help='input file path and header like disk_planet.')
    parser.add_argument('output_num',
                        type=int,
                        help=('number of input and output'))
    parser.add_argument('--step',
                        type=int,
                        default=1,
                        help=('step of the loop'))
    parser.add_argument('--lnum',
                        type=int,
                        default=1,
                        help=('number of the loop'))
    args = parser.parse_args()
    main(**vars(args))
