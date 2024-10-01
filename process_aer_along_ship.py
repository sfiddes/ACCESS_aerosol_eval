#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
from datetime import datetime
import scipy.special
import dask
import xarray as xr


# In[2]:


# some constants
Rd=287.05
cp=1005.46 # J/kg/K 1.0035
Rd_cp = Rd / cp
p0=1.0e5 # Pa
zboltz=1.3807e-23
avo=6.022e23
mm_da=avo*zboltz/Rd #molecular mass of dry air kg/mole

# Molecular masses (kgs/mol)
# n.b. mm_bc=0.012, mm_oc=mm_so=0.0168=1.4*mm_c. assume both primary & secondary organic species have mm=1.4*mm_c
mm_aer=[0.098,0.012,0.0168,0.05844,0.100,0.0168] 
cp_so4=0
cp_bc=1
cp_oc=2
cp_nacl=3
cp_du=4


# # Functions

# In[3]:


def aero_unit_conversions(aer,met): 
    
    #MMRs
    if 'field34071' in aer.data_vars: 
        meta = aer['field34071'].attrs
        aer['field34071'] = aer['field34071']*((29./62.) * 1e9) # DMS 
        aer['field34071'] = aer['field34071'].assign_attrs(meta)
        aer['field34071'] = aer['field34071'].assign_attrs({'units':'ppb'})

    if 'field34072' in aer.data_vars: 
        meta = aer['field34072'].attrs
        aer['field34072'] = aer['field34072']*((29./64.) * 1e9) # SO2
        aer['field34072'] = aer['field34072'].assign_attrs(meta)
        aer['field34072'] = aer['field34072'].assign_attrs({'units':'ppb'})

    if 'field34073' in aer.data_vars: 
        meta = aer['field34073'].attrs
        aer['field34073'] = aer['field34073']*((29./98.) * 1e12) # gas phase H2SO4
        aer['field34073'] = aer['field34073'].assign_attrs(meta)
        aer['field34073'] = aer['field34073'].assign_attrs({'units':'ppt'})

    # Number densities 
    for nd in ['field34101','field34103','field34107','field34113','field34119']:
        meta = aer[nd].attrs
        aer[nd] = aer[nd] * met.density
        aer[nd] = aer[nd].assign_attrs(meta)
        aer[nd] = aer[nd].assign_attrs({'units':'cm-3'})
    
    # Mass desnsities
    # H2SO4   
    for md in ['field34102','field34104','field34108','field34114']:
        meta = aer[md].attrs
        aer[md] = aer[md]*mm_da*(1/mm_aer[cp_so4])*(met.density/avo) #: kg/kg * kg/mol * mol/kg * mol/cm3
        aer[md] = aer[md].assign_attrs(meta)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})

    # Organics
    for md in ['field34126','field34106','field34121','field34110','field34116']:
        meta = aer[md].attrs
        aer[md] = aer[md] * mm_da*(1/mm_aer[cp_oc])*(met.density/avo)
        aer[md] = aer[md].assign_attrs(meta)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})

    # Sea salt
    for md in ['field34111','field34117']:
        meta = aer[md].attrs
        aer[md] = aer[md] * mm_da*(1/mm_aer[cp_nacl])*(met.density/avo)
        aer[md] = aer[md].assign_attrs(meta)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})

    # Black Carbon
    for md in ['field34105','field34109','field34115','field34120']:
        meta = aer[md].attrs
        aer[md] = aer[md] * mm_da*(1/mm_aer[cp_bc])*(met.density/avo)
        aer[md] = aer[md].assign_attrs(meta)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})

    # Dust 
    for md in ['field431','field432','field433','field434','field435','field436']:
        meta = aer[md].attrs
        aer[md] = aer[md] * mm_da*(1/mm_aer[cp_du])*(met.density/avo)
        aer[md] = aer[md].assign_attrs(meta)
        aer[md] = aer[md].assign_attrs({'units':'mol/cm-3'})    
        
    return aer


# In[4]:


def lognormal_cumulative_to_r(N,r,rbar,sigma):

    total_to_r=(N/2.0)*(1.0+scipy.special.erf(np.log(r/rbar)/np.sqrt(2.0)/np.log(sigma)))

    return total_to_r


# In[5]:


def nt10_calcs(aer):
    nsteps = len(aer.time)
    nmodes = 7
    if 'z0_hybrid_height' in aer.coords: 
        height = 'z0_hybrid_height'
    else: 
        height = 'z1_hybrid_height'
    nlevels=len(aer[height])
    nd = np.zeros((nmodes,nsteps,nlevels))
    rbardry = np.zeros ((nmodes,nsteps,nlevels))
    sigma_g = [1.59,1.59,1.4,2.0,1.59,1.59,2.0]

    nd[0,:,:] = aer.field34101.values[:,:] # Nuc
    nd[1,:,:] = aer.field34103.values[:,:] # Ait
    nd[2,:,:] = aer.field34107.values[:,:] # Acc
    nd[3,:,:] = aer.field34113.values[:,:] # Coa
    nd[4,:,:] = aer.field34119.values[:,:]  # Ait Insol

    rbardry[0,:,:] = (aer.field38401.values[:,:] / 2)
    rbardry[1,:,:] = (aer.field38402.values[:,:] / 2)
    rbardry[2,:,:] = (aer.field38403.values[:,:] / 2)
    rbardry[3,:,:] = (aer.field38404.values[:,:] / 2)
    rbardry[4,:,:] = (aer.field38405.values[:,:] / 2)

    nd_nuc_lt10 = lognormal_cumulative_to_r(nd[0,:,:],5e-9, rbardry[0,:], sigma_g[0])
    nd_nuc_gt10 = nd[0,:,:] - nd_nuc_lt10
    N10 = nd_nuc_gt10 + nd[1:,:,:].sum(axis=0)
    N10 = xr.DataArray(N10,coords=[aer.time,aer[height]],dims=['time',height],name='N10')
    aer = aer.assign(N10=N10)
    aer['N10'][:] = N10.values
    
    nd_nuc_lt3 = lognormal_cumulative_to_r(nd[0,:],1.5e-9, rbardry[0,:], sigma_g[0])
    nd_nuc_gt3 = nd[0,:,:] - nd_nuc_lt3
    N3 = nd_nuc_gt3 + nd[1:,:,:].sum(axis=0)
    N3 = xr.DataArray(N3,coords=[aer.time,aer[height]],dims=['time',height],name='N3')
    aer = aer.assign(N3=N3)
    aer['N3'][:] = N3.values
    return aer


# In[6]:


def CCN_calcs(aer):
    nsteps = len(aer.time)
    nmodes = 7
    if 'z0_hybrid_height' in aer.coords: 
        height = 'z0_hybrid_height'
    else: 
        height = 'z1_hybrid_height'
    nlevels=len(aer[height])
    nd = np.zeros((nmodes,nsteps,nlevels))
    rbardry = np.zeros ((nmodes,nsteps,nlevels))
    sigma_g = [1.59,1.59,1.4,2.0,1.59,1.59,2.0]

    nd[0,:,:] = aer.field34101.values[:,:] # Nuc
    nd[1,:,:] = aer.field34103.values[:,:] # Ait
    nd[2,:,:] = aer.field34107.values[:,:] # Acc
    nd[3,:,:] = aer.field34113.values[:,:] # Coa
    #nd[4,:,:] = aer.field34119.values[:,:]  # Ait Insol ** Don't want to use the insol for CCN

    rbardry[0,:,:] = (aer.field38401.values[:,:] / 2)
    rbardry[1,:,:] = (aer.field38402.values[:,:] / 2)
    rbardry[2,:,:] = (aer.field38403.values[:,:] / 2)
    rbardry[3,:,:] = (aer.field38404.values[:,:] / 2)
    #rbardry[4,:,:] = (aer.field38405.values[:,:] / 2)
    

    #CCN40
    for imode in range(4):
        nd_lt_d = lognormal_cumulative_to_r(nd[imode,:],20e-9, rbardry[imode,:], sigma_g[imode])
        nd_gt_d = nd[imode,:,:] - nd_lt_d
        
        if imode == 0: 
            CCN40 = nd_gt_d
        else: 
            CCN40 = CCN40 + nd_gt_d
    
    CCN40 = xr.DataArray(CCN40,coords=[aer.time,aer[height]],dims=['time',height],name='CCN40')
    aer = aer.assign(CCN40=CCN40)
    aer['CCN40'][:] = CCN40.values
    

    #CCN50 
    for imode in range(4):
        nd_lt_d = lognormal_cumulative_to_r(nd[imode,:],25e-9, rbardry[imode,:], sigma_g[imode])
        nd_gt_d = nd[imode,:,:] - nd_lt_d
        
        if imode == 0: 
            CCN50 = nd_gt_d
        else: 
            CCN50 = CCN50 + nd_gt_d
    
    CCN50 = xr.DataArray(CCN50,coords=[aer.time,aer[height]],dims=['time',height],name='CCN50')
    aer = aer.assign(CCN50=CCN50)
    aer['CCN50'][:] = CCN50.values
    
    #CCN60 
    for imode in range(4):
        nd_lt_d = lognormal_cumulative_to_r(nd[imode,:],30e-9, rbardry[imode,:], sigma_g[imode])
        nd_gt_d = nd[imode,:,:] - nd_lt_d
        
        if imode == 0: 
            CCN60 = nd_gt_d
        else: 
            CCN60 = CCN60 + nd_gt_d
    
    CCN60 = xr.DataArray(CCN60,coords=[aer.time,aer[height]],dims=['time',height],name='CCN60')
    aer = aer.assign(CCN60=CCN60)
    aer['CCN60'][:] = CCN60.values    
    return aer


# In[7]:


def calculate_size_distributions(nmodes,nd,rbardry,t):
    sigma_g = [1.59,1.59,1.4,2.0,1.59,1.59,2.0]

    # determine which modes are active
    mode = np.zeros((nmodes),dtype=bool)
    
    for imode in range(nmodes):
        mode[imode] = np.isfinite(nd[imode,:].any()) # if any data in array, mode is active
    
    # define points for calculating size distribution
    npts = 50 # number of bins into which interpolate modal output
    rmin = 1.0e-9
    rmax = 1.0e-5
    dryr_mid = np.zeros(npts)
    dryr_int = np.zeros(npts+1)
    for ipt in range (npts+1):
        logr = np.log(rmin)+(np.log(rmax)-np.log(rmin))*np.float(ipt)/np.float(npts)
        dryr_int[ipt] = np.exp(logr)
    for ipt in range (npts):
        dryr_mid[ipt] = 10.0**(0.5*(np.log10(dryr_int[ipt+1])+np.log10(dryr_int[ipt]))) # in m
        
    dndlogd = np.zeros((nmodes+1,npts)) # number of modes, plus total number    
        
    for ipt in range (npts):
        for imode in range (nmodes):
            
            if (mode[imode]):
            
                dndlogd[imode,ipt] = lognormal_dndlogd(nd[imode,t],
                                                       dryr_mid[ipt]*2,
                                                       rbardry[imode,t]*2,
                                                       sigma_g[imode])
            else:
                dndlogd[imode,ipt] = np.nan
            
        dndlogd[nmodes,ipt] = np.sum(dndlogd[0:nmodes,ipt])
        
    return dndlogd,dryr_mid


# In[8]:


def lognormal_dndlogd(n,d,dbar,sigma_g):

    # evaluates lognormal distribution dn/dlogd at diameter d
    # dndlogd is the differential wrt the base10 logarithm

    xpi = 3.14159265358979323846e0

    numexp = -(np.log(d)-np.log(dbar))**2.0
    denomexp = 2.0*np.log(sigma_g)*np.log(sigma_g)

    denom = np.sqrt(2.0*xpi)*np.log(sigma_g)

    dndlnd = (n/denom)*np.exp(numexp/denomexp)

    dndlogd = 2.303*dndlnd

    return dndlogd


# In[9]:


def generate_size_dists(aer,shipname,tstep):
    nsteps = len(aer.time)
    nmodes = 7
    nd = np.zeros((nmodes,nsteps))
    rbardry = np.zeros ((nmodes,nsteps))

    nd[0,:] = aer.field34101.values[:,0] # Nuc
    nd[1,:] = aer.field34103.values[:,0] # Ait
    nd[2,:] = aer.field34107.values[:,0] # Acc
    nd[3,:] = aer.field34113.values[:,0] # Coa
    nd[4,:] = aer.field34119.values[:,0]  # Ait Insol

    rbardry[0,:] = (aer.field38401.values[:,0] / 2)
    rbardry[1,:] = (aer.field38402.values[:,0] / 2)
    rbardry[2,:] = (aer.field38403.values[:,0] / 2)
    rbardry[3,:] = (aer.field38404.values[:,0] / 2)
    rbardry[4,:] = (aer.field38405.values[:,0] / 2)
    dnd = np.zeros((50,nsteps))

    for i in range(nsteps):

        dndlogd,dryr_mid = calculate_size_distributions(nmodes,nd,rbardry,i)
        dnd[:,i] = dndlogd[7,:]

    dryr_mid = dryr_mid*2.*1.0e9 # m to nm, radius to diameter
    sizedist = xr.DataArray(dnd,coords=[dryr_mid,aer.time],dims=['Dry Diameter','time']).to_dataset(name='sizedist')
    lat = xr.DataArray(aer.lat,coords=[aer.time],dims=['time']).to_dataset()
    lon = xr.DataArray(aer.lon,coords=[aer.time],dims=['time']).to_dataset()

    da = xr.merge([sizedist,lat,lon])
    da['sizedist'] = da['sizedist'].assign_attrs({'Units':'dN / dlogD (cm-3)'})
    da['Dry Diameter'] = da['Dry Diameter'].assign_attrs({'Units':'nm'})
    da = da.assign_attrs({'history':'Data extracted along {} ship track on {}'.format(
        shipname,datetime.now().date())})
    fout = '/g/data/jk72/slf563/ACCESS/output/campaign_data/'
    return da


# In[11]:


def read_obs(name):
    f_path = '/g/data/jk72/slf563/OBS/campaigns/cleaned_SO_campaign_data/'
    data = pd.read_csv(f_path+'aerosol - {} - daily means from hrly data.csv'.format(name),index_col=0)
    data.index = pd.DatetimeIndex(data.index)
    data.index = data.index.rename('time')
    data = data.rename(columns={'lat':'latitude','lon':'longitude'})
    if name == 'Kennaook-Cape Grim': 
        data['latitude'] = -40.68333
        data['longitude'] = 144.6833
    if name == 'Macquarie Island': 
        data['latitude'] = -54.38
        data['longitude'] = 158.4
    if name == 'Syowa': 
        data['latitude'] = -69.00
        data['longitude'] = 39.59
    if name == 'Kennaook-Cape Grim offset': 
        data['latitude'] = -41.93333
        data['longitude'] = 146.5583        
    return data[['latitude','longitude']]


# In[12]:


def read_ACCESS_along_ship_track(ship,job,file):

    period='daily'
    time=ship.index
           
    fdir = '/g/data/jk72/slf563/ACCESS/output/{}/{}/'.format(job,period)

    for t in time:
        
        fname = '{}a.p{}{}{:02d}{:02d}.nc'.format(job,file,t.year,t.month,t.day)
        data = xr.open_dataset(fdir+fname)
        if 'time1' in list(data.dims): data = remove_time1(data)
            
        shplat=ship.latitude.loc[t]
        shplon=ship.longitude.loc[t]

        d = data.interp(lat=shplat, lon=shplon)
        if 'lat_v' in data.dims:
            d = d.interp(lat_v=shplat)
            d = d.interp(lon_u=shplon)
            d = d.drop(('lon_u','lat_v'))
            d = d.reset_coords(('lat','lon'))
        d['lat'] = d.lat.expand_dims(dim='time')
        d['lon'] = d.lon.expand_dims(dim='time')
        if t == time[0] :
            varout = d
        else: 
            varout = xr.concat([varout,d],dim='time')

    return varout


# In[13]:


def get_met(ship,job,file,tstep,shipname):
    met = read_ACCESS_along_ship_track(ship,job,file)
    if file == 'h.met':
        u = read_ACCESS_along_ship_track(ship,job,'h.u')
        met = xr.merge([met,u])
    if 'ta' in list(met.variables): 
        density = (met.field408/(met.ta*zboltz*1.0E6))
        met = xr.merge([{'density':density}])
    else: 
        ta = met.theta*((met.field408/p0)**Rd_cp)
        density = (met.field408/(ta*zboltz*1.0E6))
        met = xr.merge([met,{'ta':ta,'density':density}])
    met = met.assign_attrs({'history':'Data extracted along {} ship track on {}'.format(
        shipname,datetime.now().date())})
    
    return met


# In[14]:


def get_aer(met,ship,job,file,tstep,shipname):
    aer = read_ACCESS_along_ship_track(ship,job,file)
    # check that aer & met have same height coords 
    if 'z3_hybrid_height' in aer.coords:
        if (aer.z3_hybrid_height.values-met.z0_hybrid_height.values).mean()==0:
            aer = aer.rename_dims({'z3_hybrid_height':'z0_hybrid_height'})
            aer = aer.rename_vars({'z3_hybrid_height':'z0_hybrid_height'})
    aer = aero_unit_conversions(aer,met)
    aer = nt10_calcs(aer)
    aer = CCN_calcs(aer)
    da = generate_size_dists(aer,shipname,tstep)
    aer = aer.assign_attrs({'history':'Data extracted along {} ship track on {}'.format(
        shipname,datetime.now().date())})
    
    return aer,da


# In[15]:


def run_processing(ship,job,shipname): 
    assert job in ['dg657','bx400','cg283','ch543','cq687','cq686','dd153','dd154','cg893'], 'Unknown job'
    if (job in ['dg657','bx400','cg283','ch543','cq687','cq686','dd153','dd154']): # offline chemistry jobs
        tstep = 'daily'
        met = get_met(ship,job,'d.glob',tstep,shipname)
        aer,da = get_aer(met,ship,job,'d.sh',tstep,shipname)
    elif (job == 'cg893'): # full chemistry run - files set up slightly differently 
        tstep = 'daily'
        ship = ship.resample('1D',kind='timestamp').mean()
        met = get_met(ship,job,'met',tstep,shipname)
        aer,da = get_aer(met,ship,job,'aer',tstep,shipname)
        
    return da, met, aer


# # Pull out data
# - dg657 - control
# - bx400 - control with old DMS
# - cg283 - BLN on
# - ch543 - OM2 DMS
# - cq687 - Rev3 DMS
# - cq686 - PMO on
# - dd153 - PMO + 1.0x Rev3 DMS
# - dd154 - sea spray wind gusts
#   

# In[16]:


#obs_names = ['CAPRICORN1','CAPRICORN2','MARCUS','CAMMPCAN','Ice2Equator',
#              'Cold Water Trial','PCAN','Kennaook-Cape Grim','Macquarie Island',
#              'Syowa'] 
jobs = ['dg657','bx400','cg283','ch543','cq687','cq686','dd153','dd154']


# In[ ]:


obs_names = ['MARCUS']


# In[ ]:


for shipname in obs_names: 
    ship = read_obs(shipname)
    
    if ship.index[0] < pd.to_datetime('2015-01-01'): 
        ship = ship.loc['2015-01-01':]
    if ship.index[-1] > pd.to_datetime('2019-12-30'): # model data cuts out on this day... 
        ship = ship.loc[:'2019-12-30']                
        
    for job in jobs: 

        if job == 'ch543': 
            if ship.index[-1] > pd.to_datetime('2018-12-30'): 
                ship = ship.loc[:'2018-12-30']
            if pd.to_datetime('2015-05-16') in ship.index:
                ship = ship.drop('2015-05-16') # for some reason this day is missing from model data now. 
        
        da, met, aer = run_processing(ship,job,shipname) 
        met = met.merge(aer)
        if job == 'cg893': 
            da.load().to_netcdf(fout+'{}_{}_{}_size_distributions.nc'.format(job,tstep,shipname))
        else: 
            met = met.merge(da,compat='override')
        met.to_netcdf('/g/data/jk72/slf563/ACCESS/output/campaign_data/{}_daily_mean_{}_vars.nc'.format(job,shipname))
        print('Finished',job)
    print('Finished',shipname)    


# In[ ]:




