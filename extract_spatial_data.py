#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import xarray as xr
from datetime import datetime
import dask 

# In[ ]:


#jobs = ['bx400','cg283','cq687','cq686','dd153','dd154','ch543','dg657]
jobs = ['dh204']

# In[ ]:


def read_spatial_fields(job): 
    fdir = '/g/data/jk72/slf563/ACCESS/output/{}/daily/'.format(job)

    ystart = 2015
    if job != 'ch543': yend = 2019
    if job == 'ch543': yend = 2018

    for year in range(ystart,yend+1): 
        print('starting', year)
        data = xr.open_mfdataset(fdir+'{}a.pd.glob{}*'.format(job,year),parallel=True)
        data = data[['rsut','rsutcs','field9203','pr','lwp','clivi','field30461','theta']]
        data = data.isel(z1_hybrid_height=0)

        data2 = xr.open_mfdataset(fdir+'{}a.pd.sh{}*'.format(job,year),parallel=True)
        data2 = data2[['field38441','field34071','field34072','field34073','field38520','field38525',
              'field38531','field38539','field17257']]

        data2['field17257'] = data2['field17257'].sum('z1_hybrid_height')
        data2['field17257'].assign_attrs({'long_name':'TOTAL DUST CONC (microg/m2)'})
        data2 = data2.isel(z1_hybrid_height=0)
        data = data.merge(data2)

        data.to_netcdf('/g/data/jk72/slf563/ACCESS/output/campaign_data/{}_spatial_fields_{}.nc'.format(job,year))
        del(data)
        del(data2)
    return


# In[ ]:


for job in jobs: 
    print('started',job,datetime.now())
    read_spatial_fields(job)
    print('finished',job,datetime.now())

# In[ ]:




