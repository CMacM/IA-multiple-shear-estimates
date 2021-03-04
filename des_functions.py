import astropy.io.fits as fits
import numpy as np
import time
from more_itertools import locate

data_dir = '/home/b7009348/projects/WGL_Project/DES-data/'

def cut_flags(filename, method, flag_value=0):
    '''Function to make cuts to DES Y1 shape data based on flag values. Default flag is 0
    which defines a good source to use'''
    start = time.time()
    print('Opening file...')
    
    with fits.open(data_dir+filename) as hdul:
        data = hdul[1].data
        
    print('Locating flags...')
    indexes = list(locate(data['flags_select'], lambda x: x == flag_value))
    
    print('Flags located, slicing data...')
    data = data[indexes]
    del indexes
    
    print('Data sliced, writing to new file...')
    fits.writeto(data_dir+'y1_'+method+'_flags=0.fits', data)
    
    end = time.time()
    print('Runtime: %g'%(end-start))
    
def cut_redshift(shapefile, zfile, method, zmin, zmax, flag_value=0):
    '''Function to make cuts to DES Y1 data based on the redshift
    of source in a provided redshift range. MUST USE ORGINAL DATA FILE SO
    INDEXES ARE PRESERVED'''
    start = time.time()
    print('Opening files...')  
    
    with fits.open(data_dir+zfile) as zhdul:
        zdata = zhdul[1].data
        
    print('Locating sources in range %g - %g...'%(zmin,zmax))
    zindexes = list(locate(zdata['MEAN_Z'], lambda x: x >= zmin and x <= zmax))
    del zdata #remove redshift data to free up memory
    
    print('Sources in range found, slicing data...')    
    with fits.open(data_dir+shapefile) as hdul:
        data = hdul[1].data
    data = data[zindexes]
    del zindexes #remove redshift range indexes to free up memory
    
    print('Redshift range sliced, locating flags...')
    indexes = list(locate(data['flags_select'], lambda x: x == flag_value))
    
    print('Flags located, slicing shape data...')
    data = data[indexes]
    del indexes #delete flag indexes to free up memory
    
    print('Data sliced, writing to new file...')
    fits.writeto(data_dir+'y1_'+method+'_z=%g-%g.fits'%(zmin,zmax), data)
    
    end = time.time()
    print('Runtime: %g'%(end-start))
            
def cut_zbin(shapefile, zfile, method, zbin, flag_value=0):
    '''Function to make cuts to DES Y1 data based on the redshift
    bin flags. MUST USE ORGINAL DATA FILE SO
    INDEXES ARE PRESERVED. Method must be provided as im3 or mcal'''
    start = time.time()
    print('Opening files...')  
    
    with fits.open(data_dir+zfile) as zhdul:
        zdata = zhdul[1].data
        
    print('Locating sources in bin %g...'%(zbin))
    zindexes = list(locate(zdata['zbin_'+method], lambda x: x == zbin))
    del zdata #remove redshift data to free up memory
    
    print('Sources in range found, slicing data...')    
    with fits.open(data_dir+shapefile) as hdul:
        data = hdul[1].data
    data = data[zindexes]
    del zindexes #remove redshift range indexes to free up memory
    
    print('Redshift range sliced, locating flags...')
    indexes = list(locate(data['flags_select'], lambda x: x == flag_value))
    
    print('Flags located, slicing shape data...')
    data = data[indexes]
    del indexes #delete flag indexes to free up memory
    
    print('Data sliced, writing to new file...')
    fits.writeto(data_dir+'y1_'+method+'_zbin=%g.fits'%(zbin), data)
    
    end = time.time()
    print('Runtime: %g'%(end-start))
    
def correct_additive_bias(filename, method, zrange):
    '''Function to apply additive bias correction to e1 and 
    e2. zrange should be supplied as a string for file naming'''
    start = time.time()
    print('Opening files...')
    
    with fits.open(data_dir+filename) as hdu:
        data = hdu[1].data
        
    print('Applying additive bias correction...')
    data['e1'] = (data['e1'] - data['c1']) / (1.0 + data['m'])
    data['e2'] = (data['e2'] - data['c2']) / (1.0 + data['m'])
    
    print('Correction applied, saving...')
    fits.writeto(data_dir+'y1_'+method+'_corrected_z='+zrange+'.fits', data)
    
def match_catalogues(im3file, mcalfile):
    '''Function to find and match obejcts in both mcal and im3
    catalogues and slice the data so only those sources in both 
    catalogues remain. Must be run on files already cut on flags.'''
    start = time.time()
    print('Opening files and collecting IDs...')
    with fits.open(data_dir+im3file) as im3hdu:
        im3data = im3hdu[1].data        
    im3IDs = im3data['coadd_objects_id']
    del im3data
    
    with fits.open(data_dir+mcalfile) as mcalhdu:
        mcaldata = mcalhdu[1].data        
    mcalIDs = mcaldata['coadd_objects_id']
    del mcaldata
    
    print('Finding ID intersections between catalogues...')
    matches, im3indices, mcalindices = np.intersect1d(im3IDs, mcalIDs, return_indices=True)
    del matches, im3IDs, mcalIDs
    
    print('Slicing im3 data...')
    with fits.open(data_dir+im3file) as im3hdu:
        im3data = im3hdu[1].data 
    im3data = im3data[im3indices]
    del im3indices
    
    print('Saving im3 data to new file...')
    fits.writeto(data_dir+'y1_im3_shapes_matched.fits', im3data)
    del im3data
    
    print('Slicing mcal data...')
    with fits.open(data_dir+mcalfile) as mcalhdu:
        mcaldata = mcalhdu[1].data
    mcaldata = mcaldata[mcalindices]
    del mcalindices
    
    print('Saving mcal data to new file...')
    fits.writeto(data_dir+'y1_mcal_shapes_matched.fits', mcaldata)
    
    end = time.time()
    print('Runtime: %g'%(end-start))
    