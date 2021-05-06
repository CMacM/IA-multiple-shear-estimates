import astropy.io.fits as fits
import numpy as np
import time
import pyccl as ccl
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
    of source in a provided redshift range.'''
    
    start = time.time()
    print('Opening files...')  
    
    with fits.open(data_dir+zfile) as zhdul:
        zdata = zhdul[1].data
    zIDs = zdata['coadd_objects_id']
        
    print('Locating sources in range %g - %g...'%(zmin,zmax))
    zindexes = list(locate(zdata['MEAN_Z'], lambda x: x > zmin and x < zmax))
    del zdata #remove redshift data to free up memory
    
    print('Sources in range found, slicing data...')
    zIDs = zIDs[zindexes]
    del zindexes
    
    print('Matching redshifts to catalogue...')
    with fits.open(data_dir+shapefile) as hdul:
        data = hdul[1].data
    catIDs = data['coadd_objects_id']
    
    matches, catIndices, zIndices = np.intersect1d(catIDs, zIDs, return_indices=True)
    del matches, catIDs, zIDs, zIndices
    
    print('Slicing catalogue data...')
    data = data[catIndices]
    del catIndices
    
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
    
def correct_bias(filename, method, zrange):
    '''Function to apply additive bias correction to e1 and 
    e2. zrange should be supplied as a string for file naming'''
    
    start = time.time()
    print('Opening files...')
    
    with fits.open(data_dir+filename) as hdu:
        data = hdu[1].data
        
    print('Applying additive bias correction...')
    data['e1'] = (data['e1'] - data['c1'])
    data['e2'] = (data['e2'] - data['c2'])
    
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

def cut_lenses(lensfile, zmin, zmax):
    '''Function to make redshift cuts to DES lenses.
    Tomographic cuts should be made as in the offical DES analysis,
    otherwise luminosity and sample mixing may occur.'''
    
    start = time.time()
    print('Opening files...')  
    with fits.open(data_dir+lensfile) as hdul:
        data = hdul[1].data
        
    print('Locating lenses in range %g - %g...'%(zmin,zmax))
    indexes = list(locate(data['ZREDMAGIC'], lambda x: x > zmin and x < zmax))
    
    print('Lenses in range found, slicing data...')    
    data = data[indexes]
    del indexes #remove redshift range indexes to free up memory
    
    print('Data sliced, writing to new file...')
    fits.writeto(data_dir+'DES_Y1A1_Lenses_z=%g-%g.fits'%(zmin,zmax), data)
    
    end = time.time()
    print('Runtime: %g'%(end-start))
    
def calculate_F(nbins, source_z, lens_z, source_weights):
    # set number of bins and code will bin data in range max-min zmc, incl. weights
    source_freq, source_bin_edges = np.histogram(source_z, bins=nbins, range=(source_z.min(), source_z.max()), weights=source_weights)
    lens_freq, lens_bin_edges = np.histogram(lens_z, bins=nbins, range=(lens_z.min(), lens_z.max()))
    
    # calculate bin width and find bin centers
    source_binsz = np.mean(np.diff(source_bin_edges))
    source_bin_centers = source_bin_edges[1:] - source_binsz/2.0

    lens_binsz = np.mean(np.diff(lens_bin_edges))
    lens_bin_centers = lens_bin_edges[1:] - lens_binsz/2.0
    
    # Set up cosmological parameters
    OmegaM = 0.293
    OmegaB = 0.0475
    n_s = 1.0
    sigma8 = 0.966
    Ho = 70.8
    h = Ho/100.0 # h = H0/100

    # Set up a cosmology object, we need this to do calculations
    cosmo = ccl.Cosmology(Omega_c = OmegaM-OmegaB, Omega_b = OmegaB, n_s = n_s, h = h, sigma8 = sigma8)
    
    # convert to comoving dist.
    source_chi = ccl.comoving_radial_distance(cosmo, 1.0/(1.0+source_bin_centers))
    lens_chi = ccl.comoving_radial_distance(cosmo, 1.0/(1.0+lens_bin_centers))
    
    # calculate rand,close
    old_rand_close = 0.0
    for i in range(len(source_chi)):
        for j in range(len(lens_chi)):
            if abs(source_chi[i] - lens_chi[j]) <= 100.0:
                rand_close = old_rand_close + source_freq[i] * lens_freq[j]
                old_rand_close = rand_close

    # calculate rand
    old_rand = 0.0
    for i in range(len(source_chi)):
        for j in range(len(lens_chi)):
            rand = old_rand + source_freq[i] * lens_freq[j]
            old_rand = rand
            
    # calculate F      
    F = rand_close/rand
    
    return F