
import astropy.io.fits as fits
import fitsio
import numpy as np
import time
import pyccl as ccl
from more_itertools import locate
import treecorr
from datetime import date
import subprocess 
import sys
import glob
import os

data_dir = '/home/b7009348/WGL_project/DES-data/'

if not os.path.exists(data_dir+'gold_catalogues'):
    os.mkdir(data_dir+'gold_catalogues')

def run_subprocess(cmd, cwd=None, env=None):
    '''A more sophistacted version of subprocess.check_call
    which can handle errors in a more informative way.'''
    P = subprocess.Popen(cmd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
    l = True
    while l:
        l = P.stdout.read(1)
        sys.stdout.write(l)
    P.wait()
    if P.returncode:
        raise subprocess.CalledProcessError(returncode=P.returncode,
                cmd=cmd)
        
def download_gold():
    '''This function is used to download the entire DES Y1 gold catalogue'''
    cmd = ['wget', '--quiet', '-m', '-P', data_dir+'/gold_catalogues', '-nH', '--cut-dirs=3', '-np', '-e',
           'robots=off', 'http://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/']
    run_subprocess(cmd)
    
def cut_gold(filename, flag_value=0):
    '''Created in addition to cut_flags. cut_flags will also cut based on gold flags, however, this function
    can be used on files that have already been cut on their own flags or on other selections.'''
    
    start = time.time()
    print('Opening files...')
    
    # create a list of all gold catalogue files
    gold_files = glob.glob(data_dir+'gold_catalogues/*.fits')
    
    print('Locating gold flags...')
    # loop through files and locate good flags, appending object IDs to list
    gold_ids = []
    old_ids = []
    for i in range(len(gold_files)):
        # open gold file
        with fits.open(gold_files[i]) as hdu:
            gold_data = hdu[1].data
        # locate good flags
        gold_indexes = list(locate(gold_data['FLAGS_GOLD'], lambda x: x == flag_value))
        new_ids = gold_data['COADD_OBJECTS_ID'][gold_indexes]
        # append object IDs cut on good flags
        gold_ids = np.append(old_ids, new_ids)
        old_ids = gold_ids
        del gold_data, gold_indexes, new_ids, old_ids
    
    with fits.open(data_dir+filename) as hdu:
        data = hdu[1].data
        shape_ids = data['coadd_objects_id']
    
    # match IDs for good gold sources to remaining IDs in shape catalogue
    matches, cat_indexes, gold_indexes = np.intersect1d(shape_ids, gold_ids, return_indices=True)
    
    print('Flags located, slicing data...')
    data = data[cat_indexes]
    del matches, cat_indexes, gold_indexes
    
    print('Saving to new file...')
    fits.writeto(data_dir+'gold_cut_'+filename, data)
    
    end = time.time()
    diff = end-start
    print('Runtime: %g'%diff)

def cut_flags(filename, method, flag_value=0):
    '''Function to make cuts to DES Y1 shape data based on flag values. Default flag is 0
    which defines a good source to use. Requires gold catalogues to be downloaded as well.'''
    
    start = time.time()
    print('Opening file...')
    
    # open the shape file
    with fits.open(data_dir+filename) as hdul:
        data = hdul[1].data
    
    # locate good flags in shape file
    print('Locating flags...')
    indexes = list(locate(data['flags_select'], lambda x: x == flag_value))
    
    # cut data on good flags in shape file
    print('Flags located, slicing data...')
    data = data[indexes]
    del indexes
    
    # create a list of all gold catalogue files
    gold_files = sorted(glob.glob(data_dir+'gold_catalogues/*.fits'))
    
    print('Locating gold flags...')
    # loop through files and locate good flags, appending object IDs to list
    gold_ids = []
    old_ids = []
    for i in range(len(gold_files)):
        # open gold file
        with fits.open(gold_files[i]) as hdu:
            gold_data = hdu[1].data
        # locate good flags
        gold_indexes = list(locate(gold_data['FLAGS_GOLD'], lambda x: x == flag_value))
        new_ids = gold_data['COADD_OBJECTS_ID'][gold_indexes]
        # append object IDs cut on good flags
        gold_ids = np.append(old_ids, new_ids)
        old_ids = gold_ids
    del gold_data, gold_indexes, new_ids, old_ids
    
    # match IDs for good gold sources to remaining IDs in shape catalogue
    matches, gold_indexes, cat_indexes = np.intersect1d(gold_ids, data['coadd_objects_id'], return_indices=True)
    
    print('Flags located, slicing data...')
    data = data[cat_indexes]
    del matches, cat_indexes, gold_indexes
        
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
    zindexes = list(locate(zdata['z_mc'], lambda x: x > zmin and x < zmax))
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
    
def match_catalogues(im3file, mcalfile, zfile):
    '''Function to find and match obejcts in both mcal and im3
    catalogues and slice the data so only those sources in both 
    catalogues remain. Must be run on files already cut on flags.'''
    
    start = time.time()
    
    print('Opening files...')  
    
    with fits.open(data_dir+zfile) as hdu:
        data= hdu[1].data
    z_mc = data['z_mc']
    zIDs = data['coadd_objects_id']
    del data
    
    print('Locating NaNs...')
    indexes = list(locate(z_mc, lambda x: np.isnan(x) == False))
    if len(indexes) == len(zIDs):
        print('No NaNs found')
    else:
        zIDs = zIDs[indexes]
    del z_mc, indexes
    
    with fits.open(data_dir+im3file) as hdul:
        im3data = hdul[1].data
    im3IDs = im3data['coadd_objects_id']
    
    matches, im3Indices, zIndices = np.intersect1d(im3IDs, zIDs, return_indices=True)
    del matches, zIDs, zIndices
    
    print('Slicing NaNs...')
    im3data = im3data[im3Indices]
    im3IDs = im3data['coadd_objects_id']
    del im3Indices

    print('Matching sources...') 
    with fits.open(data_dir+mcalfile) as mcalhdu:
        mcaldata = mcalhdu[1].data        
    mcalIDs = mcaldata['coadd_objects_id']
    del mcaldata
    
    print('Finding ID intersections between catalogues...')
    matches, im3indices, mcalindices = np.intersect1d(im3IDs, mcalIDs, return_indices=True)
    print('%d sources matched'%(np.size(matches)))
    del matches, im3IDs, mcalIDs
    
    print('Slicing im3 data...')
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
    '''Function takes source and lens redshift values (in the form of an array, not fits file) and
    uses them to calculate the parameter F, which represents the fraction of the sample we expect to
    be intrinsically aligned'''
    
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

def im3_tang_shear(lens_cat, source_cat, sens_cat, rand_cat, sep_bins, theta_min, theta_max, bin_slop=0.1):
    '''Function to calculate tangential shear based on im3shape measurements'''
    
    # preallocate output arrays
    gammat = np.zeros([sep_bins])
    theta = np.zeros([sep_bins])
    
    # carry out tangential shear calculation with lenses
    ng = treecorr.NGCorrelation(nbins=sep_bins, min_sep=theta_min, max_sep=theta_max, sep_units='arcmin', bin_slop=bin_slop)
    ng.process(lens_cat, source_cat)
    # calculate multiplicative bias correction with lenses
    nk = treecorr.NKCorrelation(nbins=sep_bins, min_sep=theta_min, max_sep=theta_max, sep_units='arcmin', bin_slop=bin_slop)
    nk.process(lens_cat, sens_cat)
    
    # carry out tangential shear calculation with randoms
    rg = treecorr.NGCorrelation(nbins=sep_bins, min_sep=theta_min, max_sep=theta_max, sep_units='arcmin', bin_slop=bin_slop)
    rg.process(rand_cat, source_cat)
    # calculate multiplicative bias correction with randoms
    rk = treecorr.NKCorrelation(nbins=sep_bins, min_sep=theta_min, max_sep=theta_max, sep_units='arcmin', bin_slop=bin_slop)
    rk.process(rand_cat, sens_cat)
 
    # collect outputs 
    xi_l = ng.xi
    sens_l = nk.xi
    xi_r = rg.xi
    sens_r = rk.xi
    
    # calculate and save tangential shear + theta
    gammat = xi_l/sens_l - xi_r/sens_r
    theta = np.exp(ng.meanlogr)
    
    return gammat, theta

def mcal_tang_shear(lens_cat, source_cat, rand_cat, Rsp, sep_bins, theta_min, theta_max, bin_slop=0.1):
    '''Function to calculate tangential shear based on metacalibration measurments'''
    
    # preallocate output arrays
    gammat = np.zeros([sep_bins])
    theta = np.zeros([sep_bins])
    
    # calculate tangential shear with lenses
    ng = treecorr.NGCorrelation(nbins=sep_bins, min_sep=theta_min, max_sep=theta_max, sep_units='arcmin', bin_slop=bin_slop)
    ng.process(lens_cat, source_cat)
    
    # calculate tangential shear with randoms
    rg = treecorr.NGCorrelation(nbins=sep_bins, min_sep=theta_min, max_sep=theta_max, sep_units='arcmin', bin_slop=bin_slop)
    rg.process(rand_cat, source_cat)
    
    xi_l = ng.xi
    xi_r = rg.xi
    
    gammat = 1.0/Rsp * (xi_l - xi_r)
    theta = np.exp(ng.meanlogr)
    
    return gammat, theta

def calculate_boost(lens_cat, rand_cat, source_cat, sep_bins, theta_min, theta_max, bin_slop=0.1):
    '''Function to calculate boost. Any shear measurement method can be used as count-count correlations
    do not depend on estimated shear, only position.'''
    
    # preallocate array
    boost = np.zeros(sep_bins)
    
    ls = treecorr.NNCorrelation(nbins=sep_bins, min_sep=theta_min, max_sep=theta_max, sep_units='arcmin', bin_slop=bin_slop)
    ls.process(lens_cat, source_cat)
    
    rs = treecorr.NNCorrelation(nbins=sep_bins, min_sep=theta_min, max_sep=theta_max, sep_units='arcmin', bin_slop=bin_slop)
    rs.process(rand_cat, source_cat)

    nrand = rand_cat.nobj
    nlens = lens_cat.nobj
    
    boost = nrand/nlens * ls.weight/rs.weight
    
    return boost

def IA_jackknife(cat_l, cat_r, cat_im3, cat_mcal, cat_k, npatches, sep_bins, fbins, theta_min, theta_max, bin_slop=0.1):
    '''This function produces jackknife estimates of the IA signal with associated errors. It should be used in
    conjunction with IA_full() to obtain a full measurment with errors. The function should be provided with catalogues that have predefined
    jackknife patches using treecorrs catalogue attributes npatches and patch_centers. To ensure the patches are set up correctly consult
    the treecorr documentation. The metacalibration catalogue should have R11 response values saved in the redshift attribute, the im3shape catalogue should have
    metacalibrtion photometric redshifts (z_mc) saved in the redshift attribute, and the randoms catalogue should have the randoms redshifts provided in the redshift
    attribute. This does not affect the calculation as the they are cut out of the catalogues used in the tangential shear calculation and only used to apply the
    metaclaibration response and calculate the parameter F on a per-patch basis.'''
    
    IA_patches = np.zeros([npatches, sep_bins])
    im3_patches = np.zeros([npatches, sep_bins])
    mcal_patches = np.zeros([npatches, sep_bins])
    boost_patches = np.zeros([npatches, sep_bins])
    F_patches = np.zeros([npatches])

    for i in range(npatches):
    
        start = time.time()
        
        theta = np.zeros([sep_bins])

        l_indexes = list(locate(cat_l.patch, lambda x: x != i))
        r_indexes = list(locate(cat_r.patch, lambda x: x != i))
        mcal_indexes = list(locate(cat_mcal.patch, lambda x: x != i))
        im3_indexes = list(locate(cat_im3.patch, lambda x: x!= i))
        k_indexes = list(locate(cat_k.patch, lambda x: x != i))

        temp_l = treecorr.Catalog(ra=cat_l.ra[l_indexes], dec=cat_l.dec[l_indexes], 
                                  ra_units='rad', dec_units='rad', w=cat_l.w[l_indexes])

        temp_r = treecorr.Catalog(ra=cat_r.ra[r_indexes], dec=cat_r.dec[r_indexes],
                                 ra_units='rad', dec_units='rad')

        temp_mcal = treecorr.Catalog(ra=cat_mcal.ra[mcal_indexes], dec=cat_mcal.dec[mcal_indexes], 
                                     ra_units='rad', dec_units='rad', g1=cat_mcal.g1[mcal_indexes],
                                    g2=cat_mcal.g2[mcal_indexes])

        temp_im3 = treecorr.Catalog(ra=cat_im3.ra[im3_indexes], dec=cat_im3.dec[im3_indexes],
                                   ra_units='rad', dec_units='rad', g1=cat_im3.g1[im3_indexes],
                                   g2=cat_im3.g2[im3_indexes], w=cat_im3.w[im3_indexes])

        temp_k = treecorr.Catalog(ra=cat_k.ra[k_indexes], dec=cat_k.dec[k_indexes], ra_units='rad',
                                 dec_units='rad', k=cat_k.k[k_indexes], w=cat_k.w[k_indexes])

        R = np.mean(cat_mcal.r[mcal_indexes])

        rand_z = cat_r.r[r_indexes]
        source_z = cat_im3.r[im3_indexes]
        source_weights = cat_im3.w[im3_indexes]

        del l_indexes, r_indexes, mcal_indexes, im3_indexes, k_indexes

        print('Patch %g located and sliced, calculating correlations...'%i)

        mcal_patches[i,:], theta = mcal_tang_shear(lens_cat=temp_l, source_cat=temp_mcal, rand_cat=temp_r, Rsp=R, sep_bins=sep_bins, theta_min=theta_min, theta_max=theta_max, bin_slop=bin_slop)

        im3_patches[i,:], theta = im3_tang_shear(lens_cat=temp_l, source_cat=temp_im3, rand_cat=temp_r, sens_cat=temp_k, sep_bins=sep_bins, theta_min=theta_min, theta_max=theta_max, bin_slop=bin_slop)

        boost_patches[i,:] = calculate_boost(lens_cat=temp_l, rand_cat=temp_r, source_cat=temp_im3, sep_bins=sep_bins, theta_min=theta_min, theta_max=theta_max, bin_slop=bin_slop)

        F_patches[i] = calculate_F(nbins=fbins, source_z=source_z, lens_z=rand_z, source_weights=source_weights)

        IA_patches[i,:] = (im3_patches[i,:] - mcal_patches[i,:]) / (boost_patches[i,:] - 1.0 + F_patches[i])

        end = time.time()
        diff = end - start

        print('IA signal estimated, runtime=%f.'%diff)

        del temp_l, temp_r, temp_mcal, temp_im3, temp_k, source_z, rand_z, source_weights, R
    
    IA_jk = np.zeros([sep_bins])
    IA_cov = np.zeros([sep_bins, sep_bins])
    
    for i in range(sep_bins): 
        bin_patches = IA_patches[:,i] 
        IA_jk[i] = 1.0/npatches * np.sum(bin_patches)
        
    for i in range(sep_bins):
        for j in range(sep_bins):
            IA_cov[i,j] = (npatches-1.0)/npatches * np.sum((IA_patches[:,i] - IA_jk[i]) * (IA_patches[:,j] - IA_jk[j]))
    
    IA_sig = np.sqrt(np.diag(IA_cov))
        
    np.savez(file=data_dir+'ia_jackknife_values-bin_slop=%g'%bin_slop, IA=IA_patches, IA_cov=IA_cov, im3=im3_patches, mcal=mcal_patches, boost=boost_patches, F=F_patches)
        
    del bin_patches, IA_patches, im3_patches, mcal_patches, theta, boost_patches, F_patches
        
    return IA_jk, IA_sig

def IA_full(cat_l, cat_r, cat_im3, cat_mcal, cat_k, sep_bins, fbins, theta_min, theta_max, bin_slop=0.1):
    '''This function estimates the IA signal using the entire sample of sources and lenses at once with no patches.
    outputs should be taken as the datapoints for an IA measurement, combined with the errors obtained from IA_jackknife().'''
    
    start = time.time()
    
    full_l = treecorr.Catalog(ra=cat_l.ra, dec=cat_l.dec, 
                                  ra_units='rad', dec_units='rad', w=cat_l.w)
    
    full_r = treecorr.Catalog(ra=cat_r.ra, dec=cat_r.dec,
                                 ra_units='rad', dec_units='rad')
    
    full_im3 = treecorr.Catalog(ra=cat_im3.ra, dec=cat_im3.dec,
                                   ra_units='rad', dec_units='rad', g1=cat_im3.g1,
                                   g2=cat_im3.g2, w=cat_im3.w)
    
    full_mcal = treecorr.Catalog(ra=cat_mcal.ra, dec=cat_mcal.dec, 
                                     ra_units='rad', dec_units='rad', g1=cat_mcal.g1,
                                    g2=cat_mcal.g2)
    
    full_k = treecorr.Catalog(ra=cat_k.ra, dec=cat_k.dec, ra_units='rad',
                                 dec_units='rad', k=cat_k.k, w=cat_k.w)

    R = np.mean(cat_mcal.r)

    rand_z = cat_r.r
    source_z = cat_im3.r
    source_weights = cat_im3.w

    IA_final = np.zeros([sep_bins])

    mcal, theta = mcal_tang_shear(lens_cat=full_l, source_cat=full_mcal, rand_cat=full_r, Rsp=R, sep_bins=sep_bins, theta_min=theta_min, theta_max=theta_max, bin_slop=bin_slop)

    im3, theta = im3_tang_shear(lens_cat=full_l, source_cat=full_im3, rand_cat=full_r, sens_cat=full_k, sep_bins=sep_bins, theta_min=theta_min, theta_max=theta_max, bin_slop=bin_slop)

    boost = calculate_boost(lens_cat=full_l, rand_cat=full_r, source_cat=full_im3, sep_bins=sep_bins, theta_min=theta_min, theta_max=theta_max, bin_slop=bin_slop)

    F = calculate_F(nbins=fbins, lens_z=rand_z, source_z=source_z, source_weights=source_weights)

    IA_final = (im3 - mcal) / (boost - 1.0 + F)
    
    np.savez(file=data_dir+'ia_full_values-bin_slop=%g'%bin_slop, IA=IA_final, im3=im3, mcal=mcal, boost=boost, F=F)

    end = time.time()
    diff = end-start

    print('Full signal estimated, runtime =%f.'%diff)

    del full_l, full_r, full_im3, full_mcal, full_k, F, boost, im3, mcal, rand_z, source_z, source_weights, R
    
    return IA_final, theta

def create_patches(lens_file, npatches):
    
    start=time.time()

    with fits.open(data_dir+lens_file) as hdu:
        data = hdu[1].data
        ra_l = data['RA']
        dec_l = data['DEC']
        w_l = data['weight']
    del data

    cat_l = treecorr.Catalog(ra=ra_l, dec=dec_l, ra_units='deg', dec_units='deg', w=w_l, npatch=npatches)
    
    cat_l.write_patch_centers(data_dir+'jackknife_patch_centers')

    end = time.time()
    diff=end-start
    
    del cat_l, ra_l, dec_l, w_l

    print('Patches saved, runtime=%f.'%diff)