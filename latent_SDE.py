# Author: Joshua Fagin

import os
import multiprocessing
import gc
import shutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch 
from torch import nn, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Categorical
from model.model import LatentSDE, RNN_baseline
from model.TF import generate_tf
from model.TF_numpy_with_brightness import generate_tf_numpy

import datetime
import torchsde
from glob import glob
import astropy
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import corner
from scipy.fft import ifft
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import convolve, welch, sawtooth, square
from scipy.constants import c
import scipy.stats as stats
from speclite import filters
import pickle
import time
from sys import platform

# This is all for the GPR baseline
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.multitask import FixedNoiseMultiTaskGP
from botorch.fit import fit_gpytorch_model
import linear_operator.settings as linop_settings
linop_settings._fast_covar_root_decomposition._default = True
linop_settings._fast_log_prob._default = True
linop_settings._fast_solves._default = True
linop_settings.cholesky_max_tries._global_value = 6
linop_settings.max_cholesky_size._global_value = 800

# I didn't end up using this because it leads to NaN in the gradients
from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.benchmark = True
import warnings
warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')

# For debugging
#torch.autograd.set_detect_anomaly(True)

import torch.distributed as dist
import torch.nn.parallel as parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# So that we have the same validation set each time
np.random.seed(0)

# Plotting settings
size = 13
plt.rc('font', size=size)
plt.rc('axes', titlesize=size)
plt.rc('axes', labelsize=size)
plt.rc('xtick', labelsize=size)
plt.rc('ytick', labelsize=size) 
plt.rc('legend', fontsize=size)
plt.rc('figure', titlesize=size)
tick_length_major = 7
tick_length_minor = 3
tick_width = 1
#If plot then display the plots. Otherwise they are only saved
plot = False
#What type of file to save figures as e.g pdf, eps, png, etc.
save_file_format = 'pdf' 
#Path to directory where to save the results
save_path = 'results' 

#make directory to save results
os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}/recovery', exist_ok=True)

#The different parameters to predict 
parameters_keys = ['spin','log_edd','f_lamp','height','theta_inc','redshift','beta','log_mass', 'log_nu_b', 'alpha_L', 'alpha_H_minus_L', 'standard_deviation'] 
# Give redshift must be set to False. I never added this to the model.
# NOT USED
give_redshift = False
if give_redshift:
    # Redshift is not a parameter to predict since we are giving it to the model
    if 'redshift' in parameters_keys:
        parameters_keys.remove('redshift')

n_params = len(parameters_keys)    
# parameters of the driving variability
parameters_keys_driving = ['log_nu_b', 'alpha_L', 'alpha_H_minus_L', 'standard_deviation'] 

n_params = len(parameters_keys)
n_params_variability = len(parameters_keys_driving)
n_params_accretion = n_params-n_params_variability

assert(n_params_variability > 0), "Must have at least one variability parameter"
assert(n_params_accretion > 0), "Must have at least one accretion parameter"

#This is used for plotting. Variables with units for axis plotting.
plotting_labels = dict()
plotting_labels['spin'] = r'$a$'
plotting_labels['log_edd'] = r'$\log_{10}(\lambda_{\mathrm{Edd}})$'
plotting_labels['f_lamp'] = r'$f_{\mathrm{Lamp}}$'
plotting_labels['theta_inc'] = r'$\theta_{\mathrm{inc}}$ [deg]'
plotting_labels['height'] = r'$(H-R_{\mathrm{in}})/R_g$'
plotting_labels['redshift'] = r'$z$'
plotting_labels['beta'] = r'$\beta$'
plotting_labels['log_mass'] = r"$\log_{10}\left(M/M_\odot\right)$"
plotting_labels['log_nu_b'] = r'$\log_{10}(\nu_b/\mathrm{day}^{-1})$'
plotting_labels['alpha_L'] = r'$\alpha_L$'
plotting_labels['alpha_H_minus_L'] = r'$\alpha_H - \alpha_L$'
plotting_labels['standard_deviation'] = r'$\sigma$ [mag]'

#This is used for plotting. Variables without units.
plotting_labels_no_units = dict()
plotting_labels_no_units['spin'] = r'$a$'
plotting_labels_no_units['log_edd'] = r'$\log_{10}(\lambda_{\mathrm{Edd}})$'
plotting_labels_no_units['f_lamp'] = r'$f_{\mathrm{Lamp}}$'
plotting_labels_no_units['theta_inc'] = r'$\theta_{\mathrm{inc}}$'
plotting_labels_no_units['height'] = r'$(H-R_{\mathrm{in}})/R_g$'
plotting_labels_no_units['redshift'] = r'$z$'
plotting_labels_no_units['beta'] = r'$\beta$'
plotting_labels_no_units['log_mass'] = r"$\log_{10}\left(M/M_\odot\right)$"
plotting_labels_no_units['log_nu_b'] = r'$\log_{10}(\nu_b)$'
plotting_labels_no_units['alpha_L'] = r'$\alpha_L$'
plotting_labels_no_units['alpha_H_minus_L'] = r'$\alpha_H - \alpha_L$'
plotting_labels_no_units['standard_deviation'] = r'$\sigma$'

#Define the max and min ranges of the parameter space
min_max_dict = dict()
min_max_dict['log_mass'] = (7.0,10.0) 
min_max_dict['spin'] = (-1.0,1.0)
min_max_dict['log_edd'] = (-2.0,0.0)
min_max_dict['f_lamp'] = (0.002,0.007)
min_max_dict['theta_inc'] = (0.0,70.0)
min_max_dict['height'] = (0.0,40.0)
min_max_dict['redshift'] = (0.1,5.0) 
min_max_dict['beta'] = (0.5,1.0) 
min_max_dict['log_nu_b'] = (-3.5,0.0)
min_max_dict['alpha_L'] = (0.25,1.5)
min_max_dict['alpha_H_minus_L'] = (0.75, 2.75)
min_max_dict['standard_deviation'] = (0.0,0.5)

min_max_array = np.array([min_max_dict[key] for key in parameters_keys])

days_per_year = 365
hours_per_day = 24
# This is the size of the light curve. The LSST cadences last for 10 years but we can make predictions for times before or after.
num_years = 10.5

# LSST filters
bandpasses=list('ugrizy')
#effective frequency of LSST bands
lambda_effective_Angstrom = np.array([3671, 4827, 6223, 7546, 8691, 9712]) #in Angstrom
lambda_effective = 1e-10*lambda_effective_Angstrom #in meter
freq_effective = c/lambda_effective #in Hz

lambda_min_LSST = 3000 #in Angstrom
lambda_max_LSST = 11000 #in Angstrom

# Should be set to True. Actually models the spectrum and integrates across the LSST response filters. 
model_spectrum = True 

# Not used if model_spectrum = True. 
mag_mean = [21.52,21.06,20.80,20.64,20.49,20.41]
mag_std = [1.29,1.08,1.01,0.98,0.95,0.94]

min_magnitude = 13.0
max_magnitude = 27.0

# Define the cosmology. We assume a flat universe with H0 = 70 km/s/Mpc and Omega_m = 0.3.
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)

# For our spectrum model
# Taken from https://github.com/MJTemple/qsogen
f1 = 'model/qsogen_files/qsosed_emlines_20210625.dat'
# wav, median_emlines, continuum, peaky_line, windy_lines, narrow_lines
emline_template = np.genfromtxt(f1, unpack=True)

# S0 galaxy template from SWIRE
# https://ui.adsabs.harvard.edu/abs/2008MNRAS.386..697R/
f2 = 'model/qsogen_files/S0_template_norm.sed'
galaxy_template = np.genfromtxt(f2, unpack=True)


# Extinction curve, format: [lambda, E(lambda-V)/E(B-V)]
# Recall flux_reddened(lambda) = flux(lambda)*10^(-A(lambda)/2.5)
# where A(lambda) = E(B-V)*[E(lambda-V)/E(B-V) + R] 
# so taking R=3.1, A(lambda) = E(B-V)*[Col#2 + 3.1]
f3 = 'model/qsogen_files/pl_ext_comp_03.sph'
reddening_curve = np.genfromtxt(f3, unpack=True)


mag_mean = float(np.array(mag_mean).mean())
mag_std = float(np.array(mag_std).mean())

# CHANGE THIS TO THE PATH TO THE DATA!
local = True if platform != 'linux' else False
if local:
    # If local, just use a few examples for testing.
    file_name_for_cadence = "../../cadence"
    num_training_LC = 50
    validation_rate = 0.1
else:
    # Cadence files generated from: rubin_sim, https://github.com/lsst/rubin_sim
    file_name_for_cadence = "../rubin/cadence_100000" 
    num_training_LC = 100_000 
    validation_rate = 0.1

assert num_training_LC > 0, "Must have at least one training light curve"
#Fraction of the total data set used as validation set.

num_validation_LC = max(int(validation_rate*num_training_LC),1) # We must have at least one validation light curve

# This 
seed_list_train = np.random.randint(0, 99999999, num_training_LC)
seed_list_val = np.random.randint(0, 99999999, num_validation_LC)

#seed_list_train = seed_list_train[:100]
#seed_list_val = seed_list_val[:100]

# Time spacing of data set in days. Ideally this would be smaller but would take up too much memory.
cadence = 1.0

n_params = len(parameters_keys)
output_dim = 2*len(bandpasses) #num bands and uncertainty
num_bands = len(bandpasses)    #num bands

# This is an array of the days included in the light curves
days = cadence*np.arange(int(num_years*days_per_year/cadence)+1) 

# Get the LSST cadence files
cadence_files = glob(f'{file_name_for_cadence}/*.dat')
num_cadences = len(cadence_files)//len(bandpasses)
del cadence_files

def generate_variability(N, dt, mean_mag, standard_deviation, log_nu_b, alpha_L, alpha_H_minus_L, redshift, extra_time_factor = 8, plot=False):
    """
    Function to generate the driving variability from a bended broken power-law PSD.
    
    N: int, number of time steps in the driving signal.
    dt: float, time step in days.
    mean_mag: float, mean magnitude of the driving signal.
    standard_deviation: float, standard deviation of the driving signal.
    log_nu_b: float, log10 of the break frequency in days.
    alpha_L: float, power law index of the low frequency part of the power spectral density.
    alpha_H_minus_L: float, difference between the power law index of the high frequency part and the low frequency part of the power spectral density.
    redshift: float, redshift of the driving signal.
    extra_time_factor: float, extra time to add to the driving signal to avoid periodic boundary conditions.
    plot: bool, if True, plot the driving signal. For debugging purposes.

    return: numpy array, driving signal.
    """
    # get alpha_H from alpha_H_minus_L and alpha_L
    alpha_H = alpha_H_minus_L + alpha_L
    
    # get nu_b from log10(nu_b), this is the rest frame frequency
    nu_b_rest = 10.0**log_nu_b 

    # get nu_b_rest from nu_b and redshift
    nu_b = nu_b_rest / (1.0+redshift) # observed frame break frequency

    # Apply the extra time to avoid the periodicity of generating a signal
    duration = extra_time_factor*N*dt 
    
    # Frequency range from 1/duration to the Nyquist frequency
    frequencies = np.linspace(1.0/duration, 1.0/(2.0*dt), int(duration//2/dt)+1)
    psd = (frequencies**-alpha_L)*(1.0+(frequencies/nu_b)**(alpha_H-alpha_L))**-1

    if plot:
        plt.figure(figsize=(6, 6))
        plt.loglog(frequencies, psd, label='PSD')
        plt.loglog(frequencies[frequencies<=nu_b],psd[0]*(frequencies[frequencies<=nu_b]/frequencies[0])**-alpha_L,
                   linestyle='--', label = f'$P(\\nu) \\propto \\nu^{{-{alpha_L}}}$')
        plt.loglog(frequencies[frequencies>nu_b],psd[-1]*(frequencies[frequencies>nu_b]/frequencies[-1])**-alpha_H,
                   linestyle='--', label = f'$P(\\nu) \\propto \\nu^{{-{alpha_H}}}$')
        plt.axvline(nu_b, color='black', linestyle='--', label=r'$\nu = \nu_b$') # This is the observed frame break frequency
        plt.xlim(frequencies[0],frequencies[-1])
        plt.xlabel('freq [1/days]')
        plt.ylabel(r'$P(\nu)$')
        plt.legend()
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        plt.savefig("Example_PSD_labeled.pdf", bbox_inches='tight')
        plt.show()

    ## Now generate the light curve from the PSD ##
        
    # Generate random phase shifts uniformly distributed in range [0, 2pi]
    random_phases = 2.0 * np.pi * np.random.random(size=frequencies.size)

    # Generate complex-valued function of frequency
    fourier_transform = np.sqrt(psd) * np.exp(1j*random_phases)

    # Make sure the function of frequency is Hermitian
    fourier_transform = np.concatenate((fourier_transform, fourier_transform[-2:0:-1].conjugate()))

    # Generate time series using inverse Fourier transform, drop the imaginary part (should be ~0)
    timeseries = ifft(fourier_transform).real

    # Normalize flux to have mean zero and variance one
    timeseries = timeseries - timeseries.mean()
    timeseries = timeseries / timeseries.std()

    # Now set to the desired mean magnitude and stdev
    timeseries = timeseries * standard_deviation
    timeseries = timeseries + mean_mag

    # Time array
    time = np.linspace(0, duration, int(duration/dt))

    # get rid of the extra time to not include the periodic boundary condition and to use unbiased mean_mag and standard_deviation
    timeseries = timeseries[N: 2*N] # Take out an N time step section
    time = time[:N]
    
    if plot:
        # Plot time series
        plt.figure(figsize=(12, 4))
        plt.plot(time, timeseries)
        plt.ylim(mean_mag-4*standard_deviation,mean_mag+4*standard_deviation)
        plt.xlim(time[0],time[-1])
        plt.gca().invert_yaxis()
        plt.xlabel('time [days]')
        plt.ylabel('magnitude')
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        plt.savefig("Example_xray_variability.pdf", bbox_inches='tight')
        plt.show()
     
    if plot:
        # The PSD cannot be recovered at very low frequency due to red noise leak from not having long enough light curve and alliasing
        frac_sep = 0.5
        frequencies_measured, psd_measured = welch(timeseries, fs=1/dt, nperseg=frac_sep*duration/extra_time_factor)
        psd_measured = psd_measured/psd_measured[int(0.5*len(psd_measured))] # normalize by point 90% through measured PSD
        # Theoretical PSD
        psd = psd/psd[int(0.5*len(psd))] # normalize by point 90% through the PSD. This is just for plotting purposes.


        plt.figure(figsize=(6, 6))
        plt.loglog(frequencies, psd, label='theoretical PSD')
        plt.loglog(frequencies_measured, psd_measured, label='measured PSD')
        plt.loglog(frequencies[frequencies<=nu_b],psd[0]*(frequencies[frequencies<=nu_b]/frequencies[0])**-alpha_L,
                   linestyle='--', label = f'$P(\\nu) \\propto \\nu^{{-{alpha_L}}}$')
        plt.loglog(frequencies[frequencies>nu_b],psd[-1]*(frequencies[frequencies>nu_b]/frequencies[-1])**-alpha_H,
                   linestyle='--', label = f'$P(\\nu) \\propto \\nu^{{-{alpha_H}}}$')
        plt.axvline(nu_b, color='black', linestyle='--', label=r'$\nu = \nu_b$')
        plt.title("Example Recovered PSD")
        plt.xlim(1.0/(duration/extra_time_factor),frequencies[-1])
        plt.xlabel('frequency [1/day]')
        plt.ylabel('PSD')
        plt.legend()
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        plt.savefig("Example_PSD_recovery.pdf", bbox_inches='tight')
        plt.show()

    return timeseries

def generate_driving_variability_torch(N, dt, mean_mag, standard_deviation, log_nu_b, alpha_L, alpha_H_minus_L, phases, redshift, device, dtype=torch.float32,  extra_time_factor = 8, plot=False):
    """
    Not used in this work but included for completeness. 
    This function generates a driving signal using PyTorch to work on GPU and with auto-differentiation.
    
    N: int, number of time steps in the driving signal.
    dt: float, time step in days.
    mean_mag: float, mean magnitude of the driving signal.
    standard_deviation: float, standard deviation of the driving signal.
    log_nu_b: float, log10 of the break frequency in days.
    alpha_L: float, power law index of the low frequency part of the power spectral density.
    alpha_H_minus_L: float, difference between the power law index of the high frequency part and the low frequency part of the power spectral density.
    phases: Either None, or torch tensor, random phases uniformly distributed in range [0, 1]. If none then the phases are generated randomly.
    redshift: float, redshift of the driving signal.
    device: torch device, device to use for the computation.
    dtype: torch dtype, data type to use for the computation.
    extra_time_factor: float, extra time to add to the driving signal to avoid periodic boundary conditions.
    plot: bool, if True, plot the driving signal. For debugging purposes.

    return: torch tensor of shape (N, 1), driving signal.
    """
    # get alpha_H from alpha_H_minus_L and alpha_L
    alpha_H = alpha_H_minus_L + alpha_L
    # get nu_b from log10(nu_b)
    nu_b = 10.0**log_nu_b

    # apply the redshift to the time domain
    dt = dt*(1.0+redshift) # We could have applied the redshift to the frequency domain instead 

    # Apply the extra time to avoid the periodicity of generating a signal
    duration = extra_time_factor*N*dt 
    
    # Frequency range from 1/duration to the Nyquist frequency
    frequencies = torch.linspace(1.0/duration, 1.0/(2.0*dt), int(duration//2/dt)+1, dtype=dtype, device=device)

    psd = (frequencies**-alpha_L)*(1.0+(frequencies/nu_b)**(alpha_H-alpha_L))**-1
    
    if plot:
        plt.figure(figsize=(6, 6))
        plt.loglog(frequencies, psd, label='PSD')
        plt.loglog(frequencies[frequencies<=nu_b],psd[0]*(frequencies[frequencies<=nu_b]/frequencies[0])**-alpha_L,
                   linestyle='--', label = f'$P(\\nu) \\propto \\nu^{{-{alpha_L}}}$')
        plt.loglog(frequencies[frequencies>nu_b],psd[-1]*(frequencies[frequencies>nu_b]/frequencies[-1])**-alpha_H,
                   linestyle='--', label = f'$P(\\nu) \\propto \\nu^{{-{alpha_H}}}$')
        plt.axvline(nu_b, color='black', linestyle='--', label=r'$\nu = \nu_b$')
        plt.xlim(frequencies[0],frequencies[-1])
        plt.xlabel('freq [1/days]')
        plt.ylabel(r'$P(\nu)$')
        plt.legend()
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        plt.savefig("Example_PSD_labeled.pdf", bbox_inches='tight')
        plt.show()

    # Now generate the light curve from the PSD
        
    # Generate random phase shifts uniformly distributed in range [0, 2pi]
    if phases is None:
        random_phases = 2.0 * np.pi * torch.rand(frequencies.shape, dtype=dtype, device=device)
    else:
        random_phases = 2.0 * np.pi * phases.type(dtype).to(device)
    #random_phases = 2.0 * np.pi * phases
    # Generate complex-valued function of frequency
    fourier_transform = torch.sqrt(psd) * torch.exp(1j*random_phases)
    
    # Make sure the function of frequency is Hermitian
    #fourier_transform = torch.cat((fourier_transform, fourier_transform[-2:0:-1].conj()))
    fourier_transform = torch.cat((fourier_transform, torch.flip(fourier_transform, dims=(0,))[1:-1]))
                                   
    # Generate time series using inverse Fourier transform, drop the imaginary part (should be ~0)
    timeseries = torch.fft.ifft(fourier_transform).real

    # Normalize flux to have mean zero and variance one
    timeseries = timeseries - timeseries.mean()
    timeseries = timeseries / timeseries.std()

    timeseries = timeseries.unsqueeze(1) * standard_deviation.unsqueeze(0)
    timeseries = timeseries + mean_mag.unsqueeze(0)


    # get rid of the extra time to not include the periodic boundary condition and to use unbiased mean_mag and standard_deviation
    #timeseries = timeseries[N: 2*N]
    timeseries = timeseries[N:2*N]

    if plot:
        # Time array
        time = torch.linspace(0, duration, int(duration/dt)).type(dtype).to(device)
        time = time[:N]

        # Plot time series
        plt.figure(figsize=(12, 4))
        plt.plot(time.detach().cpu().numpy(), timeseries.detach().cpu().numpy())
        plt.title("Normalized X-ray Flux")
        plt.ylim(mean_mag-4*standard_deviation,mean_mag+4*standard_deviation)
        plt.xlim(time[0],time[-1])
        plt.xlabel('time [days]')
        plt.ylabel('normalized flux')
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        plt.savefig("Example_xray_variability.pdf", bbox_inches='tight')
        plt.show()

    return timeseries

def make_light_curve(transfer_function, log_nu_b, alpha_L , alpha_H_minus_L, standard_deviation, band_mean_mag, band_mean_mag_without_host_galaxy, redshift, kernel_num_days, kernel_resolution, reference_band, plot=False, custom_driving_signal=None):
    """
    This function generated a light curve by convolving it with a transfer function. 

    transfer_function: numpy array of shape (length_t, num_bands), transfer function used as a kernel to convolve with DRW.
    log_nu_b: float, log10 of the break frequency in days.
    alpha_L: float, power law index of the low frequency part of the power spectral density.
    alpha_H_minus_L: float, difference between the power law index of the high frequency part and the low frequency part of the power spectral density.
    standard_deviation: float, sandard deviation of the driving signal.
    band_mean_mag: numpy array of shape (num_bands), mean magnitude of each band.
    band_mean_mag_without_host_galaxy: numpy array of shape (num_bands), mean magnitude of each band excluding the contribution of the driving signal.
    redshift: float, redshift the spectrum into the observers frame to integrate across the LSST filters
    kernel_num_days: int, number of dats in the transfer function kernels
    kernel_resolution: float, resolution of the transfer function bins
    reference_band: int, an arbitrary reference band to determine the normalization of the driving signal
    plot: bool, if True, plot the light curve and DRW. For debugging purposes.
    custom_driving_signal: function, a custom driving signal function to use instead of the default bended broken power-law

    return: numpy array of shape (length_t, num_bands), light curve.
    """

    # Extra time points to make sure the convolution is done correctly at the edges

    extra_time = kernel_num_days

    num_days = num_years*days_per_year

    t = np.linspace(0,num_days,int(num_days/cadence)+1)

    dt = kernel_resolution # time step in days
    duration = num_days+extra_time
    
    t_high_res = np.linspace(0,num_days,int(num_days/dt)+1)
    N = int(extra_time/dt)+int(num_days/dt) # driving N time steps

    ### Figure out what to do here ###
    
    # Generate the driving signal with zero mean
    driving_mean_mag = 0.0
    # Use redshift 0.0 so nu_b is in the observed frame
    if custom_driving_signal is None:
        driving = generate_variability(N, dt, driving_mean_mag, standard_deviation, log_nu_b, alpha_L, alpha_H_minus_L, redshift=0.0, extra_time_factor = 8, plot=False)
    else:
        # use some custom function instead of the broken power law
        driving = custom_driving_signal(N, dt, driving_mean_mag, standard_deviation, log_nu_b, alpha_L, alpha_H_minus_L, extra_time_factor = 8) # We just use this to test other kinds of driving signals than the broken power law after training.
    
    flux_without_host_galaxy = mag_to_flux(np.expand_dims(band_mean_mag_without_host_galaxy, axis=0) + np.expand_dims(driving, axis=1))
    flux_from_host_galaxy = mag_to_flux(np.expand_dims(band_mean_mag, axis=0))-mag_to_flux(np.expand_dims(band_mean_mag_without_host_galaxy, axis=0))

    assert np.min(flux_from_host_galaxy) >= 0.0, "Flux from host galaxy must be positive, the host galaxy must never take flux away from the quasar."

    light_curve_high_res = flux_without_host_galaxy + flux_from_host_galaxy
    
    driving_save = flux_to_mag(light_curve_high_res[:, reference_band])

    # light curve at resolution of cadence.
    light_curve = np.zeros((len(t),len(bandpasses)))

    for i in range(len(bandpasses)):
        conv = np.convolve(light_curve_high_res[:, i], transfer_function[:,i], mode='valid')
        if cadence != dt:
            light_curve[:,i] = np.interp(t, t_high_res, conv)
        else:
            light_curve[:,i] = conv

    
    del conv, light_curve_high_res
    # convert light curve from F_nu in units of erg/s/cm^2/Hz to magnitude
    light_curve = flux_to_mag(light_curve) 

    # plot the DRW and then the convolved light curve for each band
    if plot:

        # plot the light curve
        driving_norm = driving_save[transfer_function.shape[0]-1:]
        light_curve_norm1 = light_curve[:,0]
        light_curve_norm2 = light_curve[:,1]
        light_curve_norm3 = light_curve[:,2]
        driving_norm = driving_norm - driving_norm.mean()
        driving_norm = driving_norm/driving_norm.std()
        driving_norm = driving_norm*0.2+19.7
        light_curve_norm1 = light_curve_norm1 - light_curve_norm1.mean()
        light_curve_norm1 = light_curve_norm1/light_curve_norm1.std()
        light_curve_norm1 = light_curve_norm1*0.2+19.9
        light_curve_norm2 = light_curve_norm2 - light_curve_norm2.mean()
        light_curve_norm2 = light_curve_norm2/light_curve_norm2.std()
        light_curve_norm2 = light_curve_norm2*0.2+20.1
        light_curve_norm3 = light_curve_norm3 - light_curve_norm3.mean()
        light_curve_norm3 = light_curve_norm3/light_curve_norm3.std()
        light_curve_norm3 = light_curve_norm3*0.2+20.3

        # plot the time delays
        zoom_time = 50 # days
        plt.figure(figsize=(6, 4))
        plt.plot(t, driving_norm, label='driving', color='black')
        plt.plot(t,light_curve_norm1,label=bandpasses[0], color="blue")
        plt.plot(t,light_curve_norm2,label=bandpasses[1], color="green")
        plt.plot(t,light_curve_norm3,label=bandpasses[2], color="#F2D91A")
        # get time delay
        time_delay1 = np.sum(transfer_function[:,0]*np.arange(transfer_function.shape[0]))*dt
        time_delay2 = np.sum(transfer_function[:,1]*np.arange(transfer_function.shape[0]))*dt
        time_delay3 = np.sum(transfer_function[:,2]*np.arange(transfer_function.shape[0]))*dt
        # plot the time delay arrow
        start_time = 15 #np.argmax(np.abs(light_curve_norm1[:zoom_time]))
        head_width = 0.05
        head_length = 1.0
        start_height = min(driving_norm[:int(zoom_time/dt)]-0.3)
        plt.arrow(start_time, start_height, time_delay1, 0, head_width=head_width, head_length=head_length, fc='blue', ec='blue')
        # arrow text
        plt.text(start_time-9, start_height+0.025, f'{time_delay1:.1f} days', color='blue', fontsize=12)
        plt.arrow(start_time, start_height+0.1, time_delay2, 0, head_width=head_width, head_length=head_length, fc='green', ec='green')
        plt.text(start_time-9, start_height+0.1+0.025, f'{time_delay2:.1f} days', color='green', fontsize=12)
        plt.arrow(start_time, start_height+0.2, time_delay3, 0, head_width=head_width, head_length=head_length, fc='#F2D91A', ec='#F2D91A')
        plt.text(start_time-9, start_height+0.2+0.025, f'{time_delay3:.1f} days', color='#F2D91A', fontsize=12)
        plt.ylim(start_height-0.1, np.max(light_curve_norm3[:int(zoom_time/dt)]+0.1))
        plt.xlim(0, zoom_time)
        plt.gca().invert_yaxis()
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        #plt.xlim(t.min(),t.max())
        plt.xlabel('time [days]')
        plt.ylabel('magnitude')
        plt.tight_layout()
        plt.savefig("Example_time_delay.pdf", bbox_inches='tight')
        plt.show()


    if plot:
        for i in range(len(bandpasses)):
            plt.plot(transfer_function[:,i],label=bandpasses[i])
        min_val = np.argmax((transfer_function[0,:]>0.0001)+np.array(range(transfer_function.shape[-1])) * 1e-5)
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        #plt.xlim(0,min_val)
        plt.legend()
        plt.xlabel(f'time')
        plt.ylabel('PD')
        plt.savefig('transfer_function.pdf',bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(12, 3))
        for i in range(len(bandpasses)):
            plt.plot(t,light_curve[:,i],label=bandpasses[i])
        #DRW_mag = flux_to_mag(DRW/freq_effective.mean())
        #DRW_mag = DRW_mag - DRW_mag.mean()+light_curve.max()+0.05
        #plt.plot(t_hourly_extra,DRW_mag,color='black',label='DRW')
        plt.gca().invert_yaxis()
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        plt.xlim(t.min(),t.max())
        plt.legend(loc='upper left')
        plt.xlabel('time [days]')
        plt.ylabel('magnitude')
        plt.savefig('DRW_light_curve.pdf',bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    # linearly interpolate the driving_save so that it is at the same resolution as the light curve
    if cadence != dt:
        driving_save = np.interp(np.linspace(0,1,int(duration/cadence)), np.linspace(0,1,driving_save.shape[0]), driving_save)

    return light_curve, transfer_function, driving_save

class Quasar_sed():
    """
    @author: Matthew Temple
    https://github.com/MJTemple/qsogen
    # Modified by: Joshua Fagin for use in my project but original code by Matthew Temple. All credit to Matthew Temple.
    # See his paper: https://arxiv.org/abs/2109.04472

    Construct an instance of the quasar SED model.

    Attributes
    ----------
    flux : ndarray
        Flux per unit wavelength from total SED, i.e. quasar plus host galaxy.
    host_galaxy_flux : ndarray
        Flux p.u.w. from host galaxy component of the model SED.
    wavlen : ndarray
        Wavelength array in the rest frame.
    wavred : ndarray
        Wavelength array in the observed frame.

    Examples
    --------
    Create and plot quasar models using default params at redshifts z=2 and z=4
    >>> Quasar2 = Quasar_sed(z=2)
    >>> Quasar4 = Quasar_sed(z=4)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(Quasar2.wavred, Quasar2.flux, label='$z=2$ quasar model')
    >>> plt.plot(Quasar4.wavred, Quasar4.flux, label='$z=4$ quasar model')

    """
    def __init__(self,
                 params,
                 wavlen,
                 ):
        """Initialises an instance of the Quasar SED model.

        Parameters
        ----------
        z : float, optional
            Redshift. If `z` is less than 0.005 then 0.005 is used instead.
        LogL3000 : float, optional
            Monochromatic luminosity at 3000A of (unreddened) quasar model,
            used to scale model flux such that synthetic magnitudes can be
            computed.
        wavlen : ndarray, optional
            Rest-frame wavelength array. Default is log-spaced array covering
            ~890 to 30000 Angstroms. `wavlen` must be monotonically increasing,
            and if gflag==True, `wavlen` must cover 4000-5000A to allow the
            host galaxy component to be properly normalised.
        ebv : float, optional
            Extinction E(B-V) applied to quasar model. Not applied to galaxy
            component. Default is zero.
        zlum_lumval : array, optional
            Redshift-luminosity relation used to control galaxy and emission-
            line contributions. `zlum_lumval[0]` is an array of redshifts, and
            `zlum_lumval[1]` is an array of the corresponding absolute i-band
            magnitudes M_i. Default is the median M_i from SDSS DR16Q in the
            apparent magnitude range 18.6<i<19.1.
        M_i :float, optional
            Absolute i-band magnitude (at z=2), as reported in SDSS DR16Q, used
            to control scaling of emission-line and host-galaxy contributions.
            Default is to use the relevant luminosity from `zlum_lumval`, which
            gives a smooth scaling with redshift `z`.
        params : dict, optional
            Dictionary of additional parameters, including emission-line and
            host-galaxy template SEDs, reddening curve. Default is to read in
            from config.py file.

        Other Parameters
        ----------------
        tbb : float, optional
            Temperature of hot dust blackbody in Kelvin.
        bbnorm : float, optional
            Normalisation, relative to power-law continuum at 2 micron, of the
            hot dust blackbody.
        scal_emline : float, optional
            Overall scaling of emission line template. Negative values preserve
            relative equivalent widths while positive values preserve relative
            line fluxes. Default is -1.
        emline_type : float, optional
            Type of emission line template. Minimum allowed value is -2,
            corresponding to weak, highly blueshifed lines. Maximum allowed is
            +3, corresponding to strong, symmetric lines. Zero correspondes to
            the average emission line template at z=2, and -1 and +1 map to the
            high blueshift and high EW extrema observed at z=2. Default is
            None, which uses `beslope` to scale `emline_type` as a smooth
            function of `M_i`.
        scal_halpha, scal_lya, scal_nlr : float, optional
            Additional scalings for the H-alpha, Ly-alpha, and for the narrow
            optical lines. Default is 1.
        beslope : float, optional
            Baldwin effect slope, which controls the relationship between
            `emline_type` and luminosity `M_i`.
        bcnorm : float, optional
            Balmer continuum normalisation. Default is zero as default emission
            line templates already include the Balmer Continuum.
        lyForest : bool, optional
            Flag to include Lyman absorption from IGM. Default is True.
        lylim : float, optional
            Wavelength of Lyman-limit system, below which all flux is
            suppressed. Default is 912A.
        gflag : bool, optional
            Flag to include host-galaxy emission. Default is True.
        fragal : float, optional
            Fractional contribution of the host galaxy to the rest-frame 4000-
            5000A region of the total SED, for a quasar with M_i = -23.
        gplind : float, optional
            Power-law index dependence of galaxy luminosity on M_i.
        emline_template : array, optional
            Emission line templates. Array must have structure
            [wavelength, average lines, reference continuum,
            high-EW lines, high-blueshift lines, narrow lines]
        reddening_curve : array, optional
            Quasar reddening law.
            Array must have structure [wavelength lambda, E(lambda-V)/E(B-V)]
        galaxy_template : array, optional
            Host-galaxy SED template.
            Array must have structure [lambda, f_lambda].
            Default is an S0 galaxy template from the SWIRE library.

        """
        _params = params.copy()
        z = _params['z']
        self.z = max(float(z), 0.005)
        # avoid crazy flux normalisation at zero redshift

        self.wavlen = wavlen
        if np.any(self.wavlen[:-1] > self.wavlen[1:]):
            raise Exception('wavlen must be monotonic')
        self.flux = np.zeros_like(self.wavlen)
        self.host_galaxy_flux = np.zeros_like(self.wavlen)

        self.ebv =  _params['ebv']
        self.plslp1 = _params['plslp1']
        self.plslp2 = _params['plslp2']
        self.plstep = _params['plstep']
        self.tbb = _params['tbb']
        self.plbrk1 = _params['plbrk1']
        self.plbrk3 = _params['plbrk3']
        self.bbnorm = _params['bbnorm']
        self.scal_emline = _params['scal_emline']
        self.emline_type = _params['emline_type']
        self.scal_halpha = _params['scal_halpha']
        self.scal_lya = _params['scal_lya']
        self.scal_nlr = _params['scal_nlr']

        self.emline_template = _params['emline_template']
        self.reddening_curve = _params['reddening_curve']
        self.galaxy_template = _params['galaxy_template']
        self.continuum = _params['continuum'] # We don't want to modify the original continuum outside of this class

        self.beslope = _params['beslope']
        self.benorm = _params['benorm']
        self.bcnorm = _params['bcnorm']
        self.fragal = _params['fragal']
        self.gplind = _params['gplind']

        self.zlum = _params['zlum_lumval'][0]
        self.lumval = _params['zlum_lumval'][1]

        if _params['M_i'] is not None:
            self.M_i = _params['M_i']
        else:
            self.M_i = np.interp(self.z, self.zlum, self.lumval)

        #######################################################
        # READY, SET, GO!
        #######################################################

        '''
        self.set_continuum() # We don't want to use this since we have our own continuum
        self.add_blackbody() # We won't get any black body emission from the hot dust since it only emits after 10000 Angstroms

        # We don't need this because it is included in the emission line templates
        if self.bcnorm:
            self.add_balmer_continuum()

        # We don't need this because our continuum is already in f_lambda in physical units
        if LogL3000 is not None:
            self.f3000 = (10**(LogL3000 - four_pi_dL_sq(self.z))
                          / (3000*(1 + self.z)))
            self.convert_fnu_flambda(flxnrm=self.f3000, wavnrm=3000)
        else:
            self.convert_fnu_flambda()
        '''

        # Set the flux to our continuum
        self.flux = self.continuum

        self.add_emission_lines()
        if _params['gflag']:
            self.host_galaxy()
        # creates self.host_galaxy_flux object
        # need to create this before reddening qso to get correct normalisation

        # redden spectrum if E(B-V) != 0
        if self.ebv:
            self.redden_spectrum()

        # add in host galaxy flux
        if _params['gflag']:
            self.flux += self.host_galaxy_flux

        # simulate the effect of a Lyman limit system at rest wavelength Lylim
        # by setting flux equal to zero at wavelengths < Lylim angstroms
        if _params['lyForest']:
            lylim = self.wav2num(_params['lylim'])
            self.flux[:lylim] = 0.0
            self.host_galaxy_flux[:lylim] = 0.0
            # Then add in Ly forest absorption at z>1.4
            self.lyman_forest()
        
        # redshift spectrum
        #self.wavred = (self.z + 1)*self.wavlen

    def wav2num(self, wav):
        """Convert a wavelength to an index."""
        return np.argmin(np.abs(self.wavlen - wav))

    def wav2flux(self, wav):
        """Convert a wavelength to a flux.

        Different from self.flux[wav2num(wav)], as wav2flux interpolates in an
        attempt to avoid problems when wavlen has gaps. This mitigation only
        works before the emission lines are added to the model, and so wav2flux
        should only be used with a reasonably dense wavelength array.
        """
        return np.interp(wav, self.wavlen, self.flux)

    @staticmethod
    def pl(wavlen, plslp, const):
        """Define power-law in flux density per unit frequency."""
        return const*wavlen**plslp

    @staticmethod
    def bb(tbb, wav):
        """Blackbody shape in flux per unit frequency.

        Parameters
        ----------
        tbb
            Temperature in Kelvin.
        wav : float or ndarray of floats
            Wavelength in Angstroms.

        Returns
        -------
        Flux : float or ndarray of floats
            (Non-normalised) Blackbody flux density per unit frequency.

        Notes
        -----
        h*c/k_b = 1.43877735e8 KelvinAngstrom
        """
        return (wav**(-3))/(np.exp(1.43877735e8 / (tbb*wav)) - 1.0)

    def set_continuum(self, flxnrm=1.0, wavnrm=5500):
        """Set multi-powerlaw continuum in flux density per unit frequency."""
        # Flip signs of powerlaw slopes to enable calculation to be performed
        # as a function of wavelength rather than frequency
        sl1 = -self.plslp1
        sl2 = -self.plslp2
        wavbrk1 = self.plbrk1

        # Define normalisation constant to ensure continuity at wavbrk
        const2 = flxnrm/(wavnrm**sl2)
        const1 = const2*(wavbrk1**sl2)/(wavbrk1**sl1)

        # Define basic continuum using the specified normalisation fnorm at
        # wavnrm and the two slopes - sl1 (<wavbrk) sl2 (>wavbrk)
        fluxtemp = np.where(self.wavlen < wavbrk1,
                            self.pl(self.wavlen, sl1, const1),
                            self.pl(self.wavlen, sl2, const2))

        # Also add steeper power-law component for sub-Lyman-alpha wavelengths
        sl3 = sl1 - self.plstep
        wavbrk3 = self.plbrk3
        # Define normalisation constant to ensure continuity
        const3 = const1*(wavbrk3**sl1)/(wavbrk3**sl3)

        self.flux = np.where(self.wavlen < wavbrk3,
                             self.pl(self.wavlen, sl3, const3),
                             fluxtemp)

    def add_blackbody(self, wnorm=20000.):
        """Add basic blackbody spectrum to the flux distribution."""
        bbnorm = self.bbnorm  # blackbody normalisation at wavelength wnorm
        tbb = self.tbb

        if bbnorm > 0:

            bbval = self.bb(tbb, wnorm)
            cmult = bbnorm / bbval
            bb_flux = cmult*self.bb(tbb, self.wavlen)
            self.flux += bb_flux

    def convert_fnu_flambda(self, flxnrm=1.0, wavnrm=5100):
        """Convert f_nu to f_lamda, using c/lambda^2 conversion.
        Normalise such that f_lambda(wavnrm) is equal to flxnrm.
        """
        self.flux = self.flux*self.wavlen**(-2)
        self.flux = self.flux*flxnrm/self.wav2flux(wavnrm)

    def add_emission_lines(self, wavnrm=5500, wmin=6000, wmax=7000):
        """Add emission lines to the model SED.

        Emission-lines are included via 4 emission-line templates, which are
        packaged with a reference continuum. One of these templates gives the
        average line emission for a M_i=-27 SDSS DR16 quasar at z~2. The narrow
        optical lines have been isolated in a separate template to allow them
        to be re-scaled if necesssary. Two templates represent the observed
        extrema of the high-ionisation UV lines, with self.emline_type
        controlling the balance between strong, peaky, systemic emission and
        weak, highly skewed emission. Default is to let this vary as a function
        of redshift using self.beslope, which represents the Baldwin effect.
        The template scaling is specified by self.scal_emline, with positive
        values producing a scaling by intensity, whereas negative values give a
        scaling that preserves the equivalent-width of the lines relative
        to the reference continuum template. The facility to scale the H-alpha
        line by a multiple of the overall emission-line scaling is included
        through the parameter scal_halpha, and the ability to rescale the
        narrow [OIII], Hbeta, etc emission is included through scal_nlr.
        """
        scalin = self.scal_emline
        scahal = self.scal_halpha
        scalya = self.scal_lya
        scanlr = self.scal_nlr
        beslp = self.beslope
        benrm = self.benorm

        if self.emline_type is None:
            if beslp:
                vallum = self.M_i
                self.emline_type = (vallum - benrm)*beslp
            else:
                self.emline_type = 0.  # default median emlines

        varlin = self.emline_type

        linwav, medval, conval, pkyval, wdyval, nlr = self.emline_template

        if varlin == 0.:
            # average emission line template for z~2 SDSS DR16Q-like things
            linval = medval + (scanlr-1.)*nlr
        elif varlin > 0:
            # high EW emission line template
            varlin = min(varlin, 3.)
            linval = varlin*pkyval + (1-varlin)*medval + (scanlr-1.)*nlr
        else:
            # highly blueshifted emission lines
            varlin = min(abs(varlin), 2.)
            linval = varlin*wdyval + (1-varlin)*medval + (scanlr-1.)*nlr

        # remove negative dips from extreme extrapolation (i.e. abs(varlin)>>1)
        linval[(linwav > 4930) & (linwav < 5030) & (linval < 0.)] = 0.
        linval[(linwav > 1150) & (linwav < 1200) & (linval < 0.)] = 0.

        linval = np.interp(self.wavlen, linwav, linval)
        conval = np.interp(self.wavlen, linwav, conval)

        imin = self.wav2num(wmin)
        imax = self.wav2num(wmax)
        _scatmp = abs(scalin)*np.ones(len(self.wavlen))
        _scatmp[imin:imax] = _scatmp[imin:imax]*abs(scahal)
        _scatmp[:self.wav2num(1350)] = _scatmp[:self.wav2num(1350)]*abs(scalya)

        # Intensity scaling
        if scalin >= 0:
            # Normalise such that continuum flux at wavnrm equal to that
            # of the reference continuum at wavnrm
            self.flux += (_scatmp * linval *
                          self.flux[self.wav2num(wavnrm)] /
                          conval[self.wav2num(wavnrm)])
            # Ensure that -ve portion of emission line spectrum hasn't
            # resulted in spectrum with -ve fluxes
            self.flux[self.flux < 0.0] = 0.0

        # EW scaling
        else:
            self.flux += _scatmp * linval * self.flux / conval
            # Ensure that -ve portion of emission line spectrum hasn't
            # resulted in spectrum with -ve fluxes
            self.flux[self.flux < 0.0] = 0.0

    def host_galaxy(self, gwnmin=4000.0, gwnmax=5000.0):
        """Correctly normalise the host galaxy contribution."""

        if min(self.wavlen) > gwnmin or max(self.wavlen) < gwnmax:
            raise Exception(
                    'wavlen must cover 4000-5000 A for galaxy normalisation'
                    + '\n Redshift is {}'.format(self.z))

        fragal = min(self.fragal, 0.99)
        fragal = max(fragal, 0.0)

        wavgal, flxtmp = self.galaxy_template

        # Interpolate galaxy SED onto master wavlength array
        flxgal = np.interp(self.wavlen, wavgal, flxtmp)
        galcnt = np.sum(flxgal[self.wav2num(gwnmin):self.wav2num(gwnmax)])

        # Determine fraction of galaxy SED to add to unreddened quasar SED
        qsocnt = np.sum(self.flux[self.wav2num(gwnmin):self.wav2num(gwnmax)])
        # bring galaxy and quasar flux zero-points equal
        cscale = qsocnt / galcnt

        vallum = self.M_i
        galnrm = -23.   # this is value of M_i for gznorm~0.35
        # galnrm = np.interp(0.2, self.zlum, self.lumval)

        vallum = vallum - galnrm
        vallum = 10.0**(-0.4*vallum)
        tscale = vallum**(self.gplind-1)
        scagal = (fragal/(1-fragal))*tscale

        self.host_galaxy_flux = cscale * scagal * flxgal

    def redden_spectrum(self, R=3.1):
        """Redden quasar component of total SED. R=A_V/E(B-V)."""

        wavtmp, flxtmp = self.reddening_curve
        
        extref = np.interp(self.wavlen, wavtmp, flxtmp)
        exttmp = self.ebv * (extref + R)
        self.flux = self.flux*10.0**(-exttmp/2.5)

    @staticmethod
    def tau_eff(z):
        """Ly alpha optical depth from Becker et al. 2013MNRAS.430.2067B."""
        tau_eff_val = 0.751*((1 + z) / (1 + 3.5))**2.90 - 0.132
        return np.where(tau_eff_val < 0, 0., tau_eff_val)

    def lyman_forest(self):
        """Suppress flux due to incomplete transmission through the IGM.

        Include suppression due to Ly alpha, Ly beta, Ly gamma, using
        parameterisation of Becker+ 2013MNRAS.430.2067B:
        tau_eff(z) = 0.751*((1+z)/(1+3.5))**2.90-0.132
        for z > 1.45, and assuming
        tau_Lyb = 0.16*tau_Lya
        tau_Lyg = 0.056*tau_Lya
        from ratio of oscillator strengths (e.g. Keating+ 2020MNRAS.497..906K).
        """
        if self.tau_eff(self.z) > 0.:

            # Transmission shortward of Lyman-gamma
            scale = np.zeros_like(self.flux)
            wlim = 972.0
            zlook = ((1.0+self.z) * self.wavlen)/wlim - 1.0
            scale[self.wavlen < wlim] = self.tau_eff(zlook[self.wavlen < wlim])
            scale = np.exp(-0.056*scale)
            self.flux = scale * self.flux
            self.host_galaxy_flux = scale * self.host_galaxy_flux

            # Transmission shortward of Lyman-beta
            scale = np.zeros_like(self.flux)
            wlim = 1026.0
            zlook = ((1.0+self.z) * self.wavlen)/wlim - 1.0
            scale[self.wavlen < wlim] = self.tau_eff(zlook[self.wavlen < wlim])
            scale = np.exp(-0.16*scale)
            self.flux = scale * self.flux
            self.host_galaxy_flux = scale * self.host_galaxy_flux

            # Transmission shortward of Lyman-alpha
            scale = np.zeros_like(self.flux)
            wlim = 1216.0
            zlook = ((1.0+self.z) * self.wavlen)/wlim - 1.0
            scale[self.wavlen < wlim] = self.tau_eff(zlook[self.wavlen < wlim])
            scale = np.exp(-scale)
            self.flux = scale * self.flux
            self.host_galaxy_flux = scale * self.host_galaxy_flux


def load_LC(kernel_num_days, kernel_resolution, model_spectrum, lsst_filter, cosmo, emline_template, galaxy_template, reddening_curve, reference_band, min_magnitude, max_magnitude, custom_driving_signal=None, plot=False):
    """
    Function that loads a simulated LSST quasar light curve for the data loader.
    
    kernel_num_days: int, number of dats in the transfer function kernels
    kernel_resolution: float, resolution of the transfer function bins
    lsst_filter: speclite filters, used to integrate across the LSST response functions to get the mean magnitude of each band
    cosmo: astropy cosmology, used to convert redshift to luminosity distance
    emline_template: numpy array, templates for the emission lines flux
    galaxy_template: numpy array, template for the host galaxy flux
    reddening_curve: numpy array, template for the extinction
    reference_band: int, arbitrary reference band for the normalization of the driving signal
    min_magnitude: float, min magnitude of LSST observation
    max_magnitude: float, max magnitude of LSST observation
    custom_driving_signal: function, a custom driving signal function to use instead of the default bended broken power-law
    plot: bool, make plots for debugging purposes
    """
    
    # Accretion disk/BH parameters
    spin = np.random.uniform(min_max_dict["spin"][0],min_max_dict["spin"][1])
    height = np.random.uniform(min_max_dict["height"][0],min_max_dict["height"][1])
    redshift = np.random.uniform(min_max_dict["redshift"][0],min_max_dict["redshift"][1])
    log_edd = np.random.uniform(min_max_dict["log_edd"][0],min_max_dict["log_edd"][1])
    f_lamp = np.random.uniform(min_max_dict["f_lamp"][0],min_max_dict["f_lamp"][1])
    theta_inc = np.random.uniform(min_max_dict["theta_inc"][0],min_max_dict["theta_inc"][1])
    beta = np.random.uniform(min_max_dict["beta"][0],min_max_dict["beta"][1])
    log_mass = np.random.uniform(min_max_dict["log_mass"][0],min_max_dict["log_mass"][1])

    # Can fix the parameters just for plotting
    #spin = 0.0
    #height = 10.0
    #redshift = 2.0
    #log_edd = -1.0
    #f_lamp = 0.005
    #theta_inc = 45.0
    #beta = 0.75
    #log_mass = 8.0
    #plot = True

    # Variability parameters
    log_nu_b = np.random.uniform(min_max_dict["log_nu_b"][0],min_max_dict["log_nu_b"][1])
    alpha_L = np.random.uniform(min_max_dict["alpha_L"][0],min_max_dict["alpha_L"][1])
    alpha_H_minus_L = np.random.uniform(min_max_dict["alpha_H_minus_L"][0],min_max_dict["alpha_H_minus_L"][1])
    standard_deviation = np.random.uniform(min_max_dict["standard_deviation"][0],min_max_dict["standard_deviation"][1])
    
    params = dict()
    params_array = []
    for key in parameters_keys:
        params[key] = eval(key)
        params_array.append(eval(key))
    params_array = np.array(params_array)

    if model_spectrum:
        # We need 7000 Angstrum in the rest frame so 7000*(1+z) in the observed frame in order to model the spectrum. 
        # We never need more than 11000 Angstrom in the observed frame since that is the maximum wavelength of the LSST filters
        minimum_wavelength_needed_for_normalization = max(7000.0*(1.0+redshift),11000) 

        # The continuum will be smooth so 40 points should be enought. Using a higher resolution will cause memory issues.
        spectrum_points = 40 
        lambda_min = 100
        lambda_eval_obs = np.linspace(lambda_min,minimum_wavelength_needed_for_normalization,spectrum_points)
        spectrum = generate_tf_numpy(params_array,
                                    lambda_eval_obs,
                                    kernel_num_days=kernel_num_days,
                                    kernel_resolution=kernel_resolution,
                                    parameters_keys=parameters_keys,
                                    cosmo=cosmo,
                                    GR=True,
                                    just_get_spectrum=True)

        lambda_eval_rest = lambda_eval_obs/(1.0+redshift) # this is the wavelength in the rest frame
        # interpolate the spectrum to a higher resolution
        lambda_high_res_rest = np.linspace(lambda_eval_rest[0],lambda_eval_rest[-1], 1000)
        lambda_high_res_obs = lambda_high_res_rest*(1.0+redshift) # this is the wavelength in the observed frame
        spectrum = np.interp(lambda_high_res_rest, lambda_eval_rest, spectrum)


        # Get the absolute magnitude of the quasar in the i-band
        band_mean_mag = lsst_filter.get_ab_magnitudes(spectrum * u.W/u.m**3, lambda_high_res_obs * u.AA)
        band_mean_mag = np.array(list(band_mean_mag.as_array()[0]))
        m_i = band_mean_mag[3] # i-band
        M_i = m_i - 5*np.log10(cosmo.luminosity_distance(redshift).to(u.pc).value) + 5 # convert apparent magnitude to absolute magnitude
        M_i = M_i + np.random.normal(0.0, 0.1) # add 0.1 mag of noise

        # We can chop off the templates outside the range of lambda_high_res_rest[-1]
        emline_template = emline_template[emline_template[:,0] <= lambda_high_res_rest[-1]]
        galaxy_template = galaxy_template[galaxy_template[:,0] <= lambda_high_res_rest[-1]]
        reddening_curve = reddening_curve[reddening_curve[:,0] <= lambda_high_res_rest[-1]]

        # E(B-V) determines the strength of the extinction. 
        ebv = np.abs(np.random.normal(0.0, 0.075))

        spectrum_params = dict( 
                                z=redshift,
                                ebv=ebv,   #np.random.uniform(0.0,0.2), # = 0.0 # Used! Extinction E(B-V)
                                plslp1=-0.349, # Not used
                                plslp2=0.593, # Not used
                                plstep=-1.0,    # (not fit for)
                                plbrk1=3880., # Not used
                                tbb=1243.6, # Not used
                                plbrk3=1200,   # (not fit for) # Not used right now. Steeper slope before the Lyman alpha break
                                bbnorm=3.961, # Not used
                                scal_emline=-0.9936, # Used! controls if we increase the emission line heigh vs width 
                                emline_type=None, # If None uses relationship with M_i
                                scal_halpha=np.random.normal(1.0, 0.05), # 1. # Used! Controls strength of Halpha
                                scal_lya=np.random.normal(1.0, 0.05), # 1. # Used! Controls strength of Lyman alpha
                                scal_nlr=np.random.normal(1.0, 0.05), # 1. #Used! Controls strength of narrow line region
                                emline_template=emline_template, # Used! The emission line template
                                galaxy_template=galaxy_template, # Used! The galaxy template
                                reddening_curve=reddening_curve, # Used! The reddening curve
                                continuum=spectrum, # Used! The continuum spectrum
                                zlum_lumval=np.array([[0.23, 0.34, 0.6, 1.0, 1.4, 1.8, 2.2,
                                                        2.6, 3.0, 3.3, 3.7, 4.13, 4.5],
                                                        [-21.76, -22.9, -24.1, -25.4, -26.0,
                                                        -26.6, -27.1, -27.6, -27.9, -28.1, -28.4,
                                                        -28.6, -28.9]]), # Used! Redshift-luminosity relation used to control galaxy and emission-line contributions
                                M_i=M_i, # If None uses zlum_lumval instead
                                beslope=0.183, # Used.. Controls the emision line type
                                benorm=-27.,    # (not fit for)
                                bcnorm=False,  # Not used
                                lyForest=True, # Use the lyman alpha forest
                                lylim=912,   # (not fit for)
                                gflag=True, # Use the host galaxy flux
                                fragal=0.244, # Used! Controls the strength of the host galaxy
                                gplind=0.684, # Used! Also host galaxy
                            )
        
        if plot:
            spectrum_cont = np.copy(spectrum)

        # We use the Quasar_sed class from MJTemple/qsogen to generate the quasar spectrum from our continuum spectrum
        SED = Quasar_sed(params=spectrum_params,
                            wavlen=lambda_high_res_rest,
                            )

        spectrum = SED.flux
        spectrum = spectrum.clip(min=1e-30)

        host_galaxy_flux = SED.host_galaxy_flux
        host_galaxy_flux = host_galaxy_flux.clip(min=1e-30)
        
        spectrum_without_host_galaxy = spectrum - host_galaxy_flux
        spectrum_without_host_galaxy = spectrum_without_host_galaxy.clip(min=1e-30)


        del SED

        # get the magnitudes of the spectrum in the LSST bands using astropy units
        band_mean_mag = lsst_filter.get_ab_magnitudes(spectrum * u.W/u.m**3, lambda_high_res_obs * u.AA)
        # convert to numpy array
        band_mean_mag = np.array(list(band_mean_mag.as_array()[0]))

        band_mean_mag_without_host_galaxy = lsst_filter.get_ab_magnitudes(spectrum_without_host_galaxy * u.W/u.m**3, lambda_high_res_obs * u.AA)
        band_mean_mag_without_host_galaxy = np.array(list(band_mean_mag_without_host_galaxy.as_array()[0]))

        if plot:
            p_list = []
        
            fig, ax1 = plt.subplots(figsize=(12, 4))
            spectrum_cgs_units = (spectrum * u.W/u.m**3).to(u.erg/u.s/u.cm**2/u.AA).value
            max_spec_cgs = np.max(spectrum_cgs_units[(3000 < lambda_high_res_obs) & (lambda_high_res_obs < 11000)])
            p1, = ax1.plot(lambda_high_res_obs, (spectrum * u.W/u.m**3).to(u.erg/u.s/u.cm**2/u.AA).value, label=r'$F_\lambda$', color="black", linewidth=2)
            p_list.append(p1)
            # plot the continuum
            p_cont, = ax1.plot(lambda_high_res_obs, (spectrum_cont * u.W/u.m**3).to(u.erg/u.s/u.cm**2/u.AA).value, label=r'cont.', color="black", linewidth=1, linestyle="--")
            p_list.append(p_cont)

            #ax1.plot(lambda_high_res_obs, (host_galaxy_flux * u.W/u.m**3).to(u.erg/u.s/u.cm**2/u.AA).value, label=r'host', color="red", linewidth=1, linestyle="--")
            
            #plt.title(f"Example Spectrum magnitudes {band_mean_mag.round(2)}, redshift: {round(redshift,2)}, mass: {round(log_mass,2)}")
            ax1.set_ylim(0.0,1.1*max_spec_cgs)
            ax1.set_xlim(3000,11000)
            ax1.set_ylabel(r'$F_\lambda$ [$\mathrm{erg/s/cm^2/\AA}$]')
            ax1.set_xlabel('$\lambda$ [$\mathrm{\AA}$]')
            ax1.minorticks_on()
            ax1.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            ax1.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)

            ax2 = ax1.twinx()
            band_color = ['violet', 'green', 'red' , 'orange', 'blue', 'purple']
            for i in range(len(bandpasses)):
                plt.fill_between(lsst_filter[i].wavelength,lsst_filter[i]._response, color=band_color[i], alpha=0.25)
                p2, = plt.plot(lsst_filter[i].wavelength,lsst_filter[i]._response,
                                color=band_color[i], alpha=0.5, label=f"{bandpasses[i]}-band")
                p_list.append(p2)
            ax2.set_ylim(0.0,0.5)
            ax2.set_ylabel('Filter response')
            ax2.minorticks_on()
            ax2.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            ax2.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            ax1.legend(fontsize=12, handles=p_list, loc='upper right')
            plt.tight_layout()
            plt.savefig(f"{save_path}/Example_spectrum.pdf", bbox_inches='tight')
            plt.show()

        del spectrum, spectrum_without_host_galaxy

        # Augment the mean magnitude of the light curve by a random amount so it is not as model dependent. 
        # We add some scatter on each band individiually also add a constant random number
        random_fluctuations = np.random.normal(loc=0.0,scale=0.025,size=len(band_mean_mag))+np.random.normal(loc=0.0,scale=0.025)
        band_mean_mag = band_mean_mag + random_fluctuations
        band_mean_mag_without_host_galaxy = band_mean_mag_without_host_galaxy + random_fluctuations

    else:
        # If model_spectrum is False we just use random magnitudes for each band. Not currently used.
        # Choose the mean magnitude of the light curve
        mean_mag = np.random.normal(mag_mean, mag_std)
        band_mean_mag = np.repeat(mean_mag, len(bandpasses))

    # In case the mean magnitude is outside the range we want we return None
    if band_mean_mag[reference_band] < min_magnitude or band_mean_mag[reference_band] > max_magnitude:
        return None, None, None, None, None

    # Generate the transfer function from the physical parameters
    transfer_function = generate_tf_numpy(params_array, lambda_effective_Angstrom, kernel_num_days=kernel_num_days, kernel_resolution=kernel_resolution, parameters_keys=parameters_keys, cosmo=cosmo, GR=True, just_get_spectrum=False)
    # Generate the light curve from the physical parameters, driving signal parameters, and the transfer function
    LC, transfer_function_save, driving_save = make_light_curve(transfer_function, log_nu_b, alpha_L , alpha_H_minus_L, standard_deviation, band_mean_mag, band_mean_mag_without_host_galaxy, redshift, kernel_num_days, kernel_resolution, reference_band, custom_driving_signal=custom_driving_signal)
    
    # Return the light curve, transfer function, driving signal, parameters, and the mean magnitudes of the bands
    return LC, transfer_function_save, driving_save, params, band_mean_mag

def logit(x,numpy=False, eps=1e-4):
    """
    logit function

    x: float, array, or tensor: must be between 0 and 1
    numpy: bool, if numpy use numpy otherwise we use pytorch
    eps: float, for numerical stability to avoid divergences
    """
    assert(type(numpy)==bool)
    if numpy:
        assert np.max(x) <= 1. and np.min(x) >= 0.
        x = np.clip(x, eps, 1-eps) # clip to avoid infinities
        return np.log(x/(1.-x))
    else:
        assert torch.max(x) <= 1. and torch.min(x) >= 0.
        x = torch.clip(x, eps, 1-eps) # clip to avoid infinities
        return torch.log(x/(1.-x))

def expit(x,numpy=False):
    """
    expit function

    x: float, array, or tensor
    numpy: bool, if numpy use numpy otherwise we use pytorch
    """
    assert(type(numpy)==bool)
    if numpy:
        return 1./(1.+np.exp(-x))
    else:
        return 1./(1.+torch.exp(-x))

def LSST_photometric_noise(mag,m_5s,band_num):
    """
    Gets the photometric noise of an LSST observation

    mag: float, magnitude of the observation
    m_5s: float, m_5sigma depth of the observatoin
    band_num: int, band numer, used to select the gamma

    returns the standard deviation of the photometric error
    """
    #https://arxiv.org/pdf/2203.09540.pdf good paper for reference
    #They assume sigma_sys = 0.004 mag since its expected sigma_sys < 0.005 mag

    gammas = [0.038, 0.039, 0.039, 0.039, 0.039, 0.039]
    gamma = gammas[band_num]

    x = 10.0 ** (0.4 * (mag - m_5s))
    sigma_rand = np.sqrt((0.04 - gamma)*x + gamma*(x**2))

    return sigma_rand

def flux_to_mag(flux):
    """
    This function converts a flux in units of  to a magnitude in units of AB mag.

    flux: flux per frequency in units of erg/s/cm^2/Hz 
    return: magnitude in units of AB mag
    """
    flux = np.clip(flux,1e-50,1e10)
    mag = -2.5*np.log10(flux) - 48.60
    return np.clip(mag,0,50)

def mag_to_flux(mag):
    """
    This function converts a magnitude in units of AB mag to a flux in units of erg/s/cm^2/Hz.

    mag: magnitude in units of AB mag
    returns: flux per frequency in units of erg/s/cm^2/Hz
    """
    return 10.0**(-0.4*(mag+48.60))

def get_observed_LC(LC,cadence_index,cadence, min_magnitude, max_magnitude, use_LSST_cadence=0):
    """
    This function takes in a light curve at a fixed cadence and returns an observed light curve
    using LSST sampling and photometric noise.

    LC: numpy array, light curve with fixed cadence in units of magnitude
    cadence_index: int, index of cadence file to use
    cadence: float, cadence of light curve in days
    min_magnitude: minimum magnitude of an LSST observation
    max_magnitude: maximum magnitude of an LSST observation
    use_LSST_cadence: 0, 1, or 2. Use default 0 for LSST cadences. Otherwise can define some different cadences.

    returns: numpy array, observed light curve, numpy array, photometric noise
    """
    # This is the nominal LSST cadences from rubin_sim
    if use_LSST_cadence == 0:
        JD_min = 60218
        time_list = []
        m_5s_list = []

        for i,band in enumerate(bandpasses):
            file = f"{file_name_for_cadence}/sample_{cadence_index}_band_{band}_dates.dat"
            time = np.loadtxt(file, usecols=0)

            m_5s = np.loadtxt(file, usecols=1)

            time -= JD_min

            max_obs = LC.shape[0]
            time = (time/cadence).round().astype(int)

            time_list.append(time)
            m_5s_list.append(m_5s)

        min_time = np.min([time_list[i].min() for i in range(len(time_list))])
        max_time = np.max([time_list[i].max() for i in range(len(time_list))])

        # Add a random shift of the start time of LSST (We use 10.5 years instead of 10 so the survey can start anywhere in the first 0.5 years)
        time_shift = np.random.randint(-min_time,LC.shape[0]-(max_time+1))
        for i,band in enumerate(bandpasses):
            time_list[i] += time_shift

        LC_obs = np.zeros(LC.shape)
        stdev = np.zeros(LC.shape)

        for i in range(len(bandpasses)):
            time = time_list[i]
            m_5s = m_5s_list[i]
            
            sigma_list = []
            for j,t in enumerate(time):
                mag = LC[t]
                sigma = LSST_photometric_noise(mag,m_5s[j],i)
                sigma_list.append(sigma)
            sigma_list = np.array(sigma_list)

            time_unique,index_unique = np.unique(time,return_index=True)
            sigma_unique = []
            for j in range(len(index_unique)):
                if j+1 < len(index_unique):
                    sigma_list_at_t = sigma_list[index_unique[j]:index_unique[j+1]]
                else:
                    sigma_list_at_t = sigma_list[index_unique[j]:]

                # combine the photoemtric noise such that the final noise is lower than the individual noise
                new_sigma = 1/np.sqrt(np.sum(1/sigma_list_at_t**2))
                sigma_unique.append(new_sigma)

            sigma_unique = np.array(sigma_unique)

            #adding systematic and rand errors in quadrature to the photometric noise
            sigma_sys = 0.005
            sigma_unique = np.sqrt(sigma_sys**2 + sigma_unique**2)

            # We clip the photometric noise to have a maximum value of 1 mag, can lead to problems if the noise is too large
            sigma_unique = np.clip(sigma_unique,None,1.0)

            # set a limit that we need at least 5 observations in a band otherwise we don't include it at all.
            if len(time_unique) > 5:
                for t,sigma in zip(time_unique,sigma_unique):
                    # We only include observations that are within the magnitude range of LSST!
                    random_noise = np.random.normal(0.0,sigma)

                    if min_magnitude <= LC[t,i] + random_noise <= max_magnitude:
                        stdev[t,i] = sigma
                        LC_obs[t,i] = LC[t,i] + random_noise

        # If we have less than 15 observations in a band, don't include the observations in that band by setting them to zero
        num_obs_limit = 15
        for i in range(len(bandpasses)):
            if np.sum(LC_obs[:,i] != 0) < num_obs_limit:
                LC_obs[:,i] = 0.0
                stdev[:,i] = 0.0

    # Test scenarios where we observe every time step
    elif use_LSST_cadence == 1:
        LC_obs = np.copy(LC)
        stdev = np.zeros(LC.shape) + 0.01
        LC_obs = LC_obs + np.random.normal(0.0,0.01,LC.shape)

        stdev[LC_obs > max_magnitude] = 0.0
        stdev[LC_obs < min_magnitude] = 0.0
        LC_obs[LC_obs > max_magnitude] = 0.0
        LC_obs[LC_obs < min_magnitude] = 0.0

        num_obs_limit = 15
        for i in range(len(bandpasses)):
            if np.sum(LC_obs[:,i] != 0) < num_obs_limit:
                LC_obs[:,i] = 0.0
                stdev[:,i] = 0.0

    # Test scenarios where we observe only at two random sections
    elif use_LSST_cadence == 2:
        LC_obs = np.copy(LC)
        stdev = np.zeros(LC.shape) + 0.01
        LC_obs = LC_obs + np.random.normal(0.0,0.01,LC.shape)

        # Observe only at two random sections
        LC_obs[:int(LC_obs.shape[0]/5)] = 0.0
        stdev[:int(LC_obs.shape[0]/5)] = 0.0

        LC_obs[int(4*LC_obs.shape[0]/5):] = 0.0
        stdev[int(4*LC_obs.shape[0]/5):] = 0.0

        LC_obs[int(2*LC_obs.shape[0]/5):int(3*LC_obs.shape[0]/5)] = 0.0
        stdev[int(2*LC_obs.shape[0]/5):int(3*LC_obs.shape[0]/5)] = 0.0

        stdev[LC_obs > max_magnitude] = 0.0
        stdev[LC_obs < min_magnitude] = 0.0
        LC_obs[LC_obs > max_magnitude] = 0.0
        LC_obs[LC_obs < min_magnitude] = 0.0

        num_obs_limit = 15
        for i in range(len(bandpasses)):
            if np.sum(LC_obs[:,i] != 0) < num_obs_limit:
                LC_obs[:,i] = 0.0
                stdev[:,i] = 0.0

    return LC_obs, stdev

def plot_recovery(xs, true_LC, _xs, mean_mask, num, epoch, extra_name="", extra_folder="", use_ylim=False): 
    """
    Function to plot the reconstructed light curve

    xs: numpy array, input light curve, shape~[T, 2*num_bands]
    true_LC: numpy array, true light curve, shape~[T, num_bands]
    _xs: numpy array, recovered light curve, shape~[T, 2*num_bands]
    num: int, number of light curves to plot
    epoch: int, epoch of light curve to plot
    extra_name: str, extra name for saving plots
    extra_folder: str, extra folder name for saving plots
    use_ylim: bool, if True use constant ylim, else zoom in for each band
    """
    # get the bands that are observed from mean_mask

    observed_bands = np.where(mean_mask)[0]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # make plot for each band seperately
    fig, axes = plt.subplots(len(observed_bands), 1, figsize=(12,1.5*max(len(observed_bands),2)),sharex=True,sharey=use_ylim)
    fig.add_subplot(111, frameon=False)

    # eliminate the points that are not observed by setting them to np.nan
    xs_obs = np.copy(xs)
    xs_obs[xs_obs == 0.0] = np.nan 
    
    # now account for only the observed bands
    lower_lim = min(np.nanmin(xs_obs[:,observed_bands]),_xs[:,observed_bands].min(),true_LC[:,observed_bands].min())
    upper_lim = max(np.nanmin(xs_obs[:,observed_bands]),_xs[:,observed_bands].max(),true_LC[:,observed_bands].max())

    full_range = upper_lim-lower_lim
    frac_of_full_range = 0.075
    ylim = (lower_lim-frac_of_full_range*full_range,upper_lim+frac_of_full_range*full_range)

    for i, band_i in enumerate(observed_bands):
        if len(observed_bands) == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.plot(days, _xs[:,band_i], color=colors[1], linewidth=1.1, label='mean pred') #,label='mean')
        alpha_list = [0.3,0.2,0.1]
        num_sigma_list = [1,2,3]
        
        mean = _xs[:,band_i]
        std = np.sqrt(np.exp(_xs[:,band_i+num_bands]))
        for alpha,num_sigma in zip(alpha_list,num_sigma_list):
            ax.fill_between(days, 
                            mean-num_sigma*std,
                            mean+num_sigma*std,
                            color=colors[1],
                            alpha=alpha, 
                            label=f"{num_sigma}$\sigma$ unc.")
        
        ax.plot(days,true_LC[:,band_i],color='black',linewidth=0.9,
                        linestyle='--',label='truth')
        error_color = colors[0]
        ax.errorbar(days, 
                            xs_obs[:,band_i],       
                            yerr=xs_obs[:,band_i+num_bands],
                            fmt='o',
                            markersize=2,
                            mfc=error_color,
                            mec=error_color,
                            ecolor=error_color,
                            elinewidth=1,
                            capsize=2,
                            label = 'obsserved')
        if use_ylim:
            ax.set_ylim(ylim)
        else:
            min_LC = np.min(true_LC[:, band_i])
            max_LC = np.max(true_LC[:, band_i])
            ax.set_ylim(min_LC-0.2*(max_LC-min_LC),max_LC+0.2*(max_LC-min_LC))

        ax.set_xlim(days.min(),days.max())
        ax.minorticks_on()
        ax.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        ax.set_ylabel(f'{bandpasses[band_i]}-band mag')
        if i == 0:
            #axes[i].legend(fontsize=10,loc='upper left',ncol=3)
            ax.legend(fontsize=10,ncol=6)
        ax.invert_yaxis()
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('time [days]',fontsize=13)
    
    fig.tight_layout()
    plt.savefig(f"{save_path}/recovery/{extra_folder}recovery{extra_name}_{num}.pdf",bbox_inches='tight')
    plt.close()

    # Makes some extra plots for diagrams
    extra_stuff = True

    if extra_stuff:
        # plot just the observations
        fig, axes = plt.subplots(len(observed_bands), 1, figsize=(12,1.5*max(len(observed_bands),2)),sharex=True,sharey=use_ylim)
        fig.add_subplot(111, frameon=False)
        for i, band_i in enumerate(observed_bands):
            if len(observed_bands) == 1:
                ax = axes
            else:
                ax = axes[i]
            ax.errorbar(days, 
                        xs_obs[:,band_i],       
                        yerr=xs_obs[:,band_i+num_bands],
                        fmt='o',
                        markersize=2,
                        mfc=error_color,
                        mec=error_color,
                        ecolor=error_color,
                        elinewidth=1,
                        capsize=2,
                        label = 'obsserved')
            if use_ylim:
                ax.set_ylim(ylim)
            else:
                min_LC = np.min(true_LC[:, band_i])
                max_LC = np.max(true_LC[:, band_i])
                ax.set_ylim(min_LC-0.2*(max_LC-min_LC),max_LC+0.2*(max_LC-min_LC))

                #ymin, ymax = axes[i].get_ylim()
                # Define the tick interval based on the desired number of ticks (3)
                #tick_interval = (ymax - ymin) / 2
                # Set the locator for the y-axis
                #axes[i].yaxis.set_major_locator(plt.MultipleLocator(tick_interval))

            ax.set_xlim(days.min(),days.max())
            ax.minorticks_on()
            ax.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            ax.set_ylabel(f'{bandpasses[band_i]}-band mag')
            ax.invert_yaxis()
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('time [days]',fontsize=13)
        fig.tight_layout()
        plt.savefig(f"{save_path}/recovery/{extra_folder}recovery_just_observations{extra_name}_{num}.pdf",bbox_inches='tight')
        plt.close()

        # plot just the true light curve
        fig, axes = plt.subplots(len(observed_bands), 1, figsize=(12,1.5*max(len(observed_bands),2)),sharex=True,sharey=use_ylim)
        fig.add_subplot(111, frameon=False)
        for i, band_i in enumerate(observed_bands):
            if len(observed_bands) == 1:
                ax = axes
            else:
                ax = axes[i]
            ax.plot(days,true_LC[:,band_i],color='black',linewidth=0.9,
                        linestyle='--',label='truth')
            if use_ylim:
                ax.set_ylim(ylim)
            else:
                min_LC = np.min(true_LC[:, band_i])
                max_LC = np.max(true_LC[:, band_i])
                ax.set_ylim(min_LC-0.2*(max_LC-min_LC),max_LC+0.2*(max_LC-min_LC))

            ax.set_xlim(days.min(),days.max())
            ax.minorticks_on()
            ax.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            ax.set_ylabel(f'{bandpasses[band_i]}-band mag')
            ax.invert_yaxis()
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('time [days]',fontsize=13)
        fig.tight_layout()
        plt.savefig(f"{save_path}/recovery/{extra_folder}recovery_just_truth{extra_name}_{num}.pdf",bbox_inches='tight')
        plt.close()

        # plot just the reconstruction
        fig, axes = plt.subplots(len(observed_bands), 1, figsize=(12,1.5*max(len(observed_bands),2)),sharex=True,sharey=use_ylim)
        fig.add_subplot(111, frameon=False)
        for i, band_i in enumerate(observed_bands):
            if len(observed_bands) == 1:
                ax = axes
            else:
                ax = axes[i]
            ax.plot(days, _xs[:,band_i], color=colors[1], linewidth=1.1, label='mean pred')
            alpha_list = [0.3,0.2,0.1]
            num_sigma_list = [1,2,3]

            mean = _xs[:,band_i]
            std = np.sqrt(np.exp(_xs[:,band_i+num_bands]))
            for alpha,num_sigma in zip(alpha_list,num_sigma_list):
                ax.fill_between(days, 
                                mean-num_sigma*std,
                                mean+num_sigma*std,
                                color=colors[1],
                                alpha=alpha, 
                                label=f"{num_sigma}$\sigma$ unc.")
            if use_ylim:
                ax.set_ylim(ylim)
            else:
                min_LC = np.min(true_LC[:, band_i])
                max_LC = np.max(true_LC[:, band_i])
                ax.set_ylim(min_LC-0.2*(max_LC-min_LC),max_LC+0.2*(max_LC-min_LC))

            ax.set_xlim(days.min(),days.max())
            ax.minorticks_on()
            ax.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)

            ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            ax.set_ylabel(f'{bandpasses[band_i]}-band mag')
            if i == 0:
                #axes[i].legend(fontsize=10,loc='upper left',ncol=3)
                ax.legend(fontsize=10,ncol=6)
            ax.invert_yaxis()
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('time [days]',fontsize=13)
        fig.tight_layout()
        plt.savefig(f"{save_path}/recovery/{extra_folder}recovery_just_reconstruction{extra_name}_{num}.pdf",bbox_inches='tight')
        plt.close()



def plot_recovery_with_driving(xs, true_LC, _xs, mean_mask, driving_reconstructed, driving_save, kernel_num_days, driving_resolution, num, epoch, extra_name="", extra_folder="", use_ylim=False): 
    """
    Function to plot the reconstructed light curve with the driving signal

    xs: numpy array, input light curve, shape~[T, 2*num_bands]
    true_LC: numpy array, true light curve, shape~[T, num_bands]
    _xs: numpy array, recovered light curve, shape~[T, 2*num_bands]
    mean_mask: numpy array, mask that is 1 if a band is observed otherwise 0, shape [num_bands]
    driving_reconstructed: numpy array, reconstructed driving signal, shape~[T, 2]
    driving_save: numpy array, true driving signal, shape~shape~[T]
    kernel_num_days: float, number of days for kernel
    driving_resolution: float, resolution of the driving signal
    num: int, number of light curves to plot
    epoch: int, epoch of light curve to plot
    extra_name: str, extra name for saving plots
    extra_folder: str, extra folder name for saving plots
    use_ylim: bool, if True use constant ylim, else zoom in for each band
    """
    # get the bands that are observed from mean_mask
    observed_bands = np.where(mean_mask)[0]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # make plot for each band seperately
    fig, axes = plt.subplots(len(observed_bands)+1, 1, figsize=(12,1.5*max(len(observed_bands)+1, 2.5)),sharex=True,sharey=use_ylim)
    fig.add_subplot(111, frameon=False)

    # eliminate the points that are not observed by setting them to np.nan
    xs_obs = np.copy(xs)
    xs_obs[xs_obs == 0.0] = np.nan 
    
    # now account for only the observed bands
    lower_lim = min(np.nanmin(xs_obs[:,observed_bands]),_xs[:,observed_bands].min(),true_LC[:,observed_bands].min())
    upper_lim = max(np.nanmin(xs_obs[:,observed_bands]),_xs[:,observed_bands].max(),true_LC[:,observed_bands].max())

    full_range = upper_lim-lower_lim
    frac_of_full_range = 0.075
    ylim = (lower_lim-frac_of_full_range*full_range,upper_lim+frac_of_full_range*full_range)

    # First plot the driving signal
    #axes[0].plot(days, driving, color=colors[2],linewidth=1.1,label='driving signal')
    
    time = np.arange(_xs.shape[0])*cadence
    padding_size = int(kernel_num_days/driving_resolution)-1
    alpha_list = [0.3,0.2,0.1]
    num_sigma_list = [1,2,3]
    driving_mean = driving_reconstructed[:, 0]
    driving_std = np.sqrt(np.exp(driving_reconstructed[:, 1]))
    
    axes[0].plot(time, driving_mean, color=colors[1],linewidth=1.1,label='mean pred')
    for alpha,num_sigma in zip(alpha_list,num_sigma_list):
        axes[0].fill_between(time, 
                        driving_mean-num_sigma*driving_std,
                        driving_mean+num_sigma*driving_std,
                        color=colors[1],
                        alpha=alpha, 
                        label=f"{num_sigma}$\sigma$ unc.")
        
    axes[0].plot(time, driving_save[padding_size:], color='black', linestyle='--', label = 'true', linewidth=0.9)
    axes[0].minorticks_on()
    axes[0].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    axes[0].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    axes[0].set_ylabel('driving mag')
    #axes[0].legend(fontsize=10,loc='upper left',ncol=3) # Just use the ledgend for the u-band
    min_driving = np.min(driving_save[padding_size:])
    max_driving = np.max(driving_save[padding_size:])
    axes[0].set_ylim(min_driving-0.2*(max_driving-min_driving),max_driving+0.2*(max_driving-min_driving))
    axes[0].set_xlim(days.min(),days.max())
    axes[0].invert_yaxis()
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    for i, band_i in enumerate(observed_bands):
        axes[i+1].plot(days, _xs[:,band_i],color=colors[1],linewidth=1.1,label='mean pred')#,label='mean')
        alpha_list = [0.3,0.2,0.1]
        num_sigma_list = [1,2,3]
        
        mean = _xs[:,band_i]
        std = np.sqrt(np.exp(_xs[:,band_i+num_bands]))
        for alpha,num_sigma in zip(alpha_list,num_sigma_list):
            axes[i+1].fill_between(days, 
                            mean-num_sigma*std,
                            mean+num_sigma*std,
                            color=colors[1],
                            alpha=alpha, 
                            label=f"{num_sigma}$\sigma$ unc.")
        
        axes[i+1].plot(days,true_LC[:,band_i], color='black',linewidth=0.9,
                        linestyle='--',label='truth')
        error_color = colors[0]
        axes[i+1].errorbar(days, 
                            xs_obs[:,band_i],       
                            yerr=xs_obs[:,band_i+num_bands],
                            fmt='o',
                            markersize=2,
                            mfc=error_color,
                            mec=error_color,
                            ecolor=error_color,
                            elinewidth=1,
                            capsize=2,
                            label = 'obsserved')
        if use_ylim:
            axes[i+1].set_ylim(ylim)
        else:
            min_LC = np.min(true_LC[:, band_i])
            max_LC = np.max(true_LC[:, band_i])
            axes[i+1].set_ylim(min_LC-0.2*(max_LC-min_LC),max_LC+0.2*(max_LC-min_LC))

        axes[i+1].set_xlim(days.min(),days.max())
        axes[i+1].minorticks_on()
        axes[i+1].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        axes[i+1].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        axes[i+1].set_ylabel(f'{bandpasses[band_i]}-band mag')
        if i == 0:
            #axes[i+1].legend(fontsize=10, loc='upper left', ncol=3)
            axes[i+1].legend(fontsize=10, ncol=6)
        axes[i+1].invert_yaxis()
        axes[i+1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('time [days]',fontsize=13)
    
    fig.tight_layout()
    plt.savefig(f"{save_path}/recovery/{extra_folder}recovery_with_driving{extra_name}_{num}.pdf",bbox_inches='tight')
    plt.close()

def plot_recovery_with_driving_and_kernels(xs, true_LC, _xs, mean_mask, driving_reconstructed, driving_save, kernel_true, kernel_pred, kernel_num_days, kernel_size, driving_resolution, num, epoch, extra_name="", extra_folder=""): 
    """
    Function to plot the reconstructed light curve with the driving signal and the kernels/transfer functions

    xs: numpy array, input light curve, shape~[T, 2*num_bands]
    true_LC: numpy array, true light curve, shape~[T, num_bands]
    _xs: numpy array, recovered light curve, shape~[T, 2*num_bands]
    mean_mask: numpy array, mask indicating observed bands
    driving_reconstructed: numpy array, reconstructed driving signal, shape~[T, 2]
    driving_save: numpy array, true driving signal, shape~[T]
    kernel_true: numpy array, true kernel, shape~[T, num_bands]
    kernel_pred: numpy array, predicted kernel, shape~[T, num_bands]
    kernel_num_days: float, number of days for kernel
    driving_resolution: float, resolution of the driving signal
    num: int, number of light curves to plot
    epoch: int, epoch of light curve to plot
    extra_name: str, additional name for saving file
    extra_folder: str, additional folder for saving file
    use_ylim: bool, whether to use the same y-axis limits for all light curves
    """

    observed_bands = np.where(mean_mask)[0]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(len(observed_bands)+1, 2, figsize=(12,1.5*max(len(observed_bands)+1,2.5)), sharex='col', gridspec_kw={'width_ratios': [4, 1]})
    #fig.add_subplot(111, frameon=False)

    xs_obs = np.copy(xs)
    xs_obs[xs_obs == 0.0] = np.nan

    lower_lim = min(np.nanmin(xs_obs[:,observed_bands]),_xs[:,observed_bands].min(),true_LC[:,observed_bands].min())
    upper_lim = max(np.nanmin(xs_obs[:,observed_bands]),_xs[:,observed_bands].max(),true_LC[:,observed_bands].max())

    full_range = upper_lim - lower_lim
    frac_of_full_range = 0.075
    ylim = (lower_lim-frac_of_full_range*full_range, upper_lim+frac_of_full_range*full_range)

    # Plot the driving signal
    time = np.arange(_xs.shape[0]) * cadence
    padding_size = int(kernel_num_days / driving_resolution) - 1
    alpha_list = [0.3, 0.2, 0.1]
    num_sigma_list = [1, 2, 3]
    driving_mean = driving_reconstructed[:, 0]
    driving_std = np.sqrt(np.exp(driving_reconstructed[:, 1]))

    font_size = 13

    # For kernel recovery plot on the right
    kernel_resolution = kernel_num_days / kernel_size
    kernel_mean = np.sum(kernel_true * np.linspace(0, kernel_num_days, kernel_size)[:, None], axis=0)
    kernel_std = np.sqrt(np.sum(kernel_true * (np.linspace(0, kernel_num_days, kernel_size)[:, None] - np.expand_dims(kernel_mean, axis=0))**2, axis=0))
    kernel_std = np.mean(kernel_std)
    kernel_std = max(kernel_std, 1.0)

    kernel_std_bins = int(np.ceil(kernel_std / kernel_resolution))
    num_sigma_plot = 4
    max_time = min(num_sigma_plot * kernel_std, kernel_num_days)
    kernel_max_time_index = int(np.ceil(max_time / kernel_resolution))

    time_kernel = np.linspace(0, max_time, kernel_max_time_index)
    full_time = np.linspace(0, kernel_num_days, kernel_size)
    
    mean_true = np.sum(full_time[..., None] * kernel_true, axis=0)
    mean_pred = np.sum(full_time[..., None] * kernel_pred, axis=0)

    ylim_kernel = (0.0, min(max(1.05 * np.max(kernel_true), 1.05 * np.max(kernel_pred)) + 0.001, 1.0))

    axes[0, 0].plot(time, driving_mean, color=colors[1], linewidth=1.1, label='mean pred')
    for alpha, num_sigma in zip(alpha_list, num_sigma_list):
        axes[0, 0].fill_between(time, 
                                driving_mean - num_sigma * driving_std,
                                driving_mean + num_sigma * driving_std,
                                color=colors[1],
                                alpha=alpha, 
                                label=f"{num_sigma}$\sigma$ unc.")
        
    axes[0, 0].plot(time, driving_save[padding_size:], color='black', linestyle='--', label='true', linewidth=0.9)
    axes[0, 0].minorticks_on()
    axes[0, 0].tick_params(which='major', direction='in', top=True, right=True, length=tick_length_major, width=tick_width)
    axes[0, 0].tick_params(which='minor', direction='in', top=True, right=True, length=tick_length_minor, width=tick_width)
    axes[0, 0].set_ylabel('driving mag', fontsize=font_size)
    min_driving = np.min(driving_save[padding_size:])
    max_driving = np.max(driving_save[padding_size:])
    axes[0, 0].set_ylim(min_driving - 0.2 * (max_driving - min_driving), max_driving + 0.2 * (max_driving - min_driving))
    axes[0, 0].set_xlim(days.min(), days.max())
    axes[0, 0].invert_yaxis()
    axes[0, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Placeholder on the right for driving signal row
    axes[0, 1].axis('off')

    # Plot the light curves and their corresponding kernel recovery
    for i, band_i in enumerate(observed_bands):
        # Light curve plot on the left
        axes[i+1, 0].plot(days, _xs[:,band_i], color=colors[1], linewidth=1.1, label='mean pred')
        mean = _xs[:,band_i]
        std = np.sqrt(np.exp(_xs[:,band_i+num_bands]))
        for alpha, num_sigma in zip(alpha_list, num_sigma_list):
            axes[i+1, 0].fill_between(days, 
                                      mean - num_sigma * std,
                                      mean + num_sigma * std,
                                      color=colors[1],
                                      alpha=alpha, 
                                      label=f"{num_sigma}$\sigma$ unc")
        
        axes[i+1, 0].plot(days, true_LC[:,band_i], color='black', linewidth=0.9, linestyle='--', label='truth')
        error_color = colors[0]
        axes[i+1, 0].errorbar(days, 
                              xs_obs[:,band_i],       
                              yerr=xs_obs[:,band_i+num_bands],
                              fmt='o',
                              markersize=2,
                              mfc=error_color,
                              mec=error_color,
                              ecolor=error_color,
                              elinewidth=1,
                              capsize=2,
                              label='obs')

        min_LC = np.min(true_LC[:, band_i])
        max_LC = np.max(true_LC[:, band_i])
        axes[i+1, 0].set_ylim(min_LC - 0.2 * (max_LC - min_LC), max_LC + 0.2 * (max_LC - min_LC))

        axes[i+1, 0].set_xlim(days.min(), days.max())
        axes[i+1, 0].minorticks_on()
        axes[i+1, 0].tick_params(which='major', direction='in', top=True, right=True, length=tick_length_major, width=tick_width)
        axes[i+1, 0].tick_params(which='minor', direction='in', top=True, right=True, length=tick_length_minor, width=tick_width)
        axes[i+1, 0].set_ylabel(f'{bandpasses[band_i]}-band mag', fontsize=font_size)
        if i == 0:
            axes[i+1, 0].legend(fontsize=10, ncol=6) # axes[i+1, 0].legend(fontsize=10, loc='upper left', ncol=6)
        axes[i+1, 0].invert_yaxis()
        axes[i+1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        axes[i+1, 1].plot(time_kernel, kernel_true[:kernel_max_time_index, band_i], linewidth=1.0, label='truth', color=colors[0])
        axes[i+1, 1].plot(time_kernel, kernel_pred[:kernel_max_time_index, band_i], linewidth=1.0, label='pred', color=colors[1])
        axes[i+1, 1].axvline(mean_true[band_i], linestyle='--', linewidth=0.9, color=colors[0]) #, label='mean truth')
        axes[i+1, 1].axvline(mean_pred[band_i], linestyle='--', linewidth=0.9, color=colors[1]) #, label='mean pred')
        axes[i+1, 1].set_ylim(ylim_kernel)
        axes[i+1, 1].minorticks_on()
        axes[i+1, 1].tick_params(which='major', direction='in', top=True, right=True, length=tick_length_major, width=tick_width)
        axes[i+1, 1].tick_params(which='minor', direction='in', top=True, right=True, length=tick_length_minor, width=tick_width)
        if i == 0:
            axes[i+1, 1].legend(fontsize=10, loc='upper right') #, ncol=2)
        #axes[i+1, 1].set_ylabel(f'{bandpasses[band_i]}-band PD', fontsize=font_size)
        axes[i+1, 1].set_ylabel(r'$\psi_{}(\tau)$'.format(bandpasses[band_i]), fontsize=font_size)
        #axes[i+1, 1].yaxis.set_label_position('right')
        axes[i+1, 1].set_xlim(time_kernel.min(), time_kernel.max())
        axes[i+1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[-1, 0].set_xlabel('time [days]', fontsize=font_size)
    axes[-1, 1].set_xlabel(r'$\tau$ [days]', fontsize=font_size)

    fig.tight_layout(pad=1.05)
    plt.savefig(f"{save_path}/recovery/{extra_folder}recovery_with_driving_and_kernel{extra_name}_{num}.pdf",bbox_inches='tight')
    plt.close()


def plot_recovery_iterative(xs, true_LC, _xs_list, _xs, refererence_band, num, epoch, extra_name="", extra_folder="", use_ylim=False): 
    """
    Function to plot the reconstructed light curve for each iteration

    xs: numpy array, input light curve, shape~[T, num_bands]
    true_LC: numpy array, true light curve, shape~[T, num_bands]
    _xs_list: list of numpy array, list of recovered light curves for each iteration, shape~[iterations, T, num_bands]
    _xs: numpy array, final recovered light curve, shape~[T, 2*num_bands], 2*num_bands because of the uncertainty
    refererence_band: int, reference band for driving signal
    num: int, number of light curves to plot
    epoch: int, epoch of light curve to plot
    extra_name: str, additional name for saving file
    extra_folder: str, additional folder for saving file
    use_ylim: bool, whether to use the same y-axis limits for all light curves
    """

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # make plot for each band seperately
    fig, axes = plt.subplots(_xs_list.shape[0], 1, figsize=(12,1.5*_xs_list.shape[0]),sharex=True,sharey=use_ylim)
    fig.add_subplot(111, frameon=False)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # eliminate the points that are not observed by setting them to np.nan
    xs_obs = np.copy(xs)
    xs_obs[xs_obs == 0.0] = np.nan 

    band_i = refererence_band
    lower_lim = min(np.nanmin(xs_obs[:,:num_bands]),_xs_list[:,:,band_i].min(),true_LC[:,band_i].min())
    upper_lim = max(np.nanmin(xs_obs[:,:num_bands]),_xs_list[:,:,band_i].max(),true_LC[:,band_i].max())
    full_range = upper_lim-lower_lim
    frac_of_full_range = 0.1
    ylim = (lower_lim-frac_of_full_range*full_range,upper_lim+frac_of_full_range*full_range)

    for iteration in range(_xs_list.shape[0]):
        mean = _xs_list[iteration, :, band_i]
        axes[iteration].plot(days, mean, color=colors[1],linewidth=1.1,label='mean pred')
        alpha_list = [0.3,0.2,0.1]
        num_sigma_list = [1,2,3]
        
        std = np.sqrt(np.exp(_xs[:, band_i+num_bands]))
        for alpha,num_sigma in zip(alpha_list,num_sigma_list):
            axes[iteration].fill_between(days, 
                                        mean-num_sigma*std,
                                        mean+num_sigma*std,
                                        color=colors[1],
                                        alpha=alpha, 
                                        label=f"{num_sigma}$\sigma$ unc.")
        
        axes[iteration].plot(days,true_LC[:,band_i],color='black',linewidth=0.9,
                        linestyle='--',label='truth')
        error_color = colors[0]
        axes[iteration].errorbar(days, 
                            xs_obs[:,band_i],       
                            yerr=xs_obs[:,band_i+num_bands],
                            fmt='o',
                            markersize=2,
                            mfc=error_color,
                            mec=error_color,
                            ecolor=error_color,
                            elinewidth=1,
                            capsize=2,
                            label = 'obsserved')
        
        if use_ylim:
            axes[iteration].set_ylim(ylim)
        else:
            min_LC = np.min(true_LC[:, band_i])
            max_LC = np.max(true_LC[:, band_i])
            axes[iteration].set_ylim(min_LC-0.2*(max_LC-min_LC),max_LC+0.2*(max_LC-min_LC))

        axes[iteration].set_xlim(days.min(),days.max())
        axes[iteration].minorticks_on()
        axes[iteration].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        axes[iteration].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        axes[iteration].set_ylabel(f'iteration {iteration+1}')
        if iteration == len(_xs_list)-1:
            axes[iteration].legend(fontsize=10.0,loc='upper left',ncol=6)
        axes[iteration].invert_yaxis()
        axes[iteration].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('time [days]',fontsize=13)
    
    fig.tight_layout()
    plt.savefig(f"{save_path}/recovery/{extra_folder}recovery_iterative{extra_name}_{num}.pdf",bbox_inches='tight')
    plt.close()

def plot_kernel_recovery(kernel_true, kernel_pred, num, epoch, kernel_num_days, kernel_size, extra_folder=""):
    """
    Function that plots the reconstructed kernels/transfer functions

    kernel_true: numpy array, true kernel, shape~[T, num_bands]
    kernel_pred: numpy array, predicted kernel, shape~[T, num_bands]
    num: int, number of light curves to plot
    epoch: int, epoch of light curve to plot
    kernel_num_days: int, number of days for transfer function kernels
    kernel_size: int, number of bins for transfer function kernels
    extra_folder: str, additional folder for saving file
    """
    kernel_resolution = kernel_num_days/kernel_size

    kernel_mean = np.sum(kernel_true*np.linspace(0,kernel_num_days, kernel_size)[:, None],axis=0)
    kernel_std = np.sqrt(np.sum(kernel_true*(np.linspace(0,kernel_num_days, kernel_size)[:, None]-np.expand_dims(kernel_mean,axis=0))**2,axis=0))
    kernel_std = np.mean(kernel_std)
    kernel_std = max(kernel_std, 1.0) # make sure it is at least 1 day

    kernel_std_bins = int(np.ceil(kernel_std/kernel_resolution))

    # get first non-zero index
    # plot out to 3 sigma
    num_sigma_plot = 4
    max_time = num_sigma_plot*kernel_std
    max_time = min(max_time, kernel_num_days)

    kernel_max_time_index = int(np.ceil(max_time/kernel_resolution))

    time = np.linspace(0,max_time, kernel_max_time_index)
    full_time = np.linspace(0,kernel_num_days, kernel_size)
    
    mean_true = np.sum(full_time[...,None]*kernel_true,axis=0)
    #mean_true = np.log10(mean_true)
    mean_pred = np.sum(full_time[...,None]*kernel_pred,axis=0)
    #mean_pred = np.log10(mean_pred)

    ylim = (0.0, min(max(1.05*np.max(kernel_true),1.05*np.max(kernel_pred))+0.001,1.0))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
    fig.add_subplot(111, frameon=False)

    for band_i in range(num_bands):

        axes[band_i].plot(time, kernel_true[:kernel_max_time_index, band_i],linewidth=1.0,label='truth',color=colors[0])
        axes[band_i].plot(time, kernel_pred[:kernel_max_time_index, band_i],linewidth=1.0,label='pred',color=colors[1])
        axes[band_i].axvline(mean_true[band_i],linestyle='--',linewidth=0.9,color=colors[0])  #label='log mean')
        axes[band_i].axvline(mean_pred[band_i],linestyle='--',linewidth=0.9,color=colors[1])  #label='log mean')
        axes[band_i].set_ylim(ylim)
        axes[band_i].set_xlim(0,max_time) 
        axes[band_i].minorticks_on()
        axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band PD')
        if band_i == 0:
            axes[band_i].legend(fontsize=12,loc='upper right')

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('time [days]',fontsize=13)
    
    fig.tight_layout()
    plt.savefig(f"{save_path}/recovery/{extra_folder}kernels_{num}.pdf",bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
    fig.add_subplot(111, frameon=False)
    
    for band_i in range(num_bands):

        axes[band_i].plot(full_time, kernel_true[:, band_i],linewidth=1.0,label='truth',color=colors[0])
        axes[band_i].plot(full_time, kernel_pred[:, band_i],linewidth=1.0,label='pred',color=colors[1])
        axes[band_i].axvline(mean_true[band_i],linestyle='--',linewidth=0.9,color=colors[0])  #label='log mean')
        axes[band_i].axvline(mean_pred[band_i],linestyle='--',linewidth=0.9,color=colors[1])  #label='log mean')
        axes[band_i].set_ylim(ylim)
        axes[band_i].set_xlim(kernel_resolution,kernel_num_days) # Don't plot the first point, which is 0 so diverges in log space
        # set x-axis to log scale
        axes[band_i].set_xscale('log')
        axes[band_i].minorticks_on()
        axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band PD')
        if band_i == 0:
            axes[band_i].legend(fontsize=12,loc='upper right')

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('time [days]',fontsize=13)
    
    fig.tight_layout()
    plt.savefig(f"{save_path}/recovery/{extra_folder}kernels_log_{num}.pdf",bbox_inches='tight')
    plt.close()

def plot_kernel_recovery_reconstruction(kernel_pred, num, epoch, kernel_num_days, kernel_size, reference_band=3, extra_folder=""):
    """
    Plot just the reconstructed kernel/transfer functions

    kernel_pred: numpy array, predicted kernel, shape~[T, num_bands]
    num: int, number of light curves to plot
    epoch: int, epoch of light curve to plot
    kernel_num_days: int, number of days for transfer function kernels
    kernel_size: int, number of bins for transfer function kernels
    refererence_band: int, reference band to plot one example
    extra_folder: str, additional folder for saving file
    """
    kernel_resolution = kernel_num_days/kernel_size

    kernel_mean = np.sum(kernel_pred*np.linspace(0,kernel_num_days, kernel_size)[:, None],axis=0)
    kernel_std = np.sqrt(np.sum(kernel_pred*(np.linspace(0,kernel_num_days, kernel_size)[:, None]-np.expand_dims(kernel_mean,axis=0))**2,axis=0))
    kernel_std = np.mean(kernel_std)
    kernel_std = max(kernel_std, 1.0) # make sure it is at least 1 day

    kernel_std_bins = int(np.ceil(kernel_std/kernel_resolution))

    # get first non-zero index
    # plot out to 3 sigma
    num_sigma_plot = 4
    max_time = num_sigma_plot*kernel_std
    max_time = min(max_time, kernel_num_days)

    kernel_max_time_index = int(np.ceil(max_time/kernel_resolution))

    time = np.linspace(0,max_time, kernel_max_time_index)
    full_time = np.linspace(0,kernel_num_days, kernel_size)

    mean_pred = np.sum(full_time[...,None]*kernel_pred,axis=0)

    max_y = min(1.05*np.max(kernel_pred)+0.001,1.0)


    labels = ['u', 'g', 'r' , 'i', 'z', 'y']
    colors = ['violet', 'g', 'r' , 'brown', 'grey', 'black']

    plt.figure(figsize=(12, 4))
    for band_i in range(num_bands):
        plt.plot(time, kernel_pred[:kernel_max_time_index, band_i],linewidth=1.0,label=f'{labels[band_i]}-band',color=colors[band_i])
    plt.xlim(0.0, max_time)
    plt.ylim(0.0, max_y)
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    #plt.ylabel('Prob. Density',fontsize=13)
    plt.ylabel(r'$\psi(\tau)$',fontsize=13)
    plt.legend(fontsize=12,loc='upper right',ncol=6)
    #plt.xlabel('time [days]',fontsize=13)
    plt.xlabel(r'$\tau$ [days]',fontsize=13)
    plt.savefig(f"{save_path}/recovery/{extra_folder}kernels_together_{num}.pdf",bbox_inches='tight')
    plt.close()

    # plot just the reference band
    plt.figure(figsize=(12, 4))
    plt.plot(time, kernel_pred[:kernel_max_time_index, reference_band],linewidth=1.0,label=f'{labels[reference_band]}-band',color=colors[reference_band])
    plt.xlim(0.0, max_time)
    plt.ylim(0.0, max_y)
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    #plt.ylabel('Prob. Density',fontsize=13)
    plt.ylabel(r'$\psi(\tau)$',fontsize=13)
    plt.xlabel(r'$\tau$ [days]',fontsize=13)
    plt.savefig(f"{save_path}/recovery/{extra_folder}kernels_just_{labels[reference_band]}_{num}.pdf",bbox_inches='tight')
    plt.close()

def plot_kernel_recovery_iterative(kernel_true, kernel_pred_list, num, epoch, kernel_num_days, kernel_size, extra_folder=""):
    """
    Function to plot the reconstructed kernels/transfer functions for each iteration

    kernel_true: numpy array, true kernel, shape~[T, num_bands]
    kernel_pred: numpy array, predicted kernel, shape~[iterations, T, num_bands]
    num: int, number of light curves to plot
    epoch: int, epoch of light curve to plot
    kernel_num_days: int, number of days for transfer function kernels
    kernel_size: int, number of bins for transfer function kernels
    extra_folder: str, additional folder for saving file
    """
    kernel_resolution = kernel_num_days/kernel_size

    kernel_mean = np.sum(kernel_true*np.linspace(0,kernel_num_days, kernel_size)[:, None],axis=0)
    kernel_std = np.sqrt(np.sum(kernel_true*(np.linspace(0,kernel_num_days, kernel_size)[:, None]-np.expand_dims(kernel_mean,axis=0))**2,axis=0))
    kernel_std = np.mean(kernel_std)
    kernel_std = max(kernel_std, 1.0) # make sure it is at least 1 day

    kernel_std_bins = int(np.ceil(kernel_std/kernel_resolution))

    # plot out to 3 sigma
    num_sigma_plot = 4
    max_time = num_sigma_plot*kernel_std
    max_time = min(max_time, kernel_num_days)

    kernel_max_time_index = int(np.ceil(max_time/kernel_resolution))

    time = np.linspace(0,max_time, kernel_max_time_index)
    full_time = np.linspace(0,kernel_num_days, kernel_size)

    ylim = (0.0, min(max(1.05*np.max(kernel_true),1.05*np.max(kernel_pred_list))+0.001,1.0))

    fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
    fig.add_subplot(111, frameon=False)

    for band_i in range(num_bands): 
        axes[band_i].plot(time, kernel_true[:kernel_max_time_index,band_i],linewidth=1.0,label='truth')
        for j in range(kernel_pred_list.shape[0]):
            axes[band_i].plot(time, kernel_pred_list[j, :kernel_max_time_index, band_i],linewidth=1.0,label=f'iter {j+1}')

        axes[band_i].set_ylim(ylim)
        axes[band_i].set_xlim(0,max_time)
        axes[band_i].minorticks_on()
        axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band PD')
        if band_i == 0:
            axes[band_i].legend(fontsize=12,loc='upper right',ncol=4)

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('time [days]',fontsize=13)
    
    fig.tight_layout()
    plt.savefig(f"{save_path}/recovery/{extra_folder}kernels_iterative_{num}.pdf",bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
    fig.add_subplot(111, frameon=False)

    for band_i in range(num_bands):
        axes[band_i].plot(full_time, kernel_true[:, band_i],linewidth=1.0,label='truth')
        for j in range(kernel_pred_list.shape[0]):
            axes[band_i].plot(full_time, kernel_pred_list[j, :, band_i],linewidth=1.0,label=f'iter {j+1}')

        axes[band_i].set_ylim(ylim)
        axes[band_i].set_xlim(kernel_resolution,kernel_num_days)
        # set x-axis to log scale
        axes[band_i].set_xscale('log')
        axes[band_i].minorticks_on()
        axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band PD')
        if band_i == 0:
            axes[band_i].legend(fontsize=12,loc='upper right',ncol=4)

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('time [days]',fontsize=13)
    
    fig.tight_layout()
    plt.savefig(f"{save_path}/recovery/{extra_folder}kernels_iterative_log_{num}.pdf",bbox_inches='tight')
    plt.close()

def plot_kernel_recovery_sample(kernel_true, kernel, kernel_pred_samples, num, epoch, kernel_num_days, kernel_size, extra_folder=""):
    """
    Function to plot the reconstructed kernels/transfer functions drawing samples from the physical parameter posteriors, instread of just the mean reconstruction

    kernel_true: numpy array, true kernel, shape~[T, num_bands]
    kernel: numpy array, mean predicted kernel, shape~[T, num_bands]
    kernel_pred_samples: numpy array, sampled predicted kernel, shape~[num_samples, T, num_bands]
    num: int, number of light curves to plot
    epoch: int, epoch of light curve to plot
    kernel_num_days: int, number of days for transfer function kernels
    kernel_size: int, number of bins for transfer function kernels
    extra_folder: str, additional folder for saving file
    """

    kernel_resolution = kernel_num_days/kernel_size

    kernel_mean = np.sum(kernel_true*np.linspace(0,kernel_num_days, kernel_size)[:, None],axis=0)
    kernel_std = np.sqrt(np.sum(kernel_true*(np.linspace(0,kernel_num_days, kernel_size)[:, None]-np.expand_dims(kernel_mean,axis=0))**2,axis=0))
    kernel_std = np.mean(kernel_std)
    kernel_std = max(kernel_std, 1.0) # make sure it is at least 1 day
    
    kernel_std_bins = int(np.ceil(kernel_std/kernel_resolution))

    # plot out to 3 sigma
    num_sigma_plot = 4
    max_time =  num_sigma_plot*kernel_std
    max_time = min(max_time, kernel_num_days)

    kernel_max_time_index = int(np.ceil(max_time/kernel_resolution))

    time = np.linspace(0,max_time, kernel_max_time_index)
    full_time = np.linspace(0,kernel_num_days, kernel_size)

    ylim = (0.0, min(max(1.05*np.max(kernel_true),1.05*np.max(kernel_pred_samples))+0.001,1.0))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    num_samples = kernel_pred_samples.shape[0]
    alpha = 0.85

    fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
    fig.add_subplot(111, frameon=False)

    for band_i in range(num_bands):

        axes[band_i].plot(time, kernel_true[:kernel_max_time_index, band_i],linewidth=1.0,label='truth',color=colors[0])
        axes[band_i].plot(time, kernel[:kernel_max_time_index, band_i],linewidth=1.0,label='mean pred',color=colors[1])
        for pred_num in range(num_samples):
            if pred_num == 0:
                axes[band_i].plot(time, kernel_pred_samples[pred_num, :kernel_max_time_index, band_i],linewidth=1.0,label=f'sampled pred',color=colors[2],alpha=alpha, linestyle='--')
            else:
                axes[band_i].plot(time, kernel_pred_samples[pred_num, :kernel_max_time_index, band_i],linewidth=1.0,color=colors[2],alpha=alpha, linestyle='--')
        axes[band_i].set_ylim(ylim)
        axes[band_i].set_xlim(0,max_time)
        axes[band_i].minorticks_on()
        axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band PD')
        if band_i == 0:
            axes[band_i].legend(fontsize=12,loc='upper right')

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('time [days]',fontsize=13)
    
    fig.tight_layout()
    plt.savefig(f"{save_path}/recovery/{extra_folder}kernels_sampling_{num}.pdf",bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(num_bands, 1, figsize=(12,1.5*num_bands),sharex=True,sharey=True)
    fig.add_subplot(111, frameon=False)

    for band_i in range(num_bands):

        axes[band_i].plot(full_time, kernel_true[:, band_i],linewidth=1.0,label='truth',color=colors[0])
        axes[band_i].plot(full_time, kernel[:, band_i],linewidth=1.0,label='mean pred',color=colors[1])
        for pred_num in range(num_samples):
            if pred_num == 0:
                axes[band_i].plot(full_time, kernel_pred_samples[pred_num, :, band_i],linewidth=1.0,label=f'sampled pred',color=colors[2],alpha=alpha, linestyle='--')
            else:
                axes[band_i].plot(full_time, kernel_pred_samples[pred_num, :, band_i],linewidth=1.0,color=colors[2],alpha=alpha, linestyle='--')
        axes[band_i].set_ylim(ylim)
        axes[band_i].set_xlim(kernel_resolution,kernel_num_days) 
        # make x-axis log scale
        axes[band_i].set_xscale('log')
        axes[band_i].minorticks_on()
        axes[band_i].tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
        axes[band_i].tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
        axes[band_i].set_ylabel(f'{bandpasses[band_i]}-band PD')
        if band_i == 0:
            axes[band_i].legend(fontsize=12,loc='upper right')

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('time [days]',fontsize=13)
    
    fig.tight_layout()
    plt.savefig(f"{save_path}/recovery/{extra_folder}kernels_sampling_log_{num}.pdf",bbox_inches='tight')
    plt.close()

def build_data_set(kernel_num_days, kernel_resolution, seed, model_spectrum, lsst_filter, cosmo, emline_template, galaxy_template, reddening_curve, min_magnitude, max_magnitude, reference_band, cadence, augment, save_true_LC, custom_driving_signal=None, use_LSST_cadence=0):
    """
    This function builds a data set for a single light curve. Used by the PyTorch dataloader to build the training and test sets on the fly.

    kernel_num_days: int, number of dats in the transfer function kernels
    kernel_resolution: float, resolution of the transfer function bins
    seed: list, list of seeds used if augment == False so the test set can be fixed.
    model_spectrum: bool, whether to model the spectrum to get the mean magnitude of each band, otherwise just select at random. Should be True except for testing.
    lsst_filter: speclight filter object, used to get the bandpasses for the LSST filters
    cosmo: astropy cosmology object, used to get the distance modulus for the redshift
    emline_template: numpy array, template for the emission lines
    galaxy_template: numpy array, template for the galaxy
    reddening_curve: numpy array, template for the reddening curve
    min_magnitude: float, minimum magnitude observed in the light curve
    max_magnitude: float, maximum magnitude observed in the light curve
    reference_band: int, reference band for the driving signal (0-5 for LSSt bands). Generally set to the the i-band (3).
    cadence: float, time spacing in each bin of the light curve
    augment: bool, whether to augment the data set or use a fixed seed (from seed)
    save_true_LC: bool, whether to save the true light curve for validation
    custom_driving_signal: function, function to generate the driving signal. If None, use default broken power-law.
    use_LSST_cadence: int, a few different cadence schemes included, 0 is the default of LSST cadence, 1 is every day, 2 is a few observed sections.

    returns: dictionary to be batched by the dataloader
    """
    if not augment:
        np.random.seed(seed)

    # Get the light curve, transfer function, driving signal, and parameters
    brightness_flag = True
    while brightness_flag:
        LC , transfer_function_save, driving_save, params, band_mean_mag = load_LC(
                                                                                    kernel_num_days=kernel_num_days,
                                                                                    kernel_resolution=kernel_resolution,
                                                                                    model_spectrum=model_spectrum,
                                                                                    lsst_filter=lsst_filter,
                                                                                    cosmo=cosmo,
                                                                                    emline_template=emline_template,
                                                                                    galaxy_template=galaxy_template,
                                                                                    reddening_curve=reddening_curve,
                                                                                    reference_band=reference_band,
                                                                                    min_magnitude=min_magnitude,
                                                                                    max_magnitude=max_magnitude,
                                                                                    custom_driving_signal=custom_driving_signal,
                                                                                    )
        if LC is not None:
            brightness_flag = False

    driving_save = np.expand_dims(driving_save, axis=1).astype(np.float32)
    max_time = LC.shape[0]
    transfer_function_save = transfer_function_save.astype(np.float32)

    # Get parameters we want to predict and scale them from 0 to 1
    params_list = [] 
    for par in parameters_keys:
        value = params[par]
        #scale all the parameters from 0 to 1
        scaled_value = (value-min_max_dict[par][0])/(min_max_dict[par][1]-min_max_dict[par][0])
        params_list.append(scaled_value)
    params_list = np.array(params_list)

    if give_redshift:
        #scale redshift from 0 to 1
        redshift = params['redshift']
        redshift = (redshift-min_max_dict['redshift'][0])/(min_max_dict['redshift'][1]-min_max_dict['redshift'][0])
    params_list = params_list.astype(np.float32)
    LC = LC.astype(np.float32)

    #save the true LC for validation
    if save_true_LC:
        true_LC = np.copy(LC)

    #Produce an observed LC with realistic cadence and photometric noise
    cadence_index = np.random.randint(num_cadences)
    LC, stdev = get_observed_LC(LC=LC,
                                cadence_index=cadence_index,
                                cadence=cadence,
                                min_magnitude=min_magnitude,
                                max_magnitude=max_magnitude,
                                use_LSST_cadence=use_LSST_cadence)

    mask = (LC != 0.0).astype(np.float32)

    x = np.linspace(0.0,1.0,LC.shape[0]).astype(np.float32)

    #add error bars to light curve array 
    LC = np.concatenate((LC,stdev),axis=1).astype(np.float32)
    
    # replace nan and inf with 0 just in case
    LC = np.nan_to_num(LC,nan=0.0,posinf=0.0,neginf=0.0)
    true_LC = np.nan_to_num(true_LC,nan=0.0,posinf=0.0,neginf=0.0)
    transfer_function_save = np.nan_to_num(transfer_function_save,nan=0.0,posinf=0.0,neginf=0.0)
    driving_save = np.nan_to_num(driving_save,nan=0.0,posinf=0.0,neginf=0.0)
    band_mean_mag = np.nan_to_num(band_mean_mag,nan=0.0,posinf=0.0,neginf=0.0)
    
    sample = {"x":x, "y":LC, "params":params_list, "transfer_function_save":transfer_function_save, "driving_save":driving_save, "band_mean_mag":band_mean_mag, "seed":seed}
    if save_true_LC:
        sample["true_LC"] = true_LC
    if give_redshift:
        sample["redshift"] = redshift

    return sample

def custom_driving_signal_sine(N, dt, driving_mean_mag, standard_deviation, log_nu_b, alpha_L, alpha_H_minus_L, extra_time_factor):
    """
    Function to generate a sine wave driving signal for testing
    
    N: int, number of points in the light curve
    dt: float, time spacing in the light curve
    driving_mean_mag: float, mean magnitude of the driving signal
    standard_deviation: float, standard deviation of the driving signal
    log_nu_b: float, log of the break frequency of the broken power law, not used in this case
    alpha_L: float, low frequency slope of the broken power law, not used in this case
    alpha_H_minus_L: float, high frequency slope minus low frequency slope of the broken power law, not used in this case
    extra_time_factor: int, factor to extend the time series to make sure the sine wave has random phase and asymtotic mean and standard deviation
    """
    t = np.arange(extra_time_factor*N)*dt
    #period = 5.0 / 10**log_nu_b
    period = 10.0**np.random.uniform(1.0, 3.0)

    phase = np.random.uniform(0, 2*np.pi)

    x = np.sin(2 * np.pi * t / period - phase)
    x = x - np.mean(x)
    x = x / np.std(x)
    x = x * standard_deviation + driving_mean_mag
    x = x[:N]

    return x

def custom_driving_signal_sine_with_BPL(N, dt, driving_mean_mag, standard_deviation, log_nu_b, alpha_L, alpha_H_minus_L, extra_time_factor):
    """
    Function to generate a sine wave driving signal with a broken power law added for testing

    N: int, number of points in the light curve
    dt: float, time spacing in the light curve
    driving_mean_mag: float, mean magnitude of the driving signal
    standard_deviation: float, standard deviation of the driving signal
    log_nu_b: float, log of the break frequency of the broken power law
    alpha_L: float, low frequency slope of the broken power law
    alpha_H_minus_L: float, high frequency slope minus low frequency slope of the broken power law
    extra_time_factor: int, factor to extend the time series has random phase and asymtotic mean and standard deviation
    """
    #generate_variability(N, dt, mean_mag, standard_deviation, log_nu_b, alpha_L, alpha_H_minus_L, redshift, extra_time_factor = 8, plot=False):
    t = np.arange(extra_time_factor*N)*dt

    period = 10.0**np.random.uniform(1.0, 3.0)

    phase = np.random.uniform(0, 2*np.pi)

    x = np.sin(2 * np.pi * t / period - phase)

    # Add broken power law to the sine wave
    relative_amplitude = np.random.uniform(0.1, 2.0)

    x = x + relative_amplitude* generate_variability(N=extra_time_factor*N, 
                                                    dt=dt, 
                                                    mean_mag=0, 
                                                    standard_deviation=1.0, 
                                                    log_nu_b=log_nu_b,
                                                    alpha_L=alpha_L,
                                                    alpha_H_minus_L=alpha_H_minus_L,
                                                    redshift=0.0,
                                                    extra_time_factor=2, 
                                                    plot=False,
                                                    )

    x = x - np.mean(x)
    x = x / np.std(x)
    x = x * standard_deviation + driving_mean_mag
    x = x[:N]

    return x

def custom_driving_signal_DRW(N, dt, driving_mean_mag, standard_deviation, log_nu_b, alpha_L, alpha_H_minus_L, extra_time_factor):
    """
    Function to generate a DRW driving signal for testing. Same as broken power law with alpha_L = 0.0, alpha_H_minus_L = 2.0.

    N: int, number of points in the light curve
    dt: float, time spacing in the light curve
    driving_mean_mag: float, mean magnitude of the driving signal
    standard_deviation: float, standard deviation of the driving signal
    log_nu_b: float, log of the break frequency of the broken power law
    alpha_L: float, low frequency slope of the broken power law, not used in this case since set to 0.0
    alpha_H_minus_L: float, high frequency slope minus low frequency slope of the broken power law, not used in this case since set to 2.0
    extra_time_factor: int, factor to extend the time series has asymtotic mean and standard deviation
    """
    return generate_variability(N=N,
                                dt=dt,
                                mean_mag=driving_mean_mag,
                                standard_deviation=standard_deviation,
                                log_nu_b=log_nu_b,
                                alpha_L= 0.0,
                                alpha_H_minus_L= 2.0,
                                redshift=0.0,
                                extra_time_factor=extra_time_factor,
                                plot=False,
                                )

def custom_driving_signal_two_sine(N, dt, driving_mean_mag, standard_deviation, log_nu_b, alpha_L, alpha_H_minus_L, extra_time_factor):
    """
    Function to generate a two sine wave driving signal for testing

    N: int, number of points in the light curve
    dt: float, time spacing in the light curve
    driving_mean_mag: float, mean magnitude of the driving signal
    standard_deviation: float, standard deviation of the driving signal
    log_nu_b: float, log of the break frequency of the broken power law, not used in this case
    alpha_L: float, low frequency slope of the broken power law, not used in this case
    alpha_H_minus_L: float, high frequency slope minus low frequency slope of the broken power law, not used in this case
    extra_time_factor: int, factor to extend the time series to make sure the sine waves have random phase and asymtotic mean and standard deviation
    """
    t = np.arange(extra_time_factor*N)*dt
    period1 = 10.0**np.random.uniform(1.0, 3.0)
    period2 = 10.0**np.random.uniform(1.0, 3.0)

    phase1 = np.random.uniform(0, 2*np.pi)
    phase2 = np.random.uniform(0, 2*np.pi)

    relative_amplitude = np.random.uniform(0.1, 1.0)

    x = np.sin(2 * np.pi * t / period1 - phase1) + relative_amplitude * np.sin(2 * np.pi * t / period2 - phase2)
    x = x - np.mean(x)
    x = x / np.std(x)
    x = x * standard_deviation + driving_mean_mag
    x = x[:N]

    return x

def custom_driving_signal_sawtooth(N, dt, driving_mean_mag, standard_deviation, log_nu_b, alpha_L, alpha_H_minus_L, extra_time_factor):
    """
    Function to generate a sawtooth wave driving signal for testing

    N: int, number of points in the light curve
    dt: float, time spacing in the light curve
    driving_mean_mag: float, mean magnitude of the driving signal
    standard_deviation: float, standard deviation of the driving signal
    log_nu_b: float, log of the break frequency of the broken power law, not used in this case
    alpha_L: float, low frequency slope of the broken power law, not used in this case
    alpha_H_minus_L: float, high frequency slope minus low frequency slope of the broken power law, not used in this case
    extra_time_factor: int, factor to extend the time series to make sure the sawtooth has random phase and asymtotic mean and standard deviation
    """    
    t = np.arange(extra_time_factor*N)*dt
    period = 10.0**np.random.uniform(1.0, 3.0)

    phase = np.random.uniform(0, 2*np.pi)
    width = np.random.uniform(0., 1.0)

    x = sawtooth(2 * np.pi * t / period - phase, width=width)
    x = x - np.mean(x)
    x = x / np.std(x)
    x = x * standard_deviation + driving_mean_mag
    x = x[:N]

    return x

def custom_driving_signal_square_wave(N, dt, driving_mean_mag, standard_deviation, log_nu_b, alpha_L, alpha_H_minus_L, extra_time_factor):
    """
    Function to generate a square wave driving signal for testing
    
    N: int, number of points in the light curve
    dt: float, time spacing in the light curve
    driving_mean_mag: float, mean magnitude of the driving signal
    standard_deviation: float, standard deviation of the driving signal
    log_nu_b: float, log of the break frequency of the broken power law, not used in this case
    alpha_L: float, low frequency slope of the broken power law, not used in this case
    alpha_H_minus_L: float, high frequency slope minus low frequency slope of the broken power law, not used in this case
    extra_time_factor: int, factor to extend the time series to make sure the squarewave has random phase and asymtotic mean and standard deviation
    """
    t = np.arange(extra_time_factor*N)*dt
    period = 10.0**np.random.uniform(1.0, 3.0)

    phase = np.random.uniform(0, 2*np.pi)
    width = np.random.uniform(0.1, 0.9)

    x = square(2 * np.pi * t / period - phase, duty=width)
    x = x - np.mean(x)
    x = x / np.std(x)
    x = x * standard_deviation + driving_mean_mag
    x = x[:N]

    return x


class Dataset_Loader(Dataset):
    """
    Pytorch Dataloader
    """
    def __init__(self, kernel_num_days, kernel_resolution, seed_list, cadence, model_spectrum, cosmo, emline_template, galaxy_template, reddening_curve, min_magnitude, max_magnitude, reference_band, custom_driving_signal=None, use_LSST_cadence=0, augment=False, save_true_LC=False):
        """
        kernel_num_days: int, number of days in the transfer function kernels
        kernel_resolution: float, resolution of the transfer function kernels
        seed_list: list, list of seeds for the random number generator
        cadence: float, time spacing in the light curve
        model_spectrum: bool, whether to model the spectrum to get the mean magnitude of each band, otherwise just select at random. Should be True except for testing.
        cosmo: astropy cosmology object, used to get the distance modulus for the redshift
        emline_template: numpy array, template for the emission lines
        galaxy_template: numpy array, template for the galaxy
        reddening_curve: numpy array, template for the reddening curve
        min_magnitude: float, minimum magnitude observed in the light curve
        max_magnitude: float, maximum magnitude observed in the light curve
        reference_band: int, reference band for the driving signal (0-5 for LSSt bands). Generally set to the the i-band (3).
        custom_driving_signal: function, function to generate the driving signal. If None, use default broken power-law.
        use_LSST_cadence: int, a few different cadence schemes included, 0 is the default of LSST cadence, 1 is every day, 2 is a few observed sections.
        augment: bool, whether to augment the data set or use a fixed seed (from seed)
        save_true_LC: bool, whether to save the true light curve for validation
        """
        self.kernel_num_days = kernel_num_days
        self.kernel_resolution = kernel_resolution
        self.seed_list = seed_list
        self.augment = augment
        self.cadence = cadence
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.reference_band = reference_band
        self.model_spectrum = model_spectrum
        self.save_true_LC = save_true_LC
        self.custom_driving_signal = custom_driving_signal
        self.use_LSST_cadence = use_LSST_cadence

        if model_spectrum:
            # Choose the LSST filter from scpeclite
            self.lsst_filter = filters.load_filters('lsst2016-*')
            self.cosmo = cosmo
            self.emline_template = emline_template
            self.galaxy_template = galaxy_template
            self.reddening_curve = reddening_curve

    def __len__(self):
        return len(self.seed_list)

    def __getitem__(self, index):
        
        seed = self.seed_list[index] # seed for random number generator, only used if augment = True

        # Regenerate the light curve if it is too faint or too bright to be observed by LSST
        flag = True
        while flag:
            sample = build_data_set(
                                    kernel_num_days=self.kernel_num_days, 
                                    kernel_resolution=self.kernel_resolution, 
                                    seed=seed, 
                                    model_spectrum=self.model_spectrum,
                                    lsst_filter=self.lsst_filter,
                                    cosmo=self.cosmo,
                                    emline_template=self.emline_template.copy(),
                                    galaxy_template=self.galaxy_template.copy(),
                                    reddening_curve=self.reddening_curve.copy(),
                                    min_magnitude=self.min_magnitude, 
                                    max_magnitude=self.max_magnitude, 
                                    reference_band=self.reference_band,
                                    cadence=self.cadence,
                                    augment=self.augment, 
                                    save_true_LC=self.save_true_LC,
                                    custom_driving_signal=self.custom_driving_signal,
                                    use_LSST_cadence=self.use_LSST_cadence,
                                    )
            if sample is not None:
                flag = False

        return sample

class Generate_Test_LC():
    """
    Class to generate test light curves with custom driving signals
    """
    def __init__(self, kernel_num_days, kernel_resolution, seed_list, cadence, mag_mean, model_spectrum, cosmo, emline_template, galaxy_template, reddening_curve, min_magnitude, max_magnitude, reference_band, augment=False,save_true_LC=False):
        """
        kernel_num_days: int, number of days in the transfer function kernels
        kernel_resolution: float, resolution of the transfer function kernels
        seed_list: list, list of seeds for the random number generator
        cadence: float, time spacing in the light curve
        mag_mean: float, mean magnitude of the driving signal
        model_spectrum: bool, whether to model the spectrum to get the mean magnitude of each band, otherwise just select at random. Should be True except for testing.
        cosmo: astropy cosmology object, used to get the distance modulus for the redshift
        emline_template: numpy array, template for the emission lines
        galaxy_template: numpy array, template for the galaxy
        reddening_curve: numpy array, template for the reddening curve
        min_magnitude: float, minimum magnitude observed in the light curve
        max_magnitude: float, maximum magnitude observed in the light curve
        reference_band: int, reference band for the driving signal (0-5 for LSSt bands). Generally set to the the i-band (3).
        augment: bool, whether to augment the data set or use a fixed seed (from seed)
        save_true_LC: bool, whether to save the true light curve for validation
        """
        self.kernel_num_days = kernel_num_days
        self.kernel_resolution = kernel_resolution
        self.seed_list = seed_list
        self.augment = augment
        self.cadence = cadence
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.reference_band = reference_band
        self.model_spectrum = model_spectrum
        self.save_true_LC = save_true_LC

        if model_spectrum:
            # Choose the LSST filter from scpeclite
            self.lsst_filter = filters.load_filters('lsst2016-*')
            self.cosmo = cosmo
            self.emline_template = emline_template
            self.galaxy_template = galaxy_template
            self.reddening_curve = reddening_curve
    
    def sample_LC(self, custom_driving_signal, seed=None):
        """
        Function to generate a single light curve with a custom driving signal

        custom_driving_signal: function, function to generate the driving signal
        seed: int, seed for the random number generator
        """
        if seed is None:
            seed = np.random.choice(self.seed_list)
        sample = build_data_set(
                                kernel_num_days=self.kernel_num_days, 
                                kernel_resolution=self.kernel_resolution, 
                                seed=seed, 
                                model_spectrum=self.model_spectrum,
                                lsst_filter=self.lsst_filter,
                                cosmo=self.cosmo,
                                emline_template=self.emline_template.copy(),
                                galaxy_template=self.galaxy_template.copy(),
                                reddening_curve=self.reddening_curve.copy(),
                                min_magnitude=self.min_magnitude, 
                                max_magnitude=self.max_magnitude, 
                                reference_band=self.reference_band,
                                cadence=self.cadence,
                                augment=self.augment, 
                                save_true_LC=self.save_true_LC,
                                custom_driving_signal=custom_driving_signal,
                                )

        return sample

def plot_time_delay_corner(mean_time_pred, mean_time_true, L_time_delay, mean_mask, num, epoch, kernel_num_days, kernel_resolution, num_samples=1e6, relative_mean_time=True, reference_band=3, extra_folder=""):
    """
    Makes a corner plot of the time delay distribution

    mean_time_pred: predicted mean time delay
    mean_time_true: true mean time delay
    L_time_delay: lower triangle matrix of coveriance Sigma = L*L^T
    num: number of the plot
    epoch: epoch number
    kernel_num_days: number of days in the transfer function kernels
    kernel_resolution: resolution of the transfer function kernels
    num_sampled: number of samples to draw from the time delay distribution
    relative_mean_time: bool, whether to plot the time delay differences between bands or the time delays themselves
    reference_band: int, reference band for the time delay differences
    extra_folder: str, extra folder to save the plot in
    """

    observed_bands = np.where(mean_mask)[0]

    var_name_list = []
    var_name_list_no_units = []
    truth_list = []   
    if relative_mean_time:
        N_mean_band = len(bandpasses)-1

        for band_i in range(len(bandpasses)):

            if band_i != reference_band:
                var_name_list.append(rf"$(\bar{{\tau}}_{{{bandpasses[band_i]}}}-\bar{{\tau}}_{{{bandpasses[reference_band]}}})/\mathrm{{day}}$")
                var_name_list_no_units.append(rf"$(\bar{{\tau}}_{{{bandpasses[band_i]}}}-\bar{{\tau}}_{{{bandpasses[reference_band]}}})$")

    else:
        N_mean_band = len(bandpasses)
        for i in range(N_mean_band):

            var_name_list.append(rf"$\bar{{\tau}}_{{{bandpasses[i]}}}/\mathrm{{day}}$")
            var_name_list_no_units.append(rf"$\bar{{\tau}}_{{{bandpasses[i]}}}$")

    for plot_num in range(2):

        if plot_num == 0:
            # This uses the residuals between our prediced and true time delay
            sample = np.random.multivariate_normal(mean_time_pred, np.matmul(L_time_delay,L_time_delay.T), int(num_samples))
            sample = mean_time_true - sample
            truth_list = np.zeros(len(mean_time_true))
            time_delay_range = mean_time_true.max()-mean_time_true.min()
            min_val = -max(time_delay_range, cadence)
            max_val = max(time_delay_range, cadence)

        elif plot_num == 1:
            # Plot the predicted time delay distribution
            sample = np.random.multivariate_normal(mean_time_pred, np.matmul(L_time_delay,L_time_delay.T), int(num_samples))
            truth_list = np.copy(mean_time_true)
            time_delay_range = mean_time_true.max()-mean_time_true.min()
            min_val = -1.25*max(time_delay_range, cadence)
            max_val = 1.25*max(time_delay_range, cadence)

        figure = corner.corner(sample,smooth=True,labels=var_name_list,truths=truth_list,
                                label_kwargs={'fontsize':14},plot_datapoints=False,quantiles=[0.159, 0.5, 0.841],
                                show_titles=True,titles=var_name_list_no_units,title_kwargs={"fontsize": 11},
                                levels=(0.683,0.955,0.997))

        axes = np.array(figure.axes).reshape((N_mean_band, N_mean_band))
        
        # Loop over the diagonal
        for i in range(N_mean_band):

            ax = axes[i, i]
            
            ax.set_xlim(min_val,max_val)
            
            ax.minorticks_on()
            #ax.xaxis.set_major_locator(plt.LinearLocator(3))
            ax.tick_params(which='major',direction='in',top=False, right=True,length=tick_length_major,width=tick_width)
            ax.tick_params(which='minor',direction='in',top=False, right=True,length=tick_length_minor,width=tick_width)


        # Loop over the histograms
        for i in range(N_mean_band):
            for j in range(i):
                ax = axes[i, j]
                ax.set_xlim(min_val,max_val)
                ax.set_ylim(min_val,max_val)
                ax.minorticks_on()
                ax.tick_params(which='major',direction='inout',top=True, right=True,length=10,width=tick_width)
                ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)

        figure.subplots_adjust(wspace=0, hspace=0)
        if plot_num == 0:
            plt.savefig(f'{save_path}/recovery/{extra_folder}Corner_plot_time_delay_difference_num_{num}.{save_file_format}',bbox_inches='tight')
        elif plot_num == 1:
            plt.savefig(f'{save_path}/recovery/{extra_folder}Corner_plot_time_delay_num_{num}.{save_file_format}',bbox_inches='tight')
        plt.close()

def plot_driving_recovery_new(driving, driving_save, num, extra_folder=""):
    """
    Plots the predicted vs true driving variability signal.

    driving: numpy array, true driving signal
    driving_save: numpy array, predicted driving signal
    num: int, number of the plot
    extra_folder: str, extra folder to save the plot in
    """

    plt.figure(figsize=(12, 4))

    time = np.arange(driving_save.shape[0])*cadence
    plt.plot(time, driving_save)
    plt.xlim(time[0],time[-1])
    plt.gca().invert_yaxis()
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    plt.xlabel(f'time [days]')
    plt.ylabel('driving mag')
    plt.savefig(f'{save_path}/recovery/{extra_folder}driving_signal_recovery_{num}.{save_file_format}', bbox_inches='tight')
    plt.close()


def plot_driving_recovery(driving, driving_save, num, kernel_num_days, include_kernel_size=True, extra_folder=""):
    """
    Plots the predicted vs true driving variability signal.

    driving: numpy array, true driving signal
    driving_save: numpy array, predicted driving signal
    num: int, number of the plot
    kernel_num_days: int, number of days in the transfer function kernels
    include_kernel_size: bool, whether to include the kernel size in the plot
    extra_folder: str, extra folder to save the plot in
    """

    kernel_time_steps = int(kernel_num_days/cadence)

    plt.figure(figsize=(12, 4))
    if include_kernel_size:
        time = np.arange(driving.shape[0])*cadence
        #plt.plot(time,driving_true,label='driving true')
        plt.plot(time, driving, label = 'driving pred.')
        # gray out the first kernel_num_days time slightly
        plt.axvspan(0, kernel_num_days, alpha=0.125, color='gray')
    else:
        time = np.arange(len(driving_save)-kernel_time_steps)*cadence
        plt.plot(time, driving_save[kernel_time_steps:], label = 'driving true')
    plt.xlim(time[0],time[-1])
    plt.gca().invert_yaxis()
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    plt.xlabel(f'time [days]')
    plt.ylabel('driving mag')
    plt.savefig(f'{save_path}/recovery/{extra_folder}driving_signal_recovery_inc_kernel_{include_kernel_size}_{num}.{save_file_format}',bbox_inches='tight')
    plt.close()

    # Now the same thing but also including the true dirivng signal for comparison
    plt.figure(figsize=(12, 4))
    if include_kernel_size:
        time = np.arange(driving_save.shape[0])*cadence
        plt.plot(time, driving_save, label = 'driving true')
        plt.plot(time, driving, label = 'driving pred.', linestyle='--')
        # gray out the first kernel_num_days time slightly
        plt.axvspan(0, kernel_num_days, alpha=0.125, color='gray')
    else:
        time = np.arange(len(driving_save)-kernel_time_steps)*cadence
        plt.plot(time, driving_save[kernel_time_steps:], label = 'driving true')
        plt.plot(time, driving[kernel_time_steps:], label = 'driving pred.', linestyle='--')
    plt.xlim(time[0],time[-1])
    plt.gca().invert_yaxis()
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    plt.xlabel(f'time [days]')
    plt.ylabel('driving mag')
    plt.legend()
    plt.savefig(f'{save_path}/recovery/{extra_folder}driving_signal_recovery_with_true_inc_kernel_{include_kernel_size}_{num}.{save_file_format}',bbox_inches='tight')
    plt.close()

    # Now the same thing but also including the true dirivng signal for comparison and normalizing the driving signals by the median and median absolute deviation
    driving_norm = (driving - np.median(driving[kernel_time_steps:]))
    driving_norm = driving_norm/np.median(np.abs(driving_norm[kernel_time_steps:]))
    driving_save_norm = (driving_save - np.median(driving_save[kernel_time_steps:]))
    driving_save_norm = driving_save_norm/np.median(np.abs(driving_save_norm[kernel_time_steps:]))
    plt.figure(figsize=(12, 4))
    if include_kernel_size:
        time = np.arange(driving_save.shape[0])*cadence
        plt.plot(time,driving_save_norm, label = 'driving true')
        plt.plot(time,driving_norm, label = 'driving pred.', linestyle='--')
        # gray out the first kernel_num_days time slightly
        plt.axvspan(0, kernel_num_days, alpha=0.125, color='gray')
    else:
        time = np.arange(len(driving_save)-kernel_time_steps)*cadence
        plt.plot(time,driving_save_norm[kernel_time_steps:], label = 'driving true')
        plt.plot(time,driving_norm[kernel_time_steps:], label = 'driving pred.', linestyle='--') 
    plt.xlim(time[0],time[-1])
    plt.gca().invert_yaxis()
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    plt.xlabel(f'time [days]')
    plt.ylabel('relative brightness')
    plt.legend()
    plt.savefig(f'{save_path}/recovery/{extra_folder}driving_signal_recovery_with_true_norm_inc_kernel_{include_kernel_size}_{num}.{save_file_format}',bbox_inches='tight')
    plt.close()

def plot_driving_recovery_iterative(driving_list, driving_save, num, kernel_num_days, include_kernel_size=True, extra_folder=""):
    """
    Plots the predicted vs true driving variability signal.
    
    driving_list: numpy array, list of predicted driving signals of shape (num_iter, N)
    driving_save: numpy array, true driving signal
    num: int, number of the plot
    kernel_num_days: int, number of days in the transfer function kernels
    include_kernel_size: bool, whether to include the kernel size in the plot
    extra_folder: str, extra folder to save the plot in
    """
    kernel_time_steps = int(kernel_num_days/cadence)
    
    # Now the same thing but also including the true dirivng signal for comparison
    plt.figure(figsize=(12, 4))
    if include_kernel_size:
        time = np.arange(driving_save.shape[0])*cadence
        plt.plot(time, driving_save, label = 'truth', color='black')
        for i in range(driving_list.shape[0]):
            plt.plot(time, driving_list[i], label = f'iter {i+1}', linestyle='--')
        # gray out the first kernel_num_days time slightly
        plt.axvspan(0, kernel_num_days, alpha=0.125, color='gray')
    else:
        time = np.arange(len(driving_save)-kernel_time_steps)*cadence
        plt.plot(time, driving_save[kernel_time_steps:], label = 'truth', color='black')
        for i in range(driving_list.shape[0]):
            plt.plot(time, driving_list[i][kernel_time_steps:], label = f'iter {i+1}', linestyle='--')
    plt.xlim(time[0],time[-1])
    plt.gca().invert_yaxis()
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    plt.xlabel(f'time [days]')
    plt.ylabel('driving mag')
    plt.legend()
    plt.savefig(f'{save_path}/recovery/{extra_folder}driving_signal_recovery_iterative_with_true_inc_kernel_{include_kernel_size}_{num}.{save_file_format}',bbox_inches='tight')
    plt.close()

    # Now the same thing but also including the true dirivng signal for comparison and normalizing the driving signals by the median and median absolute deviation
    driving_save_norm = (driving_save - np.median(driving_save[kernel_time_steps:]))
    driving_save_norm = driving_save_norm/np.median(np.abs(driving_save_norm[kernel_time_steps:]))
    
    plt.figure(figsize=(12, 4))
    if include_kernel_size:
        plt.plot(time,driving_save_norm, label = 'truth', color='black')
        for i in range(driving_list.shape[0]):
            driving_norm = (driving_list[i] - np.median(driving_list[i][kernel_time_steps:]))
            driving_norm = driving_norm/np.median(np.abs(driving_norm[kernel_time_steps:]))
            plt.plot(time,driving_norm, label = f'iter {i+1}', linestyle='--')
        # gray out the first kernel_num_days time slightly
        plt.axvspan(0, kernel_num_days, alpha=0.125, color='gray')
    else:
        time = np.arange(len(driving_save)-kernel_time_steps)*cadence
        plt.plot(time,driving_save_norm[kernel_time_steps:], label = 'truth', color='black')
        for i in range(driving_list.shape[0]):
            driving_norm = (driving_list[i] - np.median(driving_list[i][kernel_time_steps:]))
            driving_norm = driving_norm/np.median(np.abs(driving_norm[kernel_time_steps:]))
            plt.plot(time,driving_norm[kernel_time_steps:], label = f'iter {i+1}', linestyle='--')
    plt.xlim(time[0],time[-1])
    plt.gca().invert_yaxis()
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    plt.xlabel(f'time [days]')
    plt.ylabel('relative brightness')
    plt.legend()
    plt.savefig(f'{save_path}/recovery/{extra_folder}driving_signal_recovery_iterative_with_true_norm_inc_kernel_{include_kernel_size}_{num}.{save_file_format}',bbox_inches='tight')
    plt.close()

def plot_coverage_probability(all_quantiles, coverage_prob_recovery, coverage_prob_recovery_GPR, epoch):
    """
    Plots the coverage probability of the recovery to evaluate the uncertainty estimates.

    all_quantiles: list, list of quantiles to evaluate
    coverage_prob_recovery: dictionary, dictionary of coverage probabilities for the neural network
    coverage_prob_recovery_GPR: dictionary, dictionary of coverage probabilities for the Gaussian process regression
    epoch: int, epoch number to save the plot
    """
    # convert dictionary to list
    coverage_prob_recovery_list = []
    for quantile in all_quantiles:
        coverage_prob_recovery_list.append(coverage_prob_recovery[quantile])

    coverage_prob_recovery_GPR_list = []
    if len(coverage_prob_recovery_GPR) > 0:
        for quantile in all_quantiles:
            coverage_prob_recovery_GPR_list.append(coverage_prob_recovery_GPR[quantile])


    # plot dots and connect them
    plt.plot(all_quantiles, coverage_prob_recovery_list, linestyle='-', label="NN")
    # plot the ideal case
    plt.plot(all_quantiles, all_quantiles, linestyle='--', color='black')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('fraction of truth in probability volume',fontsize=13)
    plt.xlabel('fraction of posterior probability volume',fontsize=13)
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    plt.savefig(f'{save_path}/coverage_prob_recovery_epoch{epoch}.pdf', bbox_inches='tight')
    plt.close()

    # plot dots and connect them
    plt.plot(all_quantiles, coverage_prob_recovery_list, linestyle='-', label="NN")
    if len(coverage_prob_recovery_GPR_list) > 0:
        plt.plot(all_quantiles, coverage_prob_recovery_GPR_list, linestyle='-', label="GPR")
    # plot the ideal case
    plt.plot(all_quantiles, all_quantiles, linestyle='--', color='black')
    if len(coverage_prob_recovery_GPR_list) > 0:
        plt.legend(fontsize=14)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('fraction of truth in probability volume',fontsize=13)
    plt.xlabel('fraction of posterior probability volume',fontsize=13)
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    plt.savefig(f'{save_path}/coverage_prob_recovery_with_GPR_epoch{epoch}.pdf', bbox_inches='tight')
    plt.close()

def plot_mean_versus_truth(mean_values,true_values,num_bins=20,log_mass_limit=None, name_extra=''):
    """
    This function plots the confusion matrices for the test set.

    mean_values_dict: dictionary of mean values for each parameter
    test_label_dict: dictionary of labels for each parameter
    num_bins: number of bins to use for confusion matrix
    log_mass_limit: float, limit for the log_mass to plot, None uses all values
    name_extra: str, extra name for the plot
    """
    num_parameters = len(parameters_keys)

    max_cols = 3
    num_cols = min(num_parameters,max_cols)
    num_rows = max(int(np.ceil(num_parameters/max_cols)),1)

    fig, axes = plt.subplots(num_rows, num_cols,figsize=(4*num_cols,3*num_rows))

    mean_values_dict = dict()
    test_label_dict = dict()

    for i in range(len(parameters_keys)):
        mean_values_dict[parameters_keys[i]] = mean_values[:,i]
        test_label_dict[parameters_keys[i]] = true_values[:,i]

    # create new mean_values_dict with only values above log_mass_limit
    if log_mass_limit is not None:
        new_mean_values_dict = dict()
        new_test_label_dict = dict()
        for i in range(len(mean_values_dict['log_mass'])):
            mass_val = (min_max_dict['log_mass'][1]-min_max_dict['log_mass'][0])*test_label_dict['log_mass'][i]+min_max_dict['log_mass'][0]
            if mass_val > log_mass_limit:
                for key in mean_values_dict.keys():
                    if key not in new_mean_values_dict.keys():
                        new_mean_values_dict[key] = []
                        new_test_label_dict[key] = []
                    new_mean_values_dict[key].append(mean_values_dict[key][i])
                    new_test_label_dict[key].append(test_label_dict[key][i])
        # make numpy arrays
        for key in new_mean_values_dict.keys():
            new_mean_values_dict[key] = np.array(new_mean_values_dict[key])
            new_test_label_dict[key] = np.array(new_test_label_dict[key])
        new_min_max_dict = dict()
        for key in min_max_dict.keys():
            if key not in new_min_max_dict.keys():
                new_min_max_dict[key] = []
            if key == 'log_mass':
                new_min_max_dict[key].append(log_mass_limit)
            else:
                new_min_max_dict[key].append(min_max_dict[key][0])
            new_min_max_dict[key].append(min_max_dict[key][1])
    else:
        new_min_max_dict = min_max_dict
        new_mean_values_dict = mean_values_dict
        new_test_label_dict = test_label_dict

    i=0
    j=0
    for k in range(num_cols*num_rows):
        if num_rows > 1:
            ax = axes[i,j]
        else:
            ax = axes[j]
        if k < num_parameters:
            par = parameters_keys[k]

            mean_val = (min_max_dict[par][1]-min_max_dict[par][0])*new_mean_values_dict[par]+min_max_dict[par][0]
            truth_val = (min_max_dict[par][1]-min_max_dict[par][0])*new_test_label_dict[par]+min_max_dict[par][0]

            ax.plot(np.linspace(min_max_dict[par][0],min_max_dict[par][1],100),np.linspace(min_max_dict[par][0],min_max_dict[par][1],100),color='black',linestyle='--')   

            im = ax.hist2d(truth_val,mean_val,bins=(num_bins, num_bins),
                        range=[[min_max_dict[par][0],min_max_dict[par][1]],[min_max_dict[par][0], min_max_dict[par][1]]])
            if j == num_cols-1 or k == num_parameters-1:
                fig.colorbar(im[3], ax=ax, label='number of LCs')
            else:
                fig.colorbar(im[3], ax=ax)
            ax.set_ylabel(f"pred. {plotting_labels[par]}",fontsize=13)
            ax.set_xlabel(f"true {plotting_labels[par]}",fontsize=13)
            
            #ax.set_xlim(new_min_max_dict[par][0], new_min_max_dict[par][1])
            ax.set_xlim(min_max_dict[par][0], min_max_dict[par][1])
            ax.set_ylim(min_max_dict[par][0], min_max_dict[par][1])

            ax.minorticks_on()
            
            ax.xaxis.set_major_locator(plt.LinearLocator(3))
            ax.yaxis.set_major_locator(plt.LinearLocator(3))
            
            ax.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            #ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
            ax.set_aspect('equal', adjustable='box')
        else:
            axes[i,j].set_visible(False)

        j+=1
        if j >= max_cols:
            j %= max_cols
            i += 1

    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrix_together{name_extra}_mass_lim_{log_mass_limit}_{num_bins}_bins.{save_file_format}",bbox_inches='tight')
    plt.savefig(f"{save_path}/confusion_matrix_together{name_extra}_mass_lim_{log_mass_limit}_{num_bins}_bins.png",bbox_inches='tight',dpi=1000)
    plt.close()

def plot_scatter_plots(param_labels,eval_median,eval_lower_bound,eval_upper_bound,log_mass_limit=None,max_num_samples=None):
    """
    Plot a scatter plot of the predictions with the median and the 68% confidence interval vs the true values

    param_labels, the true values of the parameters (but normalized between 0 and 1), shape (num_samples,num_parameters)
    eval_median, the median of the predictions (but for the normalized parameters between 0 and 1), shape (num_samples,num_parameters)
    eval_lower_bound, the lower bound of the 68% confidence interval of the predictions (but for the normalized parameters between 0 and 1), shape (num_samples,num_parameters)
    eval_upper_bound, the upper bound of the 68% confidence interval of the predictions (but for the normalized parameters between 0 and 1), shape (num_samples,num_parameters)
    log_mass_limit, the log mass limit of the data, if None, no mass limit is applied
    max_num_samples, the maximum number of samples to plot, if None, all samples are plotted
    """
    num_parameters = len(parameters_keys)

    max_cols = 3
    num_cols = min(num_parameters,max_cols)
    num_rows = max(int(np.ceil(num_parameters/max_cols)),1)

    fig, axes = plt.subplots(num_rows, num_cols,figsize=(4*num_cols,3*num_rows))

    if log_mass_limit is not None:
        new_min_max_dict = dict()
        for key in min_max_dict.keys():
            if key not in new_min_max_dict.keys():
                new_min_max_dict[key] = []
            if key == 'log_mass':
                new_min_max_dict[key].append(log_mass_limit)
            else:
                new_min_max_dict[key].append(min_max_dict[key][0])
            new_min_max_dict[key].append(min_max_dict[key][1])
    else:
        new_min_max_dict = min_max_dict


    if log_mass_limit is not None:
        eval_median_new = []
        eval_lower_bound_new = []
        eval_upper_bound_new = []

        param_labels_new = []

        for i in range(len(eval_median)):
            mass_val = (min_max_dict['log_mass'][1]-min_max_dict['log_mass'][0])*param_labels[i,parameters_keys.index('log_mass')]+min_max_dict['log_mass'][0]
            if mass_val > log_mass_limit:
                eval_median_new.append(eval_median[i])
                eval_lower_bound_new.append(eval_lower_bound[i])
                eval_upper_bound_new.append(eval_upper_bound[i])
                param_labels_new.append(param_labels[i])
        # make numpy arrays
        eval_median_new = np.array(eval_median_new)
        eval_lower_bound_new = np.array(eval_lower_bound_new)
        eval_upper_bound_new = np.array(eval_upper_bound_new)
        param_labels_new = np.array(param_labels_new)

    else:
        eval_median_new = eval_median
        eval_lower_bound_new = eval_lower_bound
        eval_upper_bound_new = eval_upper_bound
        param_labels_new = param_labels

    if max_num_samples is not None:
        max_num_samples = min(max_num_samples,eval_median_new.shape[0])
        eval_median_new = eval_median_new[:max_num_samples]
        eval_lower_bound_new = eval_lower_bound_new[:max_num_samples]
        eval_upper_bound_new = eval_upper_bound_new[:max_num_samples]
        param_labels_new = param_labels_new[:max_num_samples]
    
    i=0
    j=0
    for k in range(num_cols*num_rows):
        if num_rows > 1:
            ax = axes[i,j]
        else:
            ax = axes[j]
        if k < num_parameters:
            par = parameters_keys[k]

            lower_error = eval_median_new[:,k] - eval_lower_bound_new[:,k]
            upper_error = eval_upper_bound_new[:,k] - eval_median_new[:,k]
            median_val = (min_max_dict[par][1]-min_max_dict[par][0])*eval_median_new[:,k]+min_max_dict[par][0]
            lower_error_val = (min_max_dict[par][1]-min_max_dict[par][0])*lower_error
            upper_error_val = (min_max_dict[par][1]-min_max_dict[par][0])*upper_error

            asymmetric_error = [lower_error_val, upper_error_val]

            truth_val = (min_max_dict[par][1]-min_max_dict[par][0])*param_labels_new[:,k]+min_max_dict[par][0]

            # get default color cycle
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            ax.plot(np.linspace(min_max_dict[par][0],min_max_dict[par][1],100),np.linspace(min_max_dict[par][0],min_max_dict[par][1],100),color='black',linestyle='--')   
            ax.errorbar(truth_val,median_val,yerr=asymmetric_error,fmt='o',markersize=2,elinewidth=0.5,capsize=1.5,capthick=0.5,color=colors[0])

            ax.set_ylabel(f"pred. {plotting_labels[par]}",fontsize=13)
            ax.set_xlabel(f"true {plotting_labels[par]}",fontsize=13)
            
            #ax.set_xlim(new_min_max_dict[par][0], new_min_max_dict[par][1])
            ax.set_xlim(min_max_dict[par][0], min_max_dict[par][1])
            ax.set_ylim(min_max_dict[par][0], min_max_dict[par][1])
            
            ax.minorticks_on()
            
            ax.xaxis.set_major_locator(plt.LinearLocator(3))
            ax.yaxis.set_major_locator(plt.LinearLocator(3))
            
            ax.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
            ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
            
            
            #FIX
            #ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
            ax.set_aspect('equal', adjustable='box')
        else:
            axes[i,j].set_visible(False)

        j+=1
        if j >= max_cols:
            j %= max_cols
            i += 1

    plt.tight_layout()
    plt.savefig(f"{save_path}/scatter_together_mass_lim_{log_mass_limit}_max_samples_{max_num_samples}.{save_file_format}",bbox_inches='tight')
    plt.close()

def plot_coverage_probability_params(all_quantiles, coverage_prob_parameters, epoch):
    """
    Plots the coverage probability of the parameters
    
    all_quantiles: list, list of quantiles to evaluate
    coverage_prob_parameters: dictionary, dictionary of coverage probabilities for the parameters
    epoch: int, epoch number to save the plot
    """
    # convert dictionary to list

    if len(parameters_keys) <= 10:
        # The default color cycle only has 10 colors, so we need to use a different color cycle if we have more than 10 parameters
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    else:
        #colors = plt.cm.tab20.colors
        colors = plt.cm.tab20(np.linspace(0,1,len(parameters_keys)))
    for i, param in enumerate(parameters_keys):
        coverage_prob_recovery_list = []
        for quantile in all_quantiles:
            coverage_prob_recovery_list.append(coverage_prob_parameters[param][quantile])
        # plot dots and connect them
        plt.plot(all_quantiles, coverage_prob_recovery_list, linestyle='-', label=plotting_labels_no_units[param], color=colors[i])
    # plot the ideal case
    plt.plot(all_quantiles, all_quantiles, linestyle='--', color='black')
    plt.legend(loc='upper left', fontsize=9, ncol=2)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('fraction of truth in probability volume',fontsize=13)
    plt.xlabel('fraction of posterior probability volume',fontsize=13)
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    plt.savefig(f'{save_path}/coverage_prob_parameters_epoch_{epoch}.pdf', bbox_inches='tight')
    plt.close() 

def plot_coverage_probability_params_and_recovery(all_quantiles, coverage_prob_parameters, coverage_prob_recovery, epoch):
    """
    Plots the coverage probability of the parameters and recovery

    all_quantiles: list, list of quantiles to evaluate
    coverage_prob_parameters: dictionary, dictionary of coverage probabilities for the parameters
    coverage_prob_recovery: dictionary, dictionary of coverage probabilities for the recovery
    epoch: int, epoch number to save the plot
    """
    # convert dictionary to list

    if len(parameters_keys)+1 <= 10:
        # The default color cycle only has 10 colors, so we need to use a different color cycle if we have more than 10 parameters
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    else:
        #colors = plt.cm.tab20.colors
        colors = plt.cm.tab20(np.linspace(0,1,len(parameters_keys)+1))

    coverage_prob_recovery_list = []
    for quantile in all_quantiles:
        coverage_prob_recovery_list.append(coverage_prob_recovery[quantile])
    plt.plot(all_quantiles, coverage_prob_recovery_list, linestyle='-', label='light curve', color=colors[0])

    for i, param in enumerate(parameters_keys):
        coverage_prob_recovery_list = []
        for quantile in all_quantiles:
            coverage_prob_recovery_list.append(coverage_prob_parameters[param][quantile])
        # plot dots and connect them
        plt.plot(all_quantiles, coverage_prob_recovery_list, linestyle='-', label=plotting_labels_no_units[param], color=colors[i+1])
    # plot the ideal case
    plt.plot(all_quantiles, all_quantiles, linestyle='--', color='black')
    plt.legend(loc='upper left', fontsize=9, ncol=2)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('fraction of truth in probability volume',fontsize=13)
    plt.xlabel('fraction of posterior probability volume',fontsize=13)
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
    plt.savefig(f'{save_path}/coverage_prob_recovery_and_parameters_epoch_{epoch}.pdf', bbox_inches='tight')
    plt.close()

def plot_corner_plot(sample, truth, num, epoch, extra_folder=""):
    """
    Makes a corner plot of the posterior distribution of the parameters.

    sample: posterior sample of the parameters
    truth: true values of the parameters
    num: int, number of the plot for saving
    epoch: int, epoch number for saving
    extra_folder: str, extra folder to save the plot in
    """
    
    var_name_list = []
    var_name_list_no_units = []
    truth_list = []   

    sample = np.copy(sample)

    for j,par in enumerate(parameters_keys):
        sample[:,j] = (min_max_dict[par][1]-min_max_dict[par][0])*sample[:,j]+min_max_dict[par][0]
        truth_val = (min_max_dict[par][1]-min_max_dict[par][0])*truth[j]+min_max_dict[par][0]

        truth_list.append(truth_val)
        var_name_list.append(plotting_labels[par])
        var_name_list_no_units.append(plotting_labels_no_units[par])

    figure = corner.corner(sample,smooth=True,labels=var_name_list,truths=truth_list, title_fmt='.3g', 
                          label_kwargs={'fontsize':15.5},plot_datapoints=False,quantiles=[0.159, 0.5, 0.841],
                          show_titles=True,titles=var_name_list_no_units,title_kwargs={"fontsize": 11.25},
                          levels=(0.683,0.955,0.997))

    axes = np.array(figure.axes).reshape((len(parameters_keys), len(parameters_keys)))
    # Loop over the diagonal
    for i,par in enumerate(parameters_keys):
        ax = axes[i, i]
        
        ax.set_xlim(min_max_dict[par][0],min_max_dict[par][1])
        
        ax.minorticks_on()
        #ax.xaxis.set_major_locator(plt.LinearLocator(3))
        ax.tick_params(which='major',direction='in',top=False, right=True,length=tick_length_major,width=tick_width)
        ax.tick_params(which='minor',direction='in',top=False, right=True,length=tick_length_minor,width=tick_width)


    # Loop over the histograms
    for i in range(len(parameters_keys)):
        for j in range(i):
            ax = axes[i, j]  
            ax.set_xlim(min_max_dict[parameters_keys[j]][0],min_max_dict[parameters_keys[j]][1])
            ax.set_ylim(min_max_dict[parameters_keys[i]][0],min_max_dict[parameters_keys[i]][1])
            ax.minorticks_on()
            ax.tick_params(which='major',direction='inout',top=True, right=True,length=10,width=tick_width)
            ax.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)

    figure.subplots_adjust(wspace=0, hspace=0)
    
    plt.savefig(f'{save_path}/recovery/{extra_folder}Corner_plot_{num}.{save_file_format}', bbox_inches='tight')
    plt.close()

def plot_metric_vs_iteration(metric, metric_name, epoch):
    """
    Plots the metric vs the iteration number to see how the training is going.

    metric: list, list of the metric values
    metric_name: str, name of the metric for the plot
    epoch: int, epoch number for saving
    """
    plt.figure(figsize=(5, 2))
    plt.plot(np.arange(len(metric))+1,metric,'o-')
    plt.xlabel('iteration')
    plt.ylabel(metric_name)
    # only put xticks on iterations
    plt.xticks(np.arange(len(metric))+1)
    plt.minorticks_on()
    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
    plt.tick_params(which='minor',direction='in',top=False,bottom=False,left=True,right=True,length=tick_length_minor,width=tick_width)
    plt.savefig(f'{save_path}/metric_vs_iteration_{metric_name}_{epoch}.{save_file_format}',bbox_inches='tight')
    plt.close()

def gaussian_process_regression(LC, LC_std, mean_mask, mean_across_bands=False, min_std=0.01):
    """
    Perform Gaussian process regression on the light curve to estimate the uncertainty.
    Uses the GPyTorch library with a Matern kernel with nu=0.5 and FixedNoiseMultiTaskGP model and ExactMarginalLogLikelihood loss function.

    LC = light curve to be normalized, shape (num_time_steps,num_bands)
    LC_std = standard deviation of the light curve, shape (num_time_steps,num_bands)
    mean_mask = mask of which bands are observed, shape (num_bands)
    mean_across_bands = whether to average the mean and std across bands or normalize each band separately
    min_std = minimum standard deviation to use for the GP in magnitude.
    """
    device = LC.device

    num_time_steps = LC.shape[0]

    mean_mask = (mean_mask.squeeze(0) != 0.0)


    LC = LC[:, mean_mask]
    LC_std = LC_std[:, mean_mask]

    #Mask out the zero values
    mask = (LC != 0.0).type_as(LC) 
    
    epsilon = 1e-6
    #get masked mean weighted by the photometric noise
    LC_mean_per_band = torch.sum(LC/(LC_std+epsilon)**2,dim=0)/torch.sum(mask/(LC_std+epsilon)**2,dim=0)

    #get masked std weighted by the photometric noise
    mean_diff = mask*(LC - LC_mean_per_band)**2
    LC_std_per_band = torch.sqrt(torch.sum(mean_diff/(LC_std+epsilon)**2,dim=0)/torch.sum(mask/(LC_std+epsilon)**2,dim=0))

    # get mean and std averaged over all bands
    if mean_across_bands:
        LC_mean = LC_mean_per_band.mean(dim=-1)
        LC_mean = LC_mean.unsqueeze(-1) #add a dimension to make it broadcastable
        LC_pred_std = LC_std_per_band.mean(dim=-1)
        LC_pred_std = LC_pred_std.unsqueeze(-1) #add a dimension to make it broadcastable
    else:
        LC_mean = LC_mean_per_band
        LC_pred_std = LC_std_per_band

    # Normalize the light curve to have zero mean and unit variance
    LC = mask*((LC - LC_mean)/LC_pred_std)
    LC_std = LC_std/LC_pred_std

    num_tasks = LC.shape[-1]

    x_list = []
    y_list = []
    y_var_list = []
    for i in range(num_tasks):
        x = LC[:,i].nonzero()[:,0]
        y = LC[x,i]
        
        y_std = LC_std[x,i] 

        x = x/(LC.shape[0]-1) #normalize the x values to be between 0 and 1
        x = x.type_as(LC)
        y_var_list.append(y_std**2)
        x = x[:,None] #add a dimension to make it a 2D tensor
        i = i*torch.ones(x.shape, dtype=x.dtype, device=x.device)
        x_list.append(torch.cat([x,i],-1))
        y_list.append(y)

    del LC, LC_std, mask, mean_diff, LC_mean_per_band, LC_std_per_band

    train_X = torch.cat(x_list)
    train_Y = torch.cat(y_list,-1).unsqueeze(-1)
    train_Yvar = torch.cat(y_var_list,-1).unsqueeze(-1)    
    
    train_X = train_X.to(device, dtype=torch.float64)
    train_Y = train_Y.to(device, dtype=torch.float64)
    train_Yvar = train_Yvar.to(device, dtype=torch.float64)

    #Define the model
    with torch.enable_grad(), gpytorch.settings.fast_computations(False, False, False):
        model = FixedNoiseMultiTaskGP(
            train_X, train_Y, train_Yvar, task_feature=-1,
            covar_module= ScaleKernel(base_kernel = MaternKernel(nu=0.5,ard_num_dims=train_X.shape[-1]))
            ).to(device)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll.train()

        #botorch.fit.fit_gpytorch_model(mll, options={"maxiter": 10000})
        fit_gpytorch_model(mll) 
    
    del train_X, train_Y, train_Yvar
    del x_list, y_list, y_var_list
    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad(), gpytorch.settings.fast_computations(False, False, False):
        model.eval()
  
        X_test_time = torch.linspace(0.0, 1.0, num_time_steps).unsqueeze(-1).repeat(1, num_tasks).view(-1, 1).to(device, dtype=torch.float64)
        X_test_task = torch.arange(num_tasks).unsqueeze(0).repeat(num_time_steps, 1).view(-1, 1).to(device, dtype=torch.float64)
        X_test = torch.cat([X_test_time, X_test_task], dim=-1)

        eval = model(X_test)
        mean = eval.mean.view(num_time_steps, num_tasks).detach().cpu().numpy()
        std = eval.variance.view(num_time_steps, num_tasks).sqrt().detach().cpu().numpy()

        del X_test, X_test_time, X_test_task, eval, model, mll

        # Unnormalize the results
        mean = mean * LC_pred_std.detach().cpu().numpy() + LC_mean.detach().cpu().numpy() 
        std = std * LC_pred_std.detach().cpu().numpy()

        #clip the std to avoid outliers when the std is very small. Minimum std is 0.01.
        std = np.clip(std, min_std, 100.0)  

    gc.collect()
    torch.cuda.empty_cache()

    # account for the mean mask
    mean_full = np.zeros((mean.shape[0], num_bands))
    std_full = np.zeros((std.shape[0], num_bands))
    mean_full[:, mean_mask.detach().cpu().numpy()] = mean
    std_full[:, mean_mask.detach().cpu().numpy()] = std

    # Concatenate the mean and std across the bands
    return np.concatenate([mean_full, std_full],axis=-1)

def save_model(model, optimizer, save_path):
    """
    Helper function to save the model, optimizer, and scheduler to the save_path directory.

    model: pytorch model to be saved
    optimizer: pytorch optimizer to be saved
    save_path: str, directory to save
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{save_path}/checkpoint.pth")
 
def load_model_function(model, optimizer, save_path, device, distributed):
    """
    Load the model from a previous run and return the model and optimizer. Use transfer learning if the model has a different number of parameters.

    model: the model to be loaded
    optimizer: the optimizer to be loaded
    save_path: the path to the directory where the model is saved
    device: the device to use
    distributed: whether you are using distributed training
    """
    model_path = f"{save_path}/checkpoint.pth"

    try:
        if distributed:
            checkpoint = torch.load(model_path, map_location=model.device)
        else:
            checkpoint = torch.load(model_path, map_location=device)
    except:
        print("Could not load the model. Returning the original model.")
        return model, optimizer
    
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # use transfer learning since we modified the model with keys for parameters of different sizes
    same_shape = True
    old_model_state_dict = checkpoint['model_state_dict']
   
    model_state_dict = model.state_dict()
    for key in old_model_state_dict.keys():
        if key in model_state_dict.keys():
            if old_model_state_dict[key].shape == model_state_dict[key].shape:
                model_state_dict[key] = old_model_state_dict[key]
            else:
                same_shape = False
        elif key.replace('module.','') in model_state_dict.keys():
            if old_model_state_dict[key].shape == model_state_dict[key.replace('module.','')].shape:
                model_state_dict[key.replace('module.','')] = old_model_state_dict[key]
            else:
                same_shape = False
        else:
            same_shape = False

    print(f"Same shape: {same_shape}")
    model.load_state_dict(model_state_dict, strict=False)

    load_optimizer = True
    if load_optimizer:
        try:
            if same_shape:
                # Don't load the learning rate scheduler
                learning_rate = optimizer.param_groups[0]['lr']
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Use the new learning rate not the one from the checkpoint
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
            else:
                print("Could not load the optimizer because the model might have a different number of parameters")
        except:
            print("Could not load the optimizer because the model might have a different number of parameters")
    else:
        print("Not loading the optimizer since load_optimizer is set to False")

    return model, optimizer

# Didn't end up using this function
class CustomLRScheduler:
    """
    A custom learning rate scheduler that ramps up the learning rate for a certain number of epochs and then decays it exponentially.
    """
    def __init__(self, optimizer, ramp_up_epochs, ramp_up_start_lr, ramp_up_end_lr, decay_rate):
        """
        optimizer: the PyTorch optimizer to be used
        ramp_up_epochs: the number of epochs to ramp up the learning rate
        ramp_up_start_lr: the starting learning rate for the ramp up
        ramp_up_end_lr: the ending learning rate for the ramp up
        decay_rate: the decay rate for the exponential decay
        """
        self.optimizer = optimizer
        self.ramp_up_epochs = ramp_up_epochs
        self.ramp_up_start_lr = ramp_up_start_lr
        self.ramp_up_end_lr = ramp_up_end_lr
        self.decay_rate = decay_rate
        self.epoch = 0

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = ramp_up_start_lr # Start the learning rate at ramp_up_start_lr

    def step(self):
        if self.epoch < self.ramp_up_epochs:
            lr = self.ramp_up_start_lr + (self.ramp_up_end_lr - self.ramp_up_start_lr) * (self.epoch / self.ramp_up_epochs)
        else:
            lr = self.ramp_up_end_lr * (self.decay_rate ** (self.epoch - self.ramp_up_epochs))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.epoch += 1

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']

def benchmark_transfer_function_models(device, batch_size, num_batch, min_max_dict, parameters_keys, lambda_effective_Angstrom, kernel_num_days, kernel_resolution, cosmo):
    """
    Helper function to benchmark the transfer function model generation time on the CPU and GPU with different batch sizes.
    
    device: the device to use
    batch_size: the batch size to use
    num_batch: the number of batches to run
    min_max_dict: the dictionary of the minimum and maximum values for each parameter
    parameters_keys: the keys of the parameters
    lambda_effective_Angstrom: the effective wavelength to evaluate the transfer functions at
    kernel_num_days: the number of days in the kernel
    kernel_resolution: the resolution of the kernel in days
    cosmo: astropy cosmology object to use
    """
    # Draw random parameters
    log_mass = torch.rand(batch_size, device=device) * (min_max_dict['log_mass'][1] - min_max_dict['log_mass'][0]) + min_max_dict['log_mass'][0]
    log_edd = torch.rand(batch_size, device=device) * (min_max_dict['log_edd'][1] - min_max_dict['log_edd'][0]) + min_max_dict['log_edd'][0]
    f_lamp = torch.rand(batch_size, device=device) * (min_max_dict['f_lamp'][1] - min_max_dict['f_lamp'][0]) + min_max_dict['f_lamp'][0]
    height = torch.rand(batch_size, device=device) * (min_max_dict['height'][1] - min_max_dict['height'][0]) + min_max_dict['height'][0]
    theta_inc = torch.rand(batch_size, device=device) * (min_max_dict['theta_inc'][1] - min_max_dict['theta_inc'][0]) + min_max_dict['theta_inc'][0]
    beta = torch.rand(batch_size, device=device) * (min_max_dict['beta'][1] - min_max_dict['beta'][0]) + min_max_dict['beta'][0]
    spin = torch.rand(batch_size, device=device) * (min_max_dict['spin'][1] - min_max_dict['spin'][0]) + min_max_dict['spin'][0]
    redshift = torch.rand(batch_size, device=device) * (min_max_dict['redshift'][1] - min_max_dict['redshift'][0]) + min_max_dict['redshift'][0]

    # Put the parameters in a pytorch tensor
    params = torch.zeros(batch_size, len(parameters_keys)).to(device)
    params[:,0] = spin
    params[:,1] = log_edd
    params[:,2] = f_lamp
    params[:,3] = height
    params[:,4] = theta_inc
    params[:,5] = redshift
    params[:,6] = beta
    params[:,7] = log_mass

    params.requires_grad = False
    lamb = torch.tensor(lambda_effective_Angstrom, device=device)

    with torch.no_grad():
        start_time = time.time()
        for i in range(num_batch):
            _ = generate_tf(params,
                        lamb,
                        kernel_num_days=kernel_num_days,
                        kernel_resolution=kernel_resolution,
                        parameters_keys=parameters_keys,
                        GR=True,
                        plot=False)
        elapsed_time_GPU_no_grad = time.time() - start_time

    with torch.no_grad():
        start_time = time.time()
        for i in range(num_batch):
            _ = generate_tf(params.to(torch.device('cpu')),
                            lamb.to(torch.device('cpu')),
                            kernel_num_days=kernel_num_days,
                            kernel_resolution=kernel_resolution,
                            parameters_keys=parameters_keys,
                            GR=True,
                            plot=False)
        elapsed_time_cpu_no_grad = time.time() - start_time

    start_time = time.time()
    params.requires_grad = True
    for i in range(num_batch):
        _ = generate_tf(params,
                    lamb,
                    kernel_num_days=kernel_num_days,
                    kernel_resolution=kernel_resolution,
                    parameters_keys=parameters_keys,
                    GR=True,
                    plot=False)
    elapsed_time_GPU_with_grad = time.time() - start_time
    
    start_time = time.time()
    for i in range(num_batch):
        _ = generate_tf(params.to(torch.device('cpu')),
                        lamb.to(torch.device('cpu')),
                        kernel_num_days=kernel_num_days,
                        kernel_resolution=kernel_resolution,
                        parameters_keys=parameters_keys,
                        GR=True,
                        plot=False)
        
    elapsed_time_cpu_with_grad = time.time() - start_time

    params = params.detach().cpu().numpy()
    lamb = lamb.detach().cpu().numpy()

    start_time = time.time()
    for i in range(num_batch):
        for j in range(batch_size):
            _ = generate_tf_numpy(params[j],
                                lamb,
                                kernel_num_days=kernel_num_days,
                                kernel_resolution=kernel_resolution,
                                parameters_keys=parameters_keys,
                                cosmo=cosmo,
                                GR=True,
                                just_get_spectrum=False,
                                plot=False) 
            
    elapsed_time_cpu = time.time() - start_time

    return elapsed_time_GPU_no_grad, elapsed_time_cpu_no_grad, elapsed_time_GPU_with_grad, elapsed_time_cpu_with_grad, elapsed_time_cpu

def main(
    batch_size=27,                              # batch size for training
    batch_size_inference=50,                    # batch size for inference, can be longer than the training batch size since we don't need to backpropagate so save memory
    driving_latent_size=16,                     # Latent size of the driving signal latent SDE.
    context_size=128,                           # Context size in encoder.
    hidden_size=256,                            # Hidden size of NNs
    num_iterations=1,                           # Number of iterations of the RIM
    num_layers=5,                               # Number of layers in the transformer
    num_heads=8,                                # Number of heads in the transformer
    kernel_num_days=800,                        # Number of days in the kernel
    kernel_resolution=cadence,                  # Resolution of the kernel in days
    dt=5.0e-4,                                  # This is the integration time step with t-axis in the range of 0 to 1. Probably want to get this to 1e-3 but it takes a long time to train.
    KL_anneal_epochs=0,                         # Number of epochs to anneal the KL divergence
    param_anneal_epochs=0,                      # Number of epochs to anneal the parameter loss
    param_loss_weight=2.0,                      # Weight of the parameter loss
    log_pxs_weight=1.0,                         # Weight of the log_pxs loss
    log_pxs2_weight=1.0,                        # Weight of the log_pxs2 loss
    log_pxs2_leeway=0.005,                      # Leeway for the log_pxs2 loss, in magnitude
    time_delay_loss_weight=1.0,                 # Weight of the time delay loss
    num_Gaussian_parameterization=5,            # Number of Gaussian parameterizations
    relative_mean_time=True,                    # If true, use the relative time delay loss instead of the absolute time delay loss
    reference_band=3,                           # Reference band for the time delay loss. Only relevent when relative_mean_time is True.
    lr_init=8.0e-4,                             # Initial learning rate. 
    weight_decay=1e-8,                          # Weight decay, L2 regularization on the trainable parameters
    AMP=False,                                  # If true, use automatic mixed precision, does not seem to work currently due to numerical instability
    use_GPU=False,                              # If true, use GPU if available
    batch_fraction_sim=1.0,                     # Fraction of the batch used at a time for the transfer function simulation. This is to save GPU memory if needed.
    lr_gamma=0.97,                              # learning rate decay factor. Probably want to change depending on how many epochs we are training for.
    ramp_up_epochs=0,                           # Number of epochs to ramp up the learning rate
    rampup_factor=5.0,                          # Factor to ramp up the learning rate
    num_epoch=30,                               # number of epochs to train! CHANGE TO DESIRED NUMBER.
    method="euler",                             # SDE solver method, either "euler", "milstein", or "srk".
    logqp=False,                                # If true, use the logqp loss instead of the KL divergence
    mag_mean=mag_mean,                          # This is the mean of the magnitude. We subtract this off to unnormalize the data.
    min_magnitude=min_magnitude,                # This is the minimum magnitude we are able to observe.
    max_magnitude=max_magnitude,                # This is the maximum magnitude we are able to observe.
    model_spectrum=model_spectrum,              # If true then we model the quasar spectrum by using a template and our model's continuum.
    cosmo=cosmo,                                # Cosmology used to calculate the luminosity distance, Astropy cosmology object
    dropout_rate=0.0,                           # Dropout rate
    grad_clip_value=250.0,                      # Gradient clipping value
    num_plot=25,                                # Number of LC to plot during training
    compare_to_GPR=False,                       # If true, compare the latent SDE model to Gaussian process regression
    load_model=False,                           # If true, load the latent SDE model from a previous run. Do not train further.
    load_path="results",                        # Path to the directory where the model is saved
    skip_training_first_epoch=False,            # If true, skip training the first epoch, only if load_model is True
    benchmark_TF_models=True,                   # If true, benchmark the transfer function models on the CPU and GPU to compare the speed
    custom_driving_signal_test=None,            # None, custom_driving_signal_sine, custom_driving_signal_DRW, custom_driving_signal_sine_with_BPL, custom_driving_signal_two_sine, custom_driving_signal_sawtooth, custom_driving_signal_square_wave
    use_LSST_cadence_test=0,                    # 0, 1, 2 where 0 is LSST cadences, 1 is high cadence, 2 is high cadence with sections
    max_cpus=100,                               # Maximum number of CPUs to use during training and data generation
):
    assert type(batch_size) == int and batch_size > 0, "batch_size must be a positive integer"
    assert type(batch_size_inference) == int and batch_size_inference > 0, "batch_size_inference must be a positive integer"
    assert type(driving_latent_size) == int and driving_latent_size > 0, "driving_latent_size must be a positive integer"
    assert type(context_size) == int and context_size > 0, "context_size must be a positive integer"
    assert type(hidden_size) == int and hidden_size > 0, "hidden_size must be a positive integer"
    assert type(num_iterations) == int and num_iterations > 0, "num_iterations must be a positive integer"
    assert type(num_layers) == int and num_layers > 0, "num_layers must be a positive integer"
    assert type(kernel_num_days) in (int, float) and kernel_num_days > 0, "kernel_num_days must be a positive integer or float"
    assert type(kernel_resolution) in (int, float) and kernel_resolution > 0, "kernel_resolution must be a positive integer or float"
    assert type(dt) == float and dt > 0, "dt must be a positive float"
    assert type(KL_anneal_epochs) == int and KL_anneal_epochs >= 0, "KL_anneal_epochs must be a positive integer"
    assert type(param_anneal_epochs) == int, "param_anneal_epochs must be an integer"
    assert type(param_loss_weight) == float and param_loss_weight >= 0, "param_loss_weight must be a positive float"
    assert type(log_pxs_weight) == float and log_pxs_weight >= 0, "log_pxs_weight must be a positive float"
    assert type(log_pxs2_weight) == float and log_pxs2_weight >= 0, "log_pxs2_weight must be a positive float"
    assert type(num_Gaussian_parameterization) == int and num_Gaussian_parameterization > 0, "num_Gaussian_parameterization must be a positive integer"
    assert type(time_delay_loss_weight) == float and time_delay_loss_weight >= 0, "time_delay_loss_weight must be a positive float"
    assert type(relative_mean_time) == bool, "relative_mean_time must be a boolean"
    assert type(lr_init) == float and lr_init > 0, "lr_init must be a positive float"
    assert type(weight_decay) == float and weight_decay >= 0, "weight_decay must be a positive float"
    assert type(AMP) == bool, "AMP must be a boolean"
    assert type(use_GPU) == bool, "use_GPU must be a boolean"
    assert type(batch_fraction_sim) == float and batch_fraction_sim > 0.0 and batch_fraction_sim <= 1.0, "batch_fraction_sim must be a positive float between 0 and 1"
    assert type(lr_gamma) == float and lr_gamma > 0.0, "lr_gamma must be a positive float"
    assert type(num_epoch) == int and num_epoch >= 0, "num_epoch must be a positive integer"
    assert method in ["euler", "milstein", "srk"], "method must be either 'euler', 'milstein', or 'srk'"
    assert type(logqp) == bool, "logqp must be a boolean"
    assert type(mag_mean) == float, "mag_mean must be a float"
    assert type(min_magnitude) == float, "min_magnitude must be a float"
    assert type(max_magnitude) == float, "max_magnitude must be a float"
    assert type(model_spectrum) == bool, "model_spectrum must be a boolean"
    assert type(dropout_rate) == float and dropout_rate >= 0.0 and dropout_rate < 1.0, "dropout_rate must be a float between 0 and 1"
    assert type(grad_clip_value) == float and grad_clip_value >= 0.0, "grad_clip_value must be a positive float"
    assert type(num_plot) == int and num_plot >= 0, "num_plot must be a positive integer"
    assert type(load_model) == bool, "load_model must be a boolean"
    assert type(load_path) == str and len(load_path) > 0, "load_path must be a non-empty string"
    assert type(skip_training_first_epoch) == bool, "skip_training_first_epoch must be a boolean"
    assert type(benchmark_TF_models) == bool, "benchmark_TF_models must be a boolean"
    assert custom_driving_signal_test in [None, "custom_driving_signal_sine", "custom_driving_signal_DRW", "custom_driving_signal_sine_with_BPL", "custom_driving_signal_two_sine", "custom_driving_signal_sawtooth", 
                                          "custom_driving_signal_square_wave"], "custom_driving_signal_test must be None or 'custom_driving_signal_sine', 'custom_driving_signal_DRW', 'custom_driving_signal_sine_with_BPL', \
                                        'custom_driving_signal_two_sine', 'custom_driving_signal_sawtooth', 'custom_driving_signal_square_wave'"
    assert use_LSST_cadence_test in [0, 1, 2], "use_LSST_cadence_test must be 0, 1, or 2"
    assert type(max_cpus) == int and max_cpus > 0, "max_cpus must be a positive integer"

    if skip_training_first_epoch and not load_model:
        print("not skipping training the first epoch since load_model is False")

    kernel_size = int(kernel_num_days/kernel_resolution)

    distributed = torch.distributed.is_available() and use_GPU and torch.cuda.device_count() > 1

    print(f'distributed: {distributed}')

    # print the number of GPUs
    print(f'number of GPUs counted: {torch.cuda.device_count()}')

    if distributed:
        # Set up distributed training
        dist.init_process_group("nccl", timeout=datetime.timedelta(days=1))
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        device_to_use = f"cuda:{rank}"
        device_ids = [rank]

        # get number of gpu's
        world_size = dist.get_world_size()
        print(f'world size: {world_size}')

        # change the learning rate based on the number of GPUs
        lr_init = lr_init * np.sqrt(world_size)

    else:
        if torch.cuda.is_available() and use_GPU:
            device = torch.device("cuda")
            device_to_use = "cuda"
        else:
            device = torch.device("cpu")
            device_to_use = "cpu"

        # rank is 0 if not using distributed, just used for validation and testing
        rank = 0 
        world_size = 1

    print_metrics = True if rank == 0 else False

    print(f'using device type: {device_to_use}')

    if distributed:
        # Divide the number of cpus across the different GPUs for DDP
        total_cpus = min(multiprocessing.cpu_count(),max_cpus)
        print(f'max cpus used: {total_cpus}')
        num_cpus_use = int(max(int(total_cpus // world_size),1))
        
    else: 
        max_cpus = 1 if local else max_cpus
        total_cpus = min(multiprocessing.cpu_count(), max_cpus)
        num_cpus_use = int(max(int(total_cpus),1))

    print(f'using {num_cpus_use} cpus for data loader')

    if use_GPU and torch.cuda.is_available():
        print(f'GPU type: {torch.cuda.get_device_name()}')

    # benchmark the transfer function simulation with and without GPUs
    if rank == 0 and benchmark_TF_models:
        print()
        print("benchmarking the transfer function simulation time")

        num_batch_benchmark = 10 if local else 20
        batch_size_benchmark = 10 if local else 50

        elapsed_time_GPU_no_grad, elapsed_time_cpu_no_grad, elapsed_time_GPU_with_grad, elapsed_time_cpu_with_grad, elapsed_time_cpu = benchmark_transfer_function_models(device, batch_size=1, num_batch=num_batch_benchmark, min_max_dict=min_max_dict, parameters_keys=parameters_keys, 
                                                                                                                    lambda_effective_Angstrom=lambda_effective_Angstrom, kernel_num_days=kernel_num_days, kernel_resolution=kernel_resolution, cosmo=cosmo)
        print(f"elapsed time for {num_batch_benchmark} transfer function simulations on GPU without gradients: {elapsed_time_GPU_no_grad:.2f} seconds")
        print(f"elapsed time for {num_batch_benchmark} transfer function simulations on CPU without gradients: {elapsed_time_cpu_no_grad:.2f} seconds")
        print(f"elapsed time for {num_batch_benchmark} transfer function simulations on GPU with gradients: {elapsed_time_GPU_with_grad:.2f} seconds")
        print(f"elapsed time for {num_batch_benchmark} transfer function simulations on CPU with gradients: {elapsed_time_cpu_with_grad:.2f} seconds")
        print(f"elapsed time for {num_batch_benchmark} transfer function simulations on CPU version, no batches: {elapsed_time_cpu:.2f} seconds")

        print(f"speedup for GPU without gradients without batches: {elapsed_time_cpu/elapsed_time_GPU_no_grad:.3f}")
        print(f"speedup for GPU without gradients with batches: {elapsed_time_cpu_no_grad/elapsed_time_GPU_no_grad:.3f}")
        print(f"speedup for GPU with gradients without batches: {elapsed_time_cpu/elapsed_time_GPU_with_grad:.3f}")
        print(f"speedup for GPU with gradients with batches: {elapsed_time_cpu_with_grad/elapsed_time_GPU_with_grad:.3f}")

        elapsed_time_GPU_no_grad, elapsed_time_cpu_no_grad, elapsed_time_GPU_with_grad, elapsed_time_cpu_with_grad, elapsed_time_cpu = benchmark_transfer_function_models(device, batch_size=batch_size_benchmark, num_batch=num_batch_benchmark, min_max_dict=min_max_dict, parameters_keys=parameters_keys, 
                                                                                                                    lambda_effective_Angstrom=lambda_effective_Angstrom, kernel_num_days=kernel_num_days, kernel_resolution=kernel_resolution, cosmo=cosmo)        
        
        print(f"elapsed time for {num_batch_benchmark} batches of {batch_size_benchmark} transfer function simulations on GPU without gradients: {elapsed_time_GPU_no_grad:.2f} seconds")
        print(f"elapsed time for {num_batch_benchmark} batches of {batch_size_benchmark} transfer function simulations on CPU without gradients: {elapsed_time_cpu_no_grad:.2f} seconds")
        print(f"elapsed time for {num_batch_benchmark} batches of {batch_size_benchmark} transfer function simulations on GPU with gradients: {elapsed_time_GPU_with_grad:.2f} seconds")
        print(f"elapsed time for {num_batch_benchmark} batches of {batch_size_benchmark} transfer function simulations on CPU with gradients: {elapsed_time_cpu_with_grad:.2f} seconds")
        print(f"elapsed time for {num_batch_benchmark} batches of {batch_size_benchmark} transfer function simulations on CPU version, no batches: {elapsed_time_cpu:.2f} seconds")

        print(f"speedup for GPU without gradients without batches: {elapsed_time_cpu/elapsed_time_GPU_no_grad:.3f}")
        print(f"speedup for GPU without gradients with batches: {elapsed_time_cpu_no_grad/elapsed_time_GPU_no_grad:.3f}")
        print(f"speedup for GPU with gradients without batches: {elapsed_time_cpu/elapsed_time_GPU_with_grad:.3f}")
        print(f"speedup for GPU with gradients with batches: {elapsed_time_cpu_with_grad/elapsed_time_GPU_with_grad:.3f}")

    if distributed:
        dist.barrier()

    #create a new training set each epoch on the fly. The validation and test sets are fixed by their random seed.
    train_dataset = Dataset_Loader(kernel_num_days, kernel_resolution, seed_list_train, cadence, model_spectrum, 
                                   cosmo, emline_template, galaxy_template, reddening_curve, min_magnitude, max_magnitude, reference_band, custom_driving_signal=None, augment=True, save_true_LC=True)
    val_dataset = Dataset_Loader(kernel_num_days, kernel_resolution, seed_list_val, cadence, model_spectrum, 
                                 cosmo, emline_template, galaxy_template, reddening_curve, min_magnitude, max_magnitude, reference_band, custom_driving_signal=custom_driving_signal_test, use_LSST_cadence=use_LSST_cadence_test,
                                 augment=False, save_true_LC=True)


    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        #val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_sampler = None
    else:
        train_sampler = None
        val_sampler = None

    # Can use larger factor bit will use more memory
    prefetch_factor = 1
    train_loader = DataLoader(train_dataset, shuffle=not distributed,batch_size=batch_size,num_workers=max(num_cpus_use-1,1),pin_memory=True,sampler=train_sampler,prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_dataset, shuffle=False,batch_size=batch_size_inference,num_workers=max(num_cpus_use-1,1),pin_memory=True,sampler=val_sampler,prefetch_factor=prefetch_factor)

    input_size = 2*num_bands

    # Define the latent SDE model
    latent_sde = LatentSDE(input_size=input_size,
                            hidden_size=hidden_size,
                            driving_latent_size=driving_latent_size,
                            context_size=context_size,
                            num_bands=num_bands,
                            num_iterations=num_iterations, 
                            driving_resolution=cadence, 
                            device=device, 
                            freq_effective=freq_effective,
                            lambda_effective_Angstrom=lambda_effective_Angstrom, 
                            n_params_accretion=n_params_accretion, 
                            n_params_variability=n_params_variability,
                            min_max_array=min_max_array,
                            kernel_num_days=kernel_num_days,
                            kernel_resolution=kernel_resolution,
                            parameters_keys=parameters_keys,
                            dt=dt,
                            method=method,
                            logqp=logqp,
                            num_layers=num_layers,
                            num_heads=num_heads,
                            param_loss_weight=param_loss_weight,
                            log_pxs_weight=log_pxs_weight,
                            log_pxs2_weight=log_pxs2_weight,
                            log_pxs2_leeway=log_pxs2_leeway,
                            num_Gaussian_parameterization=num_Gaussian_parameterization,
                            time_delay_loss_weight=time_delay_loss_weight,
                            dropout_rate=dropout_rate, 
                            give_redshift=give_redshift, 
                            KL_anneal_epochs=KL_anneal_epochs,
                            param_anneal_epochs=param_anneal_epochs,
                            relative_mean_time=relative_mean_time,
                            reference_band=reference_band,
                            min_magnitude=min_magnitude,
                            max_magnitude=max_magnitude,
                            ).to(device)

    if distributed:
        latent_sde = DDP(latent_sde, device_ids=device_ids, find_unused_parameters=True)

    if weight_decay > 0:
        optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)

    if load_model:
        print("trying to load model from previous run")
        latent_sde, optimizer = load_model_function(latent_sde, optimizer, load_path, device, distributed)

        if distributed:
            # synchronize the model across all the GPUs
            dist.barrier()


    # Compiling the model does not work with the current version of PyTorch
    #latent_sde = torch.compile(latent_sde, mode="reduce-overhead")

    n_params_NN = sum(p.numel() for p in latent_sde.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params_NN:,}")
    
    if ramp_up_epochs > 0:
        scheduler = CustomLRScheduler(optimizer, ramp_up_epochs=ramp_up_epochs, ramp_up_start_lr=lr_init/rampup_factor, ramp_up_end_lr=lr_init, decay_rate=lr_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

    if AMP:
        scaler = GradScaler()


    loss_list_epoch = []
    NGLL_list = []
    NGLL_driving_list = []
    RMSE_light_curve_list = []
    MAE_light_curve_list = []
    KL_div_list = []
    JS_div_list = []
    MSE_time_delay_list = []
    NGLL_time_delay_list = []
    MSE_params_list = []
    MAE_params_list = []
    NGLL_params_list = []
    n_batches = len(train_loader)
    for epoch in tqdm(range(num_epoch)):
        loss_epoch = 0.0

        if print_metrics:
            print(f'epoch: {epoch}', end='\r')

            # print the learning rate
            print(f'learning rate: {optimizer.param_groups[0]["lr"]}')

        if distributed:
            train_sampler.set_epoch(epoch)

        # We want to save the band mean magnitudes to compare with data
        param_true_list = []
        band_mean_mag_list = []

        # make sure the random seed is different for each batch
        current_time = time.time()
        seed = int(1e9*(current_time-int(current_time)))
        seed = seed - rank*100 # make sure the seed is different for each rank
        np.random.seed(seed)

        if epoch == 0 and skip_training_first_epoch and load_model:
            print("Skipping training for the first epoch, evaluating the model first")
            latent_sde.eval()
        else:
            latent_sde.train()
            for i, batch in enumerate(tqdm(train_loader, disable=not print_metrics)):
                time_batch_start = time.time()

                xs = batch['y'] # [B, T, out_dim]
                true_LC = batch['true_LC']  # [B, T, out_dim]
                param_true = batch['params'].float() # [B, n_params]
                #print(param_true.max(), param_true.min())
                param_true_list.append(param_true.detach().cpu().numpy())
                param_true = logit(param_true) # convert from [0,1] to [-inf, inf] using the logit function
                transfer_function_save = batch['transfer_function_save']
                driving_save = batch['driving_save']
                band_mean_mag_list.append(batch['band_mean_mag'].float())
                if give_redshift:
                    redshift = batch['redshift'].float()
                else:
                    redshift = None
                latent_sde.zero_grad()

                if distributed:
                    mean_time_true = latent_sde.module.get_mean_time(transfer_function_save.to(device))
                else:
                    mean_time_true = latent_sde.get_mean_time(transfer_function_save.to(device))

                # Make sure there are no nans in any of the input tensors
                assert torch.sum(torch.isnan(xs)) == 0, "xs contains nans"
                assert torch.sum(torch.isnan(true_LC)) == 0, "true_LC contains nans"
                assert torch.sum(torch.isnan(param_true)) == 0, "param_true contains nans"
                assert torch.sum(torch.isnan(mean_time_true)) == 0, "mean_time_true contains nans"
                if give_redshift:
                    assert torch.sum(torch.isnan(redshift)) == 0, "redshift contains nans"
                
                # evaluate the model
                with autocast(enabled=AMP):
                    output_dict = latent_sde(torch.clone(xs).to(device),
                                    true_LC = torch.clone(true_LC).to(device),
                                    driving_true = torch.clone(driving_save).to(device),
                                    mean_time_true=mean_time_true,
                                    param_true=torch.clone(param_true).to(device),
                                    redshift = torch.clone(redshift).to(device) if give_redshift else None,
                                    batch_fraction_sim=batch_fraction_sim,
                                    print_metrics=print_metrics,
                                    )
                
                # get the loss from the output dictionary
                loss = output_dict['loss']

                # delete the rest of the output dictionary to save memory since we only need the loss for training
                del output_dict

                # backpropagate the loss
                if AMP:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)  # Important before clipping
                else:
                    loss.backward()

                # Check for nans in the gradients, if there are any we do not update the weights
                nan_in_gradients = False
                #for param in latent_sde.parameters():
                for name, param in latent_sde.named_parameters():
                    # get the name of the parameter

                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            nan_in_gradients = True
                            print(f"nan in gradients of {name}")
                            break

                        if torch.isinf(param.grad).any():
                            nan_in_gradients = True
                            print(f"inf in gradients of {name}")
                            break

                        if torch.isnan(param).any():
                            nan_in_gradients = True
                            print(f"nan in parameters of {name}")
                            break

                        if torch.isinf(param).any():
                            nan_in_gradients = True
                            print(f"inf in parameters of {name}")
                            break
                
                # Clip the gradients to avoid exploding gradients problem in RNNs
                gradient_norm = torch.nn.utils.clip_grad_norm_(latent_sde.parameters(), max_norm=float('inf'))
                if rank == 0:
                    print(f"gradient_norm: {gradient_norm}")

                # Find and print the maximum gradient value
                max_grad = 0.0
                for param in latent_sde.parameters():
                    if param.grad is not None:
                        max_grad = max(max_grad, param.grad.abs().max().item())

                if rank == 0:
                    print(f"Maximum gradient value: {max_grad}")
                
                # Can slowly increase the grad_clip_value over the course of the first epoch, only if load_model is False since we are starting training from scratch
                # This prevents instability. Found not to be necassary after improving the model stability.

                grad_clip_value_use = grad_clip_value / max(abs(param_anneal_epochs)-epoch,1)
                
                torch.nn.utils.clip_grad_norm_(latent_sde.parameters(), max_norm=grad_clip_value_use)  # clip gradients to avoid exploding gradients problem in RNNs
                torch.nn.utils.clip_grad_value_(latent_sde.parameters(), clip_value=grad_clip_value_use/5.0)  # clip gradients to avoid exploding gradients problem in RNNs
                
                # If there are no nans in the gradients, update the weights
                if not nan_in_gradients and gradient_norm != float('inf') and gradient_norm != float('nan'):
                    if AMP:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                else:
                    print("nan in gradients, not updating weights this batch!")

                # keep track of the average loss for the epoch
                loss_epoch += loss.detach().cpu().item()    
                # delete the loss to save memory
                if print_metrics:
                    print(f"loss: {loss.detach().cpu().item()}")
                    print()
                del loss

                time_batch_end = time.time()
                if print_metrics:
                    print(f'training batch {i+1}/{n_batches} complete, took {time_batch_end-time_batch_start:.2f} seconds, approx {(time_batch_end-time_batch_start)*(n_batches-i)/(60.0*60.0):.2f} hours remaining')

            # update the learning rate at the end of each epoch
            scheduler.step()
            # set the model to evaluation mode for validation
            latent_sde.zero_grad()
            latent_sde.eval()
            # update the epoch for annealing the loss
            if distributed:
                latent_sde.module.step_epoch()
            else:
                latent_sde.step_epoch()

            loss_epoch /= n_batches
            loss_list_epoch.append(loss_epoch)
            print(f"Epoch {epoch}, training loss: {round(loss_epoch,4)}")


            if distributed:
                # synchronize the model across all the GPUs
                dist.barrier()

            if rank == 0:
                # save the model each epoch
                save_model(latent_sde, optimizer, save_path)
                print(f"Saved model to {save_path}/checkpoint.pth")

                # Conver band_mean_mag_list to a numpy array and save. If the file already exists, append to it.
                band_mean_mag = np.concatenate(band_mean_mag_list, axis=0)
                param_true = np.concatenate(param_true_list, axis=0)

                if epoch == 0:
                    np.save(f"{save_path}/band_mean_mag.npy", band_mean_mag)
                    np.save(f"{save_path}/param_true.npy", param_true)
                else:
                    band_mean_mag_old = np.load(f"{save_path}/band_mean_mag.npy")
                    band_mean_mag = np.concatenate([band_mean_mag_old, band_mean_mag], axis=0)
                    np.save(f"{save_path}/band_mean_mag.npy", band_mean_mag)

                    param_true_old = np.load(f"{save_path}/param_true.npy")
                    param_true = np.concatenate([param_true_old, param_true], axis=0)
                    np.save(f"{save_path}/param_true.npy", param_true)
        
        # We only want to perform inference on the validation set if we are not using distributed training or if we are using distributed training and the rank is 0 since multiple processes are spawned.
        if rank == 0:
            # Evaluate on validation set
            with torch.no_grad(): 

                print('Validating...')

                # Evaluate the light curve reconstruction
                NGLL_val = 0.0
                NGLL_driving_val = 0.0
                RMSE_light_curve_val = 0.0
                MAE_light_curve_val = 0.0
                
                # For GPR
                NGLL_GPR_val = 0.0
                RMSE_light_curve_GPR_val = 0.0
                MAE_light_curve_GPR_val = 0.0

                # For the parameters and transfer function reconstruction
                KL_div_val = 0.0
                JS_div_val = 0.0
                MSE_time_delay_val = 0.0
                NGLL_time_delay_val = 0.0
                MSE_params_val = 0.0
                MAE_params_val = 0.0
                NGLL_params_val = 0.0

                # Evaluate the light curve reconstruction over each iteration
                log_pxs_iteration_val = np.zeros(num_iterations)
                log_pxs2_iteration_val = np.zeros(num_iterations)
                RMSE_light_curve_iteration_val = np.zeros(num_iterations)
                MAE_light_curve_iteration_val = np.zeros(num_iterations)
                MSE_params_iteration_val = np.zeros(num_iterations)
                MAE_params_iteration_val = np.zeros(num_iterations)
                NGLL_params_iteration_val = np.zeros(num_iterations)
                NGLL_time_delay_iteration_val = np.zeros(num_iterations)
                MSE_time_delay_iteration_val = np.zeros(num_iterations)
                loss_iteration_val = np.zeros(num_iterations)

                # evaluate the coverage probabilities for our recovery
                num_quantiles = 50 #resolution of the quantiles for graphing the coverage probabilities
                all_quantiles = np.linspace(0.0,1.0,num_quantiles+1)
                coverage_prob_recovery = {}
                coverage_prob_recovery_GPR = {}

                for quantile in all_quantiles:
                    coverage_prob_recovery[quantile] = 0.0
                    if compare_to_GPR:
                        coverage_prob_recovery_GPR[quantile] = 0.0

                coverage_prob_parameters = {}
                for param in parameters_keys:
                    coverage_prob_parameters[param] = {}
                    for quantile in all_quantiles:
                        coverage_prob_parameters[param][quantile] = 0.0

                mean_value_dict = {}
                test_label_dict = {}

                average_inference_time = []
                for i, batch in enumerate(tqdm(val_loader, disable=not print_metrics)):
                    batch_start_time = time.time()
                    xs = batch['y']
                    true_LC = batch['true_LC']
                    driving_save = batch['driving_save']
                    param_true = batch['params'].float()
                    param_true = logit(param_true) # convert from [0,1] to [-inf, inf] using the logit function
                    transfer_function_save = batch['transfer_function_save']
                    if give_redshift:
                        redshift = batch['redshift'].float()
                    else:
                        redshift = None

                    with autocast(enabled=AMP):
                        before_inference = time.time()
                        if distributed:
                            output_dict = latent_sde.module.predict(torch.clone(xs).to(device),
                                                            redshift = torch.clone(redshift).to(device) if give_redshift else None,
                                                            batch_fraction_sim=batch_fraction_sim,
                                                            )
                        else:
                            output_dict = latent_sde.predict(torch.clone(xs).to(device),
                                                        redshift = torch.clone(redshift).to(device) if give_redshift else None,
                                                        batch_fraction_sim=batch_fraction_sim,
                                                        )
                        after_inference = time.time()
                        average_inference_time.append(after_inference - before_inference)
                            
                    loss = output_dict['loss']
                    _xs = output_dict['_xs']
                    driving_reconstructed = output_dict['driving_reconstructed']
                    z0 = output_dict['z0']
                    kernels = output_dict['kernels']
                    param_pred_reshape = output_dict['param_pred_reshape']
                    param_pred_L_matrix = output_dict['param_pred_L_matrix']
                    gaussian_mixture_coefficients = output_dict['gaussian_mixture_coefficients']
                    mean_time_pred = output_dict['mean_time_pred']
                    L_time_delay = output_dict['L_time_delay']
                    _xs_list = output_dict['_xs_list']
                    kernel_list = output_dict['kernel_list']
                    param_pred_mean_list = output_dict['param_pred_mean_list']
                    param_pred_L_list = output_dict['param_pred_L_list']
                    gaussian_mixture_coefficients_list = output_dict['gaussian_mixture_coefficients_list']
                    mean_time_pred_list = output_dict['mean_time_pred_list']
                    loss_list = output_dict['loss_list']
                    mean_mask = output_dict['mean_mask']

                    del output_dict

                    if compare_to_GPR and epoch == num_epoch-1:
                        #LC,LC_std,device,mean_across_bands=False,min_std=0.01
                        _xs_GPR = []
                        for batch_indx in range(xs.shape[0]):
                            #print(f"number of observations {torch.sum(xs[batch_indx,:,:num_bands] != 0).item()}")
                            _xs_GPR_val = gaussian_process_regression(LC=torch.clone(xs[batch_indx,:,:num_bands]).to(device),
                                                                    LC_std=torch.clone(xs[batch_indx,:,num_bands:]).to(device),
                                                                    mean_mask=torch.clone(mean_mask[batch_indx,:,:]).to(device),
                                                                    )
                            _xs_GPR.append(_xs_GPR_val)
                        _xs_GPR = np.stack(_xs_GPR,axis=0)
                        del _xs_GPR_val

                    # _xs[:, :, :num_bands] is the predicted light curve
                    # _xs[:, :, num_bands:] is the predicted log variance

                    # never used so delete to save memory
                    del redshift, z0

                    # Metrics for light curve reconstruction
                    # Also account for the bands that are not observed using the mean mask
                    #log_pxs = 0.5*(((true_LC.to(device)-_xs[:,:,:num_bands])**2)/torch.exp(_xs[:,:,num_bands:])+_xs[:,:,num_bands:]+np.log(2*np.pi)).mean()
                    #RMSE_light_curve = torch.sqrt(((true_LC.to(device)-_xs[:,:,:num_bands])**2).mean())
                    #MAE_light_curve = (torch.abs(true_LC.to(device)-_xs[:,:,:num_bands])).mean()
                    log_pxs = torch.mean( mean_mask * ( 0.5*(((true_LC.to(device)-_xs[:,:,:num_bands])**2)/torch.exp(_xs[:,:,num_bands:]+1e-7)+_xs[:,:,num_bands:]+np.log(2*np.pi)) ) ) / torch.mean(mean_mask)
                    padding_size = int(kernel_num_days/kernel_resolution)-1
                    log_pxs_driving = 0.5*torch.mean((driving_save[:, padding_size:].squeeze(-1).to(device)-driving_reconstructed[:,:,0])**2/(torch.exp(driving_reconstructed[:,:,1])+1e-7)+driving_reconstructed[:,:,1]+np.log(2*np.pi))
                    RMSE_light_curve = torch.sqrt( torch.mean( mean_mask * ((true_LC.to(device)-_xs[:,:,:num_bands])**2) ) / torch.mean(mean_mask) )
                    MAE_light_curve = torch.mean( mean_mask * (torch.abs(true_LC.to(device)-_xs[:,:,:num_bands])) ) / torch.mean(mean_mask)

                    NGLL_val += log_pxs.detach().cpu().item()
                    NGLL_driving_val += log_pxs_driving.detach().cpu().item()
                    RMSE_light_curve_val += RMSE_light_curve.detach().cpu().item()
                    MAE_light_curve_val += MAE_light_curve.detach().cpu().item()

                    if compare_to_GPR and epoch == num_epoch-1:
                        
                        # Also acount for the bands that are not observed using the mean mask
                        #log_pxs = 0.5*(((true_LC.to(device)-_xs[:,:,:num_bands])**2)/torch.exp(_xs[:,:,num_bands:])+_xs[:,:,num_bands:]+np.log(2*np.pi)).mean(dim=(1,2)).detach().cpu().numpy()
                        #RMSE_light_curve = torch.sqrt(((true_LC.to(device)-_xs[:,:,:num_bands])**2).mean(dim=(1,2))).detach().cpu().numpy()
                        #MAE_light_curve = (torch.abs(true_LC.to(device)-_xs[:,:,:num_bands])).mean(dim=(1,2)).detach().cpu().numpy()
                        # We keep the batch dimension to average over the batch

                        _xs = _xs.to(device) # The _xs is mean [B, T, :num_bands] and log_var [B, T, num_bands:]
                        log_pxs = torch.mean( mean_mask * ( 0.5*(((true_LC.to(device)-_xs[:,:,:num_bands])**2)/torch.exp(_xs[:,:,num_bands:]) + _xs[:,:,num_bands:]+np.log(2*np.pi)) ), dim=(1,2) ) / torch.mean(mean_mask, dim=(1,2))
                        RMSE_light_curve = torch.sqrt( torch.mean( mean_mask * ((true_LC.to(device)-_xs[:,:,:num_bands])**2), dim=(1,2)) / torch.mean(mean_mask, dim=(1,2)) )
                        MAE_light_curve = torch.mean( mean_mask * (torch.abs(true_LC.to(device)-_xs[:,:,:num_bands])), dim=(1,2) ) / torch.mean(mean_mask, dim=(1,2))

                        # convert form torch tensors to numpy arrays
                        log_pxs = log_pxs.detach().cpu().numpy()
                        RMSE_light_curve = RMSE_light_curve.detach().cpu().numpy()
                        MAE_light_curve = MAE_light_curve.detach().cpu().numpy()

                        if i == 0:
                            NGLL_median = log_pxs
                            RMSE_light_curve_median = RMSE_light_curve
                            MAE_light_curve_median = MAE_light_curve
                        else:
                            NGLL_median = np.concatenate((NGLL_median, log_pxs), axis=0)
                            RMSE_light_curve_median = np.concatenate((RMSE_light_curve_median, RMSE_light_curve), axis=0)
                            MAE_light_curve_median = np.concatenate((MAE_light_curve_median, MAE_light_curve), axis=0)
                        del log_pxs, RMSE_light_curve, MAE_light_curve

                        # We keep the batch dimension to average over the batch
                        _xs_GPR = torch.tensor(_xs_GPR).type_as(true_LC).to(device) # The _xs_GPR is mean [B, T, :num_bands] and std [B, T, num_bands:]
                        log_pxs_GPR = torch.mean( mean_mask * ( 0.5*(((true_LC.to(device)-_xs_GPR[:,:,:num_bands])**2)/(_xs_GPR[:,:,num_bands:]**2+1e-6) + torch.log(_xs_GPR[:,:,num_bands:]**2+1e-6)+np.log(2*np.pi)) ), dim=(1,2) ) / torch.mean(mean_mask, dim=(1,2))
                        RMSE_light_curve_GPR = torch.sqrt( torch.mean( mean_mask * ((true_LC.to(device)-_xs_GPR[:,:,:num_bands])**2), dim=(1,2)) / torch.mean(mean_mask, dim=(1,2)) )
                        MAE_light_curve_GPR = torch.mean( mean_mask * (torch.abs(true_LC.to(device)-_xs_GPR[:,:,:num_bands])), dim=(1,2) ) / torch.mean(mean_mask, dim=(1,2))

                        # convert form torch tensors to numpy arrays
                        log_pxs_GPR = log_pxs_GPR.detach().cpu().numpy()
                        RMSE_light_curve_GPR = RMSE_light_curve_GPR.detach().cpu().numpy()
                        MAE_light_curve_GPR = MAE_light_curve_GPR.detach().cpu().numpy()

                        if i == 0:
                            NGLL_GPR_median = log_pxs_GPR
                            RMSE_light_curve_GPR_median = RMSE_light_curve_GPR
                            MAE_light_curve_GPR_median = MAE_light_curve_GPR
                        else:
                            NGLL_GPR_median = np.concatenate((NGLL_GPR_median, log_pxs_GPR), axis=0)
                            RMSE_light_curve_GPR_median = np.concatenate((RMSE_light_curve_GPR_median, RMSE_light_curve_GPR), axis=0)
                            MAE_light_curve_GPR_median = np.concatenate((MAE_light_curve_GPR_median, MAE_light_curve_GPR), axis=0)

                        # Sum over the batch and add to the total
                        NGLL_GPR_val += log_pxs_GPR.mean()
                        RMSE_light_curve_GPR_val += RMSE_light_curve_GPR.mean()
                        MAE_light_curve_GPR_val += MAE_light_curve_GPR.mean()

                        del log_pxs_GPR, RMSE_light_curve_GPR, MAE_light_curve_GPR

                    if distributed:
                        KL_div_val += latent_sde.module.KL_div(transfer_function_save.to(device), kernels).detach().cpu().item()
                        JS_div_val = latent_sde.module.JS_div(transfer_function_save.to(device), kernels).detach().cpu().item()
                        # metrics for the time delay
                        mean_time_pred = latent_sde.module.get_mean_time(kernels)
                        mean_time_true = latent_sde.module.get_mean_time(transfer_function_save.to(device))
                    else:
                        KL_div_val += latent_sde.KL_div(transfer_function_save.to(device), kernels).detach().cpu().item()
                        JS_div_val = latent_sde.JS_div(transfer_function_save.to(device), kernels).detach().cpu().item()
                        # metrics for the time delay
                        mean_time_pred = latent_sde.get_mean_time(kernels)
                        mean_time_true = latent_sde.get_mean_time(transfer_function_save.to(device))

                    if relative_mean_time:
                        # subtract the reference band from the other bands
                        mean_time_true = mean_time_true - mean_time_true[:, reference_band].unsqueeze(1)
                        # get rid of the reference band since it is always 0
                        mean_time_true = torch.cat((mean_time_true[:,:reference_band], mean_time_true[:,reference_band+1:]), dim=1)

                        mean_time_pred = mean_time_pred - mean_time_pred[:, reference_band].unsqueeze(1)
                        mean_time_pred = torch.cat((mean_time_pred[:,:reference_band], mean_time_pred[:,reference_band+1:]), dim=1)

                        N_mean_band = num_bands-1
                    else:
                        N_mean_band = num_bands
                    
                    # Now also account for the bands that are not observed using the mean mask
                    MSE_time_delay_val += ((mean_time_true-mean_time_pred)**2).mean().detach().cpu().item()
                    #MSE_time_delay_val += torch.sum( mean_mask * ((mean_time_true-mean_time_pred)**2) ) / torch.sum(mean_mask)

                    time_delay_dist = MultivariateNormal(loc=mean_time_pred, scale_tril=L_time_delay)
                    time_delay_loss = -time_delay_dist.log_prob(mean_time_true).mean()/N_mean_band
                    #time_delay_loss = -torch.sum(mean_mask * time_delay_dist.log_prob(mean_time_true)) / torch.sum(mean_mask)
                    NGLL_time_delay_val += time_delay_loss.detach().cpu().item()

                    param_pred_mean = torch.sum(param_pred_reshape * gaussian_mixture_coefficients.unsqueeze(1), dim=-1)
                    # metrics for the parameters
                    MSE_params_val += ((expit(param_true.to(device))-expit(param_pred_mean))**2).mean().detach().cpu().item() # convert from [-inf, inf] to [0,1] using the expit function
                    MAE_params_val += (torch.abs(expit(param_true.to(device))-expit(param_pred_mean))).mean().detach().cpu().item() # convert from [-inf, inf] to [0,1] using the expit function

                    # Initialize the log probabilities
                    log_probs = []
                    for Gauss_indx in range(num_Gaussian_parameterization):

                        param_dist = MultivariateNormal(loc=param_pred_reshape[:, :, Gauss_indx], scale_tril=param_pred_L_matrix[:, :, :, Gauss_indx])
                        
                        # Calculate the log probability of the current Gaussian component
                        log_prob = param_dist.log_prob(param_true.to(device))
                        
                        # Add the weighted log probability to the list
                        weighted_log_prob = log_prob + torch.log(gaussian_mixture_coefficients[:, Gauss_indx])
                        log_probs.append(weighted_log_prob)

                    # Stack the log probabilities along the Gaussian component axis
                    log_probs = torch.stack(log_probs, dim=1)

                    # Apply the log-sum-exp trick to calculate the log-sum of exponentiated values
                    log_sum_exp = torch.logsumexp(log_probs, dim=1)

                    # Compute the NLL by taking the negative log-sum-exp value and scaling it
                    param_loss = -log_sum_exp.mean() / n_params

                    NGLL_params_val += param_loss.detach().cpu().item() # Still in the logit space


                    # Make a list of all the RMSE_params_val, MAE_params_val, NGLL_params_val so we can get uncertainty on the mean
                    if i == 0:
                        RMSE_params_val_list = torch.sqrt((expit(param_true.to(device))-expit(param_pred_mean))**2).mean(dim=1).detach().cpu().numpy()
                        MAE_params_val_list = torch.abs(expit(param_true.to(device))-expit(param_pred_mean)).mean(dim=1).detach().cpu().numpy()
                        NGLL_params_val_list = -log_sum_exp.detach().cpu().numpy() / n_params 
                    else:
                        RMSE_params_val_list = np.concatenate((RMSE_params_val_list, torch.sqrt((expit(param_true.to(device))-expit(param_pred_mean))**2).mean(dim=1).detach().cpu().numpy()), axis=0)
                        MAE_params_val_list = np.concatenate((MAE_params_val_list, torch.abs(expit(param_true.to(device))-expit(param_pred_mean)).mean(dim=1).detach().cpu().numpy()), axis=0)
                        NGLL_params_val_list = np.concatenate((NGLL_params_val_list, -log_sum_exp.detach().cpu().numpy() / n_params), axis=0)

                    del log_probs, log_sum_exp

                    mean_mask = mean_mask.detach().cpu().numpy()
                    true_LC = true_LC.detach().cpu().numpy()
                    xs = xs.detach().cpu().numpy()
                    # metric across the iterations
                    log_pxs_iteration = []
                    log_pxs2_iteration = []
                    RMSE_light_curve_iteration = []
                    MAE_light_curve_iteration = []
                    MSE_params_iteration = []
                    MAE_params_iteration = []
                    NGLL_params_iteration = []
                    MSE_time_delay_iteration = []
                    NGLL_time_delay_iteration = []
                    
                    
                    for iteration in range(num_iterations):

                        # Also account for the bands that are not observed using the mean mask
                        #log_pxs = 0.5*(((true_LC-_xs_list[iteration,:,:,:num_bands])**2)/np.exp(_xs_list[iteration,:,:,num_bands:])+_xs_list[iteration,:,:,num_bands:]+np.log(2*np.pi)).mean()
                        log_pxs = np.mean( mean_mask * ( 0.5*(((true_LC-_xs_list[iteration,:,:,:num_bands])**2)/np.exp(_xs_list[iteration,:,:,num_bands:])+_xs_list[iteration,:,:,num_bands:]+np.log(2*np.pi)) ) ) / np.mean(mean_mask)
                        log_pxs_iteration.append(log_pxs)
                        
                        mask = (xs[:,:,:num_bands] != 0.0).astype(float)
                        log_pxs2 = 0.5*np.sum(mask*(((xs[:,:,:num_bands]-_xs_list[iteration,:,:,:num_bands])**2)/(xs[:,:,num_bands:]**2+log_pxs2_leeway**2)))/np.sum(mask)
                        log_pxs2_iteration.append(log_pxs2)

                        #RMSE_light_curve = np.sqrt(((true_LC-_xs_list[iteration,:,:,:num_bands])**2).mean())
                        RMSE_light_curve = np.sqrt( np.mean( mean_mask * ((true_LC-_xs_list[iteration,:,:,:num_bands])**2) ) / np.mean(mean_mask) )
                        RMSE_light_curve_iteration.append(RMSE_light_curve)

                        #MAE_light_curve = (torch.abs(true_LC-_xs_list[iteration,:,:,:num_bands])).mean()
                        MAE_light_curve = np.mean( mean_mask * (np.abs(true_LC-_xs_list[iteration,:,:,:num_bands])) ) / np.mean(mean_mask)
                        MAE_light_curve_iteration.append(MAE_light_curve)

                        param_pred_mean = np.sum(param_pred_mean_list[iteration,:,:,:] * np.expand_dims(gaussian_mixture_coefficients_list[iteration],axis=1), axis=-1)
                        MSE_params = np.mean((expit(param_true.detach().cpu().numpy(), numpy=True)-expit(param_pred_mean, numpy=True))**2)
                        MSE_params_iteration.append(MSE_params)

                        MAE_params = np.mean(np.abs(expit(param_true.detach().cpu().numpy(), numpy=True)-expit(param_pred_mean, numpy=True)))
                        MAE_params_iteration.append(MAE_params)
                        
                        log_probs = []
                        for Gauss_indx in range(num_Gaussian_parameterization):
                            param_dist_iter = MultivariateNormal(loc=torch.tensor(param_pred_mean_list[iteration,:,:,Gauss_indx]).type_as(param_true).to(device),
                                                                scale_tril=torch.tensor(param_pred_L_list[iteration,:,:,:,Gauss_indx]).type_as(param_true).to(device))
                            
                            # Calculate the log probability of the current Gaussian component
                            log_prob = param_dist_iter.log_prob(param_true.to(device))
                            
                            # Add the weighted log probability to the list
                            weighted_log_prob = log_prob + torch.log(torch.tensor(gaussian_mixture_coefficients_list[iteration, :, Gauss_indx]).type_as(param_true).to(device))
                            log_probs.append(weighted_log_prob)

                        # Stack the log probabilities along the Gaussian component axis
                        log_probs = torch.stack(log_probs, dim=1)

                        # Apply the log-sum-exp trick to calculate the log-sum of exponentiated values
                        log_sum_exp = torch.logsumexp(log_probs, dim=1)

                        # Compute the NLL by taking the negative log-sum-exp value and scaling it
                        param_loss = -log_sum_exp.mean() / n_params
                        del log_probs, log_sum_exp, weighted_log_prob

                        NGLL_params = param_loss.detach().cpu().item() # Still in the logit space
                        NGLL_params_iteration.append(NGLL_params)

                        # metrics for the time delay
                        mean_time_pred = torch.tensor(mean_time_pred_list[iteration], device=device, dtype=mean_time_true.dtype)
                        MSE_time_delay = ((mean_time_true-mean_time_pred)**2).mean().detach().cpu().item()
                        MSE_time_delay_iteration.append(MSE_time_delay)

                        time_delay_dist = MultivariateNormal(loc=mean_time_pred, scale_tril=L_time_delay)
                        time_delay_loss = -time_delay_dist.log_prob(mean_time_true).mean()/N_mean_band
                        NGLL_time_delay = time_delay_loss.detach().cpu().item()
                        NGLL_time_delay_iteration.append(NGLL_time_delay)

                    del param_dist_iter, param_loss, log_pxs, RMSE_light_curve, MAE_light_curve

                    log_pxs_iteration = np.array(log_pxs_iteration)
                    log_pxs2_iteration = np.array(log_pxs2_iteration)
                    RMSE_light_curve_iteration = np.array(RMSE_light_curve_iteration)
                    MAE_light_curve_iteration = np.array(MAE_light_curve_iteration)
                    MSE_params_iteration = np.array(MSE_params_iteration)
                    MAE_params_iteration = np.array(MAE_params_iteration)
                    NGLL_params_iteration = np.array(NGLL_params_iteration)
                    MSE_time_delay_iteration = np.array(MSE_time_delay_iteration)
                    NGLL_time_delay_iteration = np.array(NGLL_time_delay_iteration)
                    
                    log_pxs_iteration_val += log_pxs_iteration
                    log_pxs2_iteration_val += log_pxs2_iteration
                    RMSE_light_curve_iteration_val += RMSE_light_curve_iteration
                    MAE_light_curve_iteration_val += MAE_light_curve_iteration
                    MSE_params_iteration_val += MSE_params_iteration
                    MAE_params_iteration_val += MAE_params_iteration
                    NGLL_params_iteration_val += NGLL_params_iteration
                    MSE_time_delay_iteration_val += MSE_time_delay_iteration
                    NGLL_time_delay_iteration_val += NGLL_time_delay_iteration

                    # coverage probabilities
                    # Only account for the bands that are observed
                    recovery_mean = _xs[:,:,:num_bands].detach().cpu().numpy()
                    recovery_std = torch.sqrt(torch.exp(_xs[:,:,num_bands:])).detach().cpu().numpy()
                    num_sigma_recovery = np.abs((recovery_mean-true_LC)/recovery_std) 
                    for quantile in all_quantiles:
                        #coverage_prob_recovery[quantile] += (num_sigma_recovery <= stats.norm.ppf(1-(1-quantile)/2)).astype(recovery_mean.dtype).mean()
                        coverage_prob_recovery[quantile] += np.mean( mean_mask * (num_sigma_recovery <= stats.norm.ppf(1-(1-quantile)/2)).astype(recovery_mean.dtype) ) / np.mean(mean_mask)
                    # coverage probabilities for GPR baseline
                    if compare_to_GPR and num_epoch-1 == epoch:
                        recovery_mean_GPR = _xs_GPR[:,:,:num_bands].detach().cpu().numpy()
                        recovery_std_GPR = _xs_GPR[:,:,num_bands:].detach().cpu().numpy()
                        num_sigma_recovery_GPR = np.abs((recovery_mean_GPR-true_LC)/recovery_std_GPR)
                        for quantile in all_quantiles:
                            coverage_prob_recovery_GPR[quantile] += np.mean( mean_mask * (num_sigma_recovery_GPR <= stats.norm.ppf(1-(1-quantile)/2)).astype(recovery_mean.dtype) ) / np.mean(mean_mask)
                        del recovery_mean_GPR, recovery_std_GPR, num_sigma_recovery_GPR

                    # coverage probabilities for the parameters
                    num_samples = 100_000
                    sample = 0.0
                    dist_list = []
                    for Gauss_indx in range(num_Gaussian_parameterization):
                        param_dist = MultivariateNormal(loc=param_pred_reshape[:,:,Gauss_indx], scale_tril=param_pred_L_matrix[:,:,:,Gauss_indx])
                        #sample = sample+param_dist.sample((num_samples,))
                        dist_list.append(param_dist)

                    if distributed:
                        sample = latent_sde.module.sample_mixture(num_samples=num_samples, 
                                                        dists=dist_list, 
                                                        weights=gaussian_mixture_coefficients)
                    else:
                        sample = latent_sde.sample_mixture(num_samples=num_samples, 
                                                    dists=dist_list, 
                                                    weights=gaussian_mixture_coefficients)
                
                
                    #sample = sample/num_Gaussian_parameterization
                    sample = expit(sample) # convert from [-inf, inf] to [0,1] using the expit function
                    for quantile in all_quantiles:
                        for param in parameters_keys:
                            
                            sample_val = sample[:, :, parameters_keys.index(param)]

                            param_true_tensor = expit(param_true[:, parameters_keys.index(param)]).to(device)
                            quantile_L = torch.quantile(sample_val, torch.tensor([0.5-quantile/2.]).type_as(param_true).to(device), dim=0)
                            quantile_H = torch.quantile(sample_val, torch.tensor([0.5+quantile/2.]).type_as(param_true).to(device), dim=0)

                            value = ((quantile_L <= param_true_tensor) & (param_true_tensor <= quantile_H)).float().mean().item()
                            coverage_prob_parameters[param][quantile] += value
                    del param_true_tensor, quantile_L, quantile_H, value, sample_val

                    sample = sample.detach().cpu().numpy() # shape (num_samples, batch_size, n_params)
                    if i == 0:
                        eval_lower_bound = np.quantile(sample,0.5-0.68/2.,axis=0)
                        eval_upper_bound = np.quantile(sample,0.5+0.68/2.,axis=0)
                        eval_median = np.quantile(sample,0.5,axis=0)
                    else:
                        eval_lower_bound = np.concatenate((eval_lower_bound,np.quantile(sample,0.5-0.68/2.,axis=0)),axis=0)
                        eval_upper_bound = np.concatenate((eval_upper_bound,np.quantile(sample,0.5+0.68/2.,axis=0)),axis=0)
                        eval_median = np.concatenate((eval_median,np.quantile(sample,0.5,axis=0)),axis=0)

                    if i == 0:
                        mean_values = sample.mean(axis=0)
                        true_values = expit(param_true).detach().cpu().numpy()
                    else:
                        mean_values = np.concatenate((mean_values, sample.mean(axis=0)), axis=0)
                        true_values = np.concatenate((true_values, expit(param_true).detach().cpu().numpy()), axis=0)

                    del mean_time_true, mean_time_pred, N_mean_band, time_delay_dist, time_delay_loss, param_true, mean_mask, true_LC, _xs, xs, driving_reconstructed, param_pred_reshape, param_pred_L_matrix, L_time_delay, loss
                    del recovery_mean, recovery_std, num_sigma_recovery
                    del _xs_list, kernel_list, param_pred_mean_list, param_pred_L_list, gaussian_mixture_coefficients_list, mean_time_pred_list, loss_list
                    del sample
                    
                    # run garbage collector to free up memory just in case
                    gc.collect()
                    torch.cuda.empty_cache()

                    batch_end_time = time.time()
                    if print_metrics:
                        print(f'validation batches {i+1}/{len(val_loader)} complete, this took {batch_end_time-batch_start_time:.4f} seconds, approximately {(batch_end_time-batch_start_time)*(len(val_loader)-(i+1))/3600:.3f} hours remaining')


                # save the mean values and true values for the confusion matrices
                if epoch == num_epoch-1:
                    np.save(f"{save_path}/mean_values_test.npy", mean_values)
                    np.save(f"{save_path}/eval_lower_bound_test.npy", eval_lower_bound)
                    np.save(f"{save_path}/eval_upper_bound_test.npy", eval_upper_bound)
                    np.save(f"{save_path}/eval_median_test.npy", eval_median)
                    np.save(f"{save_path}/true_values_test.npy", true_values)

                log_mass_limit_list = [None, 8.0, 8.5, 9.0, 9.5]
                for log_mass_limit in log_mass_limit_list: 
                    try:
                        plot_mean_versus_truth(mean_values,true_values,num_bins=20,log_mass_limit=log_mass_limit)
                        # plot the confusion matrices with the median instead of the mean
                        plot_mean_versus_truth(eval_median,true_values,num_bins=20,log_mass_limit=log_mass_limit,name_extra='_median')
                        plot_scatter_plots(true_values,eval_median,eval_lower_bound,eval_upper_bound,log_mass_limit=log_mass_limit,max_num_samples=100)
                        plot_scatter_plots(true_values,mean_values,eval_lower_bound,eval_upper_bound,log_mass_limit=log_mass_limit,max_num_samples=50)
                    except:
                        plt.close('all')

                # divide by the number of batches
                NGLL_val /= len(val_loader)
                NGLL_driving_val /= len(val_loader)
                RMSE_light_curve_val /= len(val_loader)
                MAE_light_curve_val /= len(val_loader)

                NGLL_GPR_val /= len(val_loader)
                RMSE_light_curve_GPR_val /= len(val_loader)
                MAE_light_curve_GPR_val /= len(val_loader)

                KL_div_val /= len(val_loader)
                JS_div_val /= len(val_loader)
                MSE_time_delay_val /= len(val_loader)
                NGLL_time_delay_val /= len(val_loader)
                MSE_params_val /= len(val_loader)
                MAE_params_val /= len(val_loader)
                NGLL_params_val /= len(val_loader)

                log_pxs_iteration_val /= len(val_loader)
                log_pxs2_iteration_val /= len(val_loader)
                RMSE_light_curve_iteration_val /= len(val_loader)
                MAE_light_curve_iteration_val /= len(val_loader)
                MSE_params_iteration_val /= len(val_loader)
                MAE_params_iteration_val /= len(val_loader)
                NGLL_params_iteration_val /= len(val_loader)
                MSE_time_delay_iteration_val /= len(val_loader)
                NGLL_time_delay_iteration_val /= len(val_loader)

                average_inference_time = np.median(average_inference_time)
                average_inference_time /= len(val_loader)

                loss_iteration_val = log_pxs_weight*log_pxs_iteration_val + log_pxs2_weight*log_pxs2_iteration_val + param_loss_weight*NGLL_params_iteration_val + time_delay_loss_weight*NGLL_time_delay_iteration_val

                print()
                print(f"Average inference time per batch: {average_inference_time:.4f} s")
                print(f"Average inference time per light curve: {average_inference_time/batch_size_inference:.4f} s")
                print(f"Average inference time for 1,000,000 light curves: {average_inference_time/batch_size_inference*1e6 / 3600:.4f} hours")
                print()
                if num_iterations > 1:
                    plot_metric_vs_iteration(log_pxs_iteration_val, "NGLL recovery", epoch)
                    plot_metric_vs_iteration(log_pxs2_iteration_val, "NGLL2 recovery", epoch)
                    plot_metric_vs_iteration(RMSE_light_curve_iteration_val, "RMSE light curve", epoch)
                    plot_metric_vs_iteration(MAE_light_curve_iteration_val, "MAE light curve", epoch)
                    plot_metric_vs_iteration(MSE_params_iteration_val, "MSE params", epoch)
                    plot_metric_vs_iteration(np.sqrt(MSE_params_iteration_val), "RMSE params", epoch)
                    plot_metric_vs_iteration(MAE_params_iteration_val, "MAE params", epoch)
                    plot_metric_vs_iteration(NGLL_params_iteration_val, "NGLL params", epoch)
                    plot_metric_vs_iteration(MSE_time_delay_iteration_val, "MSE time delay", epoch)
                    plot_metric_vs_iteration(NGLL_time_delay_iteration_val, "NGLL time delay", epoch)
                    plot_metric_vs_iteration(loss_iteration_val, "loss", epoch)

                    # save the output of all the metrics
                    np.save(f"{save_path}/recovery/log_pxs_iteration_val.npy", log_pxs_iteration_val)
                    np.save(f"{save_path}/recovery/log_pxs2_iteration_val.npy", log_pxs2_iteration_val)
                    np.save(f"{save_path}/recovery/RMSE_light_curve_iteration_val.npy", RMSE_light_curve_iteration_val)
                    np.save(f"{save_path}/recovery/MAE_light_curve_iteration_val.npy", MAE_light_curve_iteration_val)
                    np.save(f"{save_path}/recovery/MSE_params_iteration_val.npy", MSE_params_iteration_val)
                    np.save(f"{save_path}/recovery/MAE_params_iteration_val.npy", MAE_params_iteration_val)
                    np.save(f"{save_path}/recovery/NGLL_params_iteration_val.npy", NGLL_params_iteration_val)
                    np.save(f"{save_path}/recovery/MSE_time_delay_iteration_val.npy", MSE_time_delay_iteration_val)
                    np.save(f"{save_path}/recovery/NGLL_time_delay_iteration_val.npy", NGLL_time_delay_iteration_val)
                    np.save(f"{save_path}/recovery/loss_iteration_val.npy", loss_iteration_val)

                # divide by the number of batches
                for quantile in all_quantiles:
                    coverage_prob_recovery[quantile] /= len(val_loader)
                    if compare_to_GPR:
                        coverage_prob_recovery_GPR[quantile] /= len(val_loader)

                for quantile in all_quantiles:
                    for param in parameters_keys:
                        coverage_prob_parameters[param][quantile] /= len(val_loader)
                
                plot_coverage_probability(all_quantiles, coverage_prob_recovery, coverage_prob_recovery_GPR, epoch)
                plot_coverage_probability_params(all_quantiles, coverage_prob_parameters, epoch)
                plot_coverage_probability_params_and_recovery(all_quantiles, coverage_prob_parameters, coverage_prob_recovery, epoch)

                # save the coverage probabilities in case we want to do a more statistical analysis
                # use pickle to save the dictionaries
                with open(f"{save_path}/coverage_prob_recovery.pkl", 'wb') as f:
                    pickle.dump(coverage_prob_recovery, f)
                with open(f"{save_path}/coverage_prob_recovery_GPR.pkl", 'wb') as f:
                    pickle.dump(coverage_prob_recovery_GPR, f)
                with open(f"{save_path}/coverage_prob_parameters.pkl", 'wb') as f:
                    pickle.dump(coverage_prob_parameters, f)

                NGLL_list.append(NGLL_val)
                NGLL_driving_list.append(NGLL_driving_val)
                RMSE_light_curve_list.append(RMSE_light_curve_val)
                MAE_light_curve_list.append(MAE_light_curve_val)
                KL_div_list.append(KL_div_val)
                JS_div_list.append(JS_div_val)
                MSE_time_delay_list.append(MSE_time_delay_val)
                NGLL_time_delay_list.append(NGLL_time_delay_val)
                MSE_params_list.append(MSE_params_val)
                MAE_params_list.append(MAE_params_val)
                NGLL_params_list.append(NGLL_params_val)

                print(f"Epoch {epoch}")

                print(f"validation NGLL: {NGLL_val:.4f}, RMSE_light_curve: {RMSE_light_curve_val:.6f}, MAE_light_curve: {MAE_light_curve_val:.6f}, KL_div: {KL_div_val:.6f}, JS_div: {JS_div_val:.6f}, MSE_time_delay: {MSE_time_delay_val:.5f}, NGLL_time_delay: {NGLL_time_delay_val:.4f}, MSE_params: {MSE_params_val:.6f}, RMSE_params: {np.sqrt(MSE_params_val):.6f}, MAE_params: {MAE_params_val:.6f}, NGLL_params: {NGLL_params_val:.6f}")
                if num_iterations > 1:
                    print(f"log_pxs_iteration: {log_pxs_iteration_val.round(4)}, log_pxs2_iteration: {log_pxs2_iteration_val.round(4)}, RMSE_light_curve_iteration: {RMSE_light_curve_iteration_val.round(4)}, MAE_light_curve_iteration: {MAE_light_curve_iteration_val.round(4)}, MSE_params_iteration: {MSE_params_iteration_val.round(4)}, RMSE_params_iteration: {np.sqrt(MSE_params_iteration_val).round(4)}, MAE_params_iteration: {MAE_params_iteration_val.round(4)}, NGLL_params_iteration: {NGLL_params_iteration_val.round(4)}, MSE_time_delay_iteration: {MSE_time_delay_iteration_val.round(4)}, NGLL_time_delay_iteration: {NGLL_time_delay_iteration_val.round(4)}")

                # The parameter NGLL, RMSE, MAE with uncertainty
                print()
                NGLL_param_val_median = np.median(NGLL_params_val_list)
                RMSE_param_val_median = np.median(RMSE_params_val_list)
                MAE_param_val_median = np.median(MAE_params_val_list)

                NGLL_param_val_median_abs_dev = np.median(np.abs(NGLL_params_val_list-NGLL_param_val_median))/np.sqrt(len(NGLL_params_val_list))
                RMSE_param_val_median_abs_dev = np.median(np.abs(RMSE_params_val_list-RMSE_param_val_median))/np.sqrt(len(RMSE_params_val_list))
                MAE_param_val_median_abs_dev = np.median(np.abs(MAE_params_val_list-MAE_param_val_median))/np.sqrt(len(MAE_params_val_list))

                print(f"validation median NGLL_params: {NGLL_param_val_median:.6f} +/- {NGLL_param_val_median_abs_dev:.6f}, RMSE_params: {RMSE_param_val_median:.6f} +/- {RMSE_param_val_median_abs_dev:.6f}, MAE_params: {MAE_param_val_median:.6f} +/- {MAE_param_val_median_abs_dev:.6f}")

                NGLL_param_val_mean = np.mean(NGLL_params_val_list)
                RMSE_param_val_mean = np.mean(RMSE_params_val_list)
                MAE_param_val_mean = np.mean(MAE_params_val_list)

                NGLL_param_val_mean_abs_dev = np.std(NGLL_params_val_list)/np.sqrt(len(NGLL_params_val_list))
                RMSE_param_val_mean_abs_dev = np.std(RMSE_params_val_list)/np.sqrt(len(RMSE_params_val_list))
                MAE_param_val_mean_abs_dev = np.std(MAE_params_val_list)/np.sqrt(len(MAE_params_val_list))

                print(f"validation mean NGLL_params: {NGLL_param_val_mean:.6f} +/- {NGLL_param_val_mean_abs_dev:.6f}, RMSE_params: {RMSE_param_val_mean:.6f} +/- {RMSE_param_val_mean_abs_dev:.6f}, MAE_params: {MAE_param_val_mean:.6f} +/- {MAE_param_val_mean_abs_dev:.6f}")
                print()

                # save the metrics in a text file
                with open(f"{save_path}/metrics_params.txt", "a") as f:
                    f.write(f"Epoch {epoch}\n")
                    f.write(f"validation median NGLL_params: {NGLL_param_val_median:.6f} +/- {NGLL_param_val_median_abs_dev:.6f}, RMSE_params: {RMSE_param_val_median:.6f} +/- {RMSE_param_val_median_abs_dev:.6f}, MAE_params: {MAE_param_val_median:.6f} +/- {MAE_param_val_median_abs_dev:.6f}\n")
                    f.write(f"validation mean NGLL_params: {NGLL_param_val_mean:.6f} +/- {NGLL_param_val_mean_abs_dev:.6f}, RMSE_params: {RMSE_param_val_mean:.6f} +/- {RMSE_param_val_mean_abs_dev:.6f}, MAE_params: {MAE_param_val_mean:.6f} +/- {MAE_param_val_mean_abs_dev:.6f}\n")
                    f.write("\n")

                if compare_to_GPR and epoch == num_epoch-1:
                    print(f"validation NGLL: {round(NGLL_val,4)}, RMSE_light_curve: {round(RMSE_light_curve_val,6)}, MAE_light_curve: {round(MAE_light_curve_val,6)}")
                    print(f"validation NGLL_GPR: {round(NGLL_GPR_val,4)}, RMSE_light_curve_GPR: {round(RMSE_light_curve_GPR_val,6)}, MAE_light_curve_GPR: {round(MAE_light_curve_GPR_val,6)}")

                    # Evaluate the median and median absolute deviation on the median of the NGLL, RMSE and MAE
                    # Save the arrays in case we want to do a more statistical analysis
                    np.save(f"{save_path}/NGLL_median.npy", NGLL_median)
                    np.save(f"{save_path}/RMSE_light_curve_median.npy", RMSE_light_curve_median)
                    np.save(f"{save_path}/MAE_light_curve_median.npy", MAE_light_curve_median)
                    np.save(f"{save_path}/NGLL_GPR_median.npy", NGLL_GPR_median)
                    np.save(f"{save_path}/RMSE_light_curve_GPR_median.npy", RMSE_light_curve_GPR_median)
                    np.save(f"{save_path}/MAE_light_curve_GPR_median.npy", MAE_light_curve_GPR_median)

                    NGLL_val_median = np.median(NGLL_median)
                    RMSE_light_curve_val_median = np.median(RMSE_light_curve_median)
                    MAE_light_curve_val_median = np.median(MAE_light_curve_median)

                    NGLL_val_median_abs_dev = np.median(np.abs(NGLL_median-NGLL_val_median))/np.sqrt(len(NGLL_median))
                    RMSE_light_curve_val_median_abs_dev = np.median(np.abs(RMSE_light_curve_median-RMSE_light_curve_val_median))/np.sqrt(len(RMSE_light_curve_median))
                    MAE_light_curve_val_median_abs_dev = np.median(np.abs(MAE_light_curve_median-MAE_light_curve_val_median))/np.sqrt(len(MAE_light_curve_median))

                    NGLL_GPR_val_median = np.median(NGLL_GPR_median)
                    RMSE_light_curve_GPR_val_median = np.median(RMSE_light_curve_GPR_median)
                    MAE_light_curve_GPR_val_median = np.median(MAE_light_curve_GPR_median)

                    NGLL_GPR_val_median_abs_dev = np.median(np.abs(NGLL_GPR_median-NGLL_GPR_val_median))/np.sqrt(len(NGLL_GPR_median))
                    RMSE_light_curve_GPR_val_median_abs_dev = np.median(np.abs(RMSE_light_curve_GPR_median-RMSE_light_curve_GPR_val_median))/np.sqrt(len(RMSE_light_curve_GPR_median))
                    MAE_light_curve_GPR_val_median_abs_dev = np.median(np.abs(MAE_light_curve_GPR_median-MAE_light_curve_GPR_val_median))/np.sqrt(len(MAE_light_curve_GPR_median))

                    print()

                    print(f"validation median NGLL: {NGLL_val_median:.4f} +/- {NGLL_val_median_abs_dev:.4f}, RMSE_light_curve: {RMSE_light_curve_val_median:.6f} +/- {RMSE_light_curve_val_median_abs_dev:.6f}, MAE_light_curve: {MAE_light_curve_val_median:.6f} +/- {MAE_light_curve_val_median_abs_dev:.6f}")
                    print(f"validation median NGLL_GPR: {NGLL_GPR_val_median:.4f} +/- {NGLL_GPR_val_median_abs_dev:.4f}, RMSE_light_curve_GPR: {RMSE_light_curve_GPR_val_median:.6f} +/- {RMSE_light_curve_GPR_val_median_abs_dev:.6f}, MAE_light_curve_GPR: {MAE_light_curve_GPR_val_median:.6f} +/- {MAE_light_curve_GPR_val_median_abs_dev:.6f}")

                    NGLL_val_mean = np.mean(NGLL_median)
                    RMSE_light_curve_val_mean = np.mean(RMSE_light_curve_median)
                    MAE_light_curve_val_mean = np.mean(MAE_light_curve_median)

                    NGLL_val_mean_abs_dev = np.std(NGLL_median)/np.sqrt(len(NGLL_median))
                    RMSE_light_curve_val_mean_abs_dev = np.std(RMSE_light_curve_median)/np.sqrt(len(RMSE_light_curve_median))
                    MAE_light_curve_val_mean_abs_dev = np.std(MAE_light_curve_median)/np.sqrt(len(MAE_light_curve_median))

                    NGLL_GPR_val_mean = np.mean(NGLL_GPR_median)
                    RMSE_light_curve_GPR_val_mean = np.mean(RMSE_light_curve_GPR_median)
                    MAE_light_curve_GPR_val_mean = np.mean(MAE_light_curve_GPR_median)

                    NGLL_GPR_val_mean_abs_dev = np.std(NGLL_GPR_median)/np.sqrt(len(NGLL_GPR_median))
                    RMSE_light_curve_GPR_val_mean_abs_dev = np.std(RMSE_light_curve_GPR_median)/np.sqrt(len(RMSE_light_curve_GPR_median))
                    MAE_light_curve_GPR_val_mean_abs_dev = np.std(MAE_light_curve_GPR_median)/np.sqrt(len(MAE_light_curve_GPR_median))

                    print()
                    print(f"validation mean NGLL: {NGLL_val_mean:.4f} +/- {NGLL_val_mean_abs_dev:.4f}, RMSE_light_curve: {RMSE_light_curve_val_mean:.6f} +/- {RMSE_light_curve_val_mean_abs_dev:.6f}, MAE_light_curve: {MAE_light_curve_val_mean:.6f} +/- {MAE_light_curve_val_mean_abs_dev:.6f}")
                    print(f"validation mean NGLL_GPR: {NGLL_GPR_val_mean:.4f} +/- {NGLL_GPR_val_mean_abs_dev:.4f}, RMSE_light_curve_GPR: {RMSE_light_curve_GPR_val_mean:.6f} +/- {RMSE_light_curve_GPR_val_mean_abs_dev:.6f}, MAE_light_curve_GPR: {MAE_light_curve_GPR_val_mean:.6f} +/- {MAE_light_curve_GPR_val_mean_abs_dev:.6f}")
                    print()

                    # perform the statistical tests
                    NGLL_p_value = stats.ttest_rel(NGLL_median, NGLL_GPR_median)[1]
                    RMSE_light_curve_p_value = stats.ttest_rel(RMSE_light_curve_median, RMSE_light_curve_GPR_median)[1]
                    MAE_light_curve_p_value = stats.ttest_rel(MAE_light_curve_median, MAE_light_curve_GPR_median)[1]
                    print(f"t-test: p-value NGLL: {NGLL_p_value:.6f}, RMSE_light_curve: {RMSE_light_curve_p_value:.6f}, MAE_light_curve: {MAE_light_curve_p_value:.6f}")

                    NGLL_p_value = stats.mannwhitneyu(NGLL_median, NGLL_GPR_median, alternative='two-sided')[1]
                    RMSE_light_curve_p_value = stats.mannwhitneyu(RMSE_light_curve_median, RMSE_light_curve_GPR_median, alternative='two-sided')[1]
                    MAE_light_curve_p_value = stats.mannwhitneyu(MAE_light_curve_median, MAE_light_curve_GPR_median, alternative='two-sided')[1]
                    print(f"Mann-Whitney U test: p-value NGLL: {NGLL_p_value:.6f}, RMSE_light_curve: {RMSE_light_curve_p_value:.6f}, MAE_light_curve: {MAE_light_curve_p_value:.6f}")

                    NGLL_p_value = stats.ks_2samp(NGLL_median, NGLL_GPR_median)[1]
                    RMSE_light_curve_p_value = stats.ks_2samp(RMSE_light_curve_median, RMSE_light_curve_GPR_median)[1]
                    MAE_light_curve_p_value = stats.ks_2samp(MAE_light_curve_median, MAE_light_curve_GPR_median)[1]
                    print(f"Kolmogorov-Smirnov test: p-value NGLL: {NGLL_p_value:.6f}, RMSE_light_curve: {RMSE_light_curve_p_value:.6f}, MAE_light_curve: {MAE_light_curve_p_value:.6f}")
                
                    # save metrics and p-values in a text file
                    with open(f"{save_path}/metrics_LC.txt", "a") as f:
                        f.write(f"validation median NGLL: {NGLL_val_median:.4f} +/- {NGLL_val_median_abs_dev:.4f}, RMSE_light_curve: {RMSE_light_curve_val_median:.6f} +/- {RMSE_light_curve_val_median_abs_dev:.6f}, MAE_light_curve: {MAE_light_curve_val_median:.6f} +/- {MAE_light_curve_val_median_abs_dev:.6f}\n")
                        f.write(f"validation mean NGLL: {NGLL_val_mean:.4f} +/- {NGLL_val_mean_abs_dev:.4f}, RMSE_light_curve: {RMSE_light_curve_val_mean:.6f} +/- {RMSE_light_curve_val_mean_abs_dev:.6f}, MAE_light_curve: {MAE_light_curve_val_mean:.6f} +/- {MAE_light_curve_val_mean_abs_dev:.6f}\n")
                        
                        f.write(f"validation median NGLL_GPR: {NGLL_GPR_val_median:.4f} +/- {NGLL_GPR_val_median_abs_dev:.4f}, RMSE_light_curve_GPR: {RMSE_light_curve_GPR_val_median:.6f} +/- {RMSE_light_curve_GPR_val_median_abs_dev:.6f}, MAE_light_curve_GPR: {MAE_light_curve_GPR_val_median:.6f} +/- {MAE_light_curve_GPR_val_median_abs_dev:.6f}\n")
                        f.write(f"validation mean NGLL_GPR: {NGLL_GPR_val_mean:.4f} +/- {NGLL_GPR_val_mean_abs_dev:.4f}, RMSE_light_curve_GPR: {RMSE_light_curve_GPR_val_mean:.6f} +/- {RMSE_light_curve_GPR_val_mean_abs_dev:.6f}, MAE_light_curve_GPR: {MAE_light_curve_GPR_val_mean:.6f} +/- {MAE_light_curve_GPR_val_mean_abs_dev:.6f}\n")

                        f.write(f"t-test: p-value NGLL: {NGLL_p_value:.6f}, RMSE_light_curve: {RMSE_light_curve_p_value:.6f}, MAE_light_curve: {MAE_light_curve_p_value:.6f}\n")
                        f.write(f"Mann-Whitney U test: p-value NGLL: {NGLL_p_value:.6f}, RMSE_light_curve: {RMSE_light_curve_p_value:.6f}, MAE_light_curve: {MAE_light_curve_p_value:.6f}\n")
                        f.write(f"Kolmogorov-Smirnov test: p-value NGLL: {NGLL_p_value:.6f}, RMSE_light_curve: {RMSE_light_curve_p_value:.6f}, MAE_light_curve: {MAE_light_curve_p_value:.6f}\n")
                        
                        f.write("\n")


                # Plot loss, updating each epoch so we can track the progress during training.
                if len(loss_list_epoch) > 1:

                    epoch_list = np.arange(len(loss_list_epoch))+1 # start from 1
                    
                    plt.plot(epoch_list, loss_list_epoch)
                    plt.minorticks_on()
                    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                    plt.xlabel('epoch')
                    plt.ylabel('loss')
                    plt.xlim(epoch_list[0], epoch_list[-1])
                    plt.savefig(f"{save_path}/loss.pdf", bbox_inches='tight')
                    plt.close()

                    epoch_list = np.arange(len(NGLL_list))+1 # start from 1

                    # plot NGLL
                    plt.plot(epoch_list, NGLL_list)
                    plt.minorticks_on()
                    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                    plt.xlabel('epoch')
                    plt.ylabel('NGLL')
                    plt.xlim(epoch_list[0], epoch_list[-1])
                    plt.savefig(f"{save_path}/NGLL.pdf", bbox_inches='tight')
                    plt.close()

                    # plot NGLL_driving
                    plt.plot(epoch_list, NGLL_driving_list)
                    plt.minorticks_on()
                    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                    plt.xlabel('epoch')
                    plt.ylabel('NGLL_driving')
                    plt.xlim(epoch_list[0], epoch_list[-1])
                    plt.savefig(f"{save_path}/NGLL_driving.pdf", bbox_inches='tight')
                    plt.close()

                    # plot MSE_light_curve
                    plt.plot(epoch_list, RMSE_light_curve_list)
                    plt.minorticks_on()
                    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                    plt.xlabel('epoch')
                    plt.ylabel('RMSE light curve')
                    plt.xlim(epoch_list[0], epoch_list[-1])
                    plt.savefig(f"{save_path}/MSE_light_curve.pdf", bbox_inches='tight')
                    plt.close()

                    # plot MAE_light_curve
                    plt.plot(epoch_list, MAE_light_curve_list)
                    plt.minorticks_on()
                    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                    plt.xlabel('epoch')
                    plt.ylabel('MAE light curve')
                    plt.xlim(epoch_list[0], epoch_list[-1])
                    plt.savefig(f"{save_path}/MAE_light_curve.pdf", bbox_inches='tight')
                    plt.close()

                    # plot KL_div
                    plt.plot(epoch_list, KL_div_list)
                    plt.minorticks_on()
                    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                    plt.xlabel('epoch')
                    plt.ylabel('KL_div')
                    plt.xlim(epoch_list[0], epoch_list[-1])
                    plt.savefig(f"{save_path}/KL_div.pdf", bbox_inches='tight')
                    plt.close()

                    # plot JS_div
                    plt.plot(epoch_list, JS_div_list)
                    plt.minorticks_on()
                    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                    plt.xlabel('epoch')
                    plt.ylabel('JS_div')
                    plt.xlim(epoch_list[0], epoch_list[-1])
                    plt.savefig(f"{save_path}/JS_div.pdf", bbox_inches='tight')
                    plt.close()

                    # plot MSE_time_delay
                    plt.plot(epoch_list, MSE_time_delay_list)
                    plt.minorticks_on()
                    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                    plt.xlabel('epoch')
                    plt.ylabel('MSE log time delay')
                    plt.xlim(epoch_list[0], epoch_list[-1])
                    plt.savefig(f"{save_path}/MSE_time_delay.pdf", bbox_inches='tight')
                    plt.close()

                    # plot NGLL_time_delay
                    plt.plot(epoch_list, NGLL_time_delay_list)
                    plt.minorticks_on()
                    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                    plt.xlabel('epoch')
                    plt.ylabel('NGLL_time_delay')
                    plt.xlim(epoch_list[0], epoch_list[-1])
                    plt.savefig(f"{save_path}/NGLL_time_delay.pdf", bbox_inches='tight')
                    plt.close()

                    # plot MSE parameters
                    plt.plot(epoch_list, MSE_params_list)
                    plt.minorticks_on()
                    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                    plt.xlabel('epoch')
                    plt.ylabel('MSE')
                    plt.xlim(epoch_list[0], epoch_list[-1])
                    plt.savefig(f"{save_path}/MSE_params.pdf", bbox_inches='tight')
                    plt.close()

                    # plot MAE parameters
                    plt.plot(epoch_list, MAE_params_list)
                    plt.minorticks_on()
                    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                    plt.xlabel('epoch')
                    plt.ylabel('MAE')
                    plt.xlim(epoch_list[0], epoch_list[-1])
                    plt.savefig(f"{save_path}/MAE_params.pdf", bbox_inches='tight')
                    plt.close()

                    # plot NGLL parameters
                    plt.plot(epoch_list, NGLL_params_list)
                    plt.minorticks_on()
                    plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                    plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                    plt.xlabel('epoch')
                    plt.ylabel('NGLL')
                    plt.xlim(epoch_list[0], epoch_list[-1])
                    plt.savefig(f"{save_path}/NGLL_params.pdf", bbox_inches='tight')
                    plt.close()

    
        # Now we load a single batch to visualize the results
        @torch.no_grad()
        def test_batch(batch, extra_folder=""):
            
            xs = batch['y']
            true_LC = batch['true_LC']
            param_true = batch['params'].float()
            transfer_function_save = batch['transfer_function_save'] 
            driving_save = batch['driving_save']

            if give_redshift:
                redshift = batch['redshift'].float()
            else:
                redshift = None

            with autocast(enabled=AMP):
                if distributed: 
                    output_dict = latent_sde.module.predict(torch.clone(xs).to(device),
                                                        save_iterations = True,
                                                        redshift = torch.clone(redshift).to(device) if give_redshift else None,
                                                        )
                else:
                    output_dict = latent_sde.predict(torch.clone(xs).to(device),
                                                save_iterations = True,
                                                redshift = torch.clone(redshift).to(device) if give_redshift else None,
                                                )
            loss = output_dict['loss']
            _xs = output_dict['_xs']
            driving_reconstructed = output_dict['driving_reconstructed']
            z0 = output_dict['z0']
            kernels = output_dict['kernels']
            param_pred_reshape = output_dict['param_pred_reshape']
            param_pred_L_matrix = output_dict['param_pred_L_matrix']
            gaussian_mixture_coefficients = output_dict['gaussian_mixture_coefficients']
            mean_time_pred = output_dict['mean_time_pred']
            L_time_delay = output_dict['L_time_delay']
            _xs_list = output_dict['_xs_list']
            kernel_list = output_dict['kernel_list']
            driving_list = output_dict['driving_list']
            param_pred_mean_list = output_dict['param_pred_mean_list']
            param_pred_L_list = output_dict['param_pred_L_list']
            gaussian_mixture_coefficients_list = output_dict['gaussian_mixture_coefficients_list']
            mean_mask = output_dict['mean_mask']
            del output_dict
            del redshift, gaussian_mixture_coefficients_list

            # not used 
            del loss
            num_recovery_plots = min(num_plot,_xs.shape[0])

            # metrics for the time delay
            if distributed:
                mean_time_pred = latent_sde.module.get_mean_time(kernels)
                mean_time_true = latent_sde.module.get_mean_time(transfer_function_save.to(device))

                mean, std, mean_mask = latent_sde.module.get_weighted_mean_std(torch.clone(xs).to(device))
            else:
                mean_time_pred = latent_sde.get_mean_time(kernels)
                mean_time_true = latent_sde.get_mean_time(transfer_function_save.to(device))

                mean, std, mean_mask = latent_sde.get_weighted_mean_std(torch.clone(xs).to(device))

            if relative_mean_time:

                mean_time_true = mean_time_true - mean_time_true[:, reference_band].unsqueeze(1)
                mean_time_true = torch.cat((mean_time_true[:,:reference_band], mean_time_true[:,reference_band+1:]), dim=1)
                
                mean_time_pred = mean_time_pred - mean_time_pred[:, reference_band].unsqueeze(1)
                mean_time_pred = torch.cat((mean_time_pred[:,:reference_band], mean_time_pred[:,reference_band+1:]), dim=1)

                N_mean_band = num_bands-1
            else:
                N_mean_band = num_bands

            mean_time_pred = mean_time_pred.detach().cpu().numpy()
            mean_time_true = mean_time_true.detach().cpu().numpy()

            num_samples = 200_000
            sample = 0.0
            dist_list = []
            for Gauss_indx in range(num_Gaussian_parameterization):
                param_dist = MultivariateNormal(loc=param_pred_reshape[:, : , Gauss_indx], scale_tril=param_pred_L_matrix[:, :, :, Gauss_indx])
                dist_list.append(param_dist)
            
            if distributed:
                sample = latent_sde.module.sample_mixture(num_samples=num_samples, 
                                    dists=dist_list, 
                                    weights=gaussian_mixture_coefficients)
            else:
                sample = latent_sde.sample_mixture(num_samples=num_samples, 
                                            dists=dist_list, 
                                            weights=gaussian_mixture_coefficients)
            sample = expit(sample) # convert from [-inf, inf] to [0,1] using the expit function
            # To plot the samples of different kernels
            kernel_num_samples = 5
            kernel_pred_samples = np.zeros((kernel_num_samples, num_recovery_plots, kernel_size, num_bands))
            for num in range(kernel_num_samples):
                if distributed:
                    kernel = latent_sde.module.sample_kernel(sample[num, :num_recovery_plots, :n_params_accretion], batch_fraction_sim=batch_fraction_sim).detach().cpu().numpy()
                else:
                    kernel = latent_sde.sample_kernel(sample[num, :num_recovery_plots, :n_params_accretion], batch_fraction_sim=batch_fraction_sim).detach().cpu().numpy() 
                kernel_pred_samples[num] = kernel

            del param_dist, param_pred_reshape, param_pred_L_matrix

            if compare_to_GPR and epoch == num_epoch-1:
                light_curve_GPR = []
                for num in range(num_recovery_plots):

                    light_curve_GPR_val = gaussian_process_regression(LC=torch.clone(xs[num,:,:num_bands]).to(device),
                                                                LC_std=torch.clone(xs[num,:,num_bands:]).to(device),
                                                                mean_mask=torch.clone(mean_mask[num]).to(device),
                                                                )
                    light_curve_GPR.append(light_curve_GPR_val)
                    
                light_curve_GPR = np.stack(light_curve_GPR, axis=0)
                
                # convert the uncertainty from std to log-variance to have the same format as our predictions _xs
                light_curve_GPR[:, :, num_bands:] = (light_curve_GPR[:, :, num_bands:] != 0).astype(light_curve_GPR.dtype) * np.log(light_curve_GPR[:, :, num_bands:]**2 + 1e-6)
                
            sample = sample.detach().cpu().numpy()
            xs = xs.detach().cpu().numpy()
            true_LC = true_LC.detach().cpu().numpy()
            _xs = _xs.detach().cpu().numpy()
            transfer_function_save = transfer_function_save.detach().cpu().numpy()
            kernels = kernels.detach().cpu().numpy()
            L_time_delay = L_time_delay.detach().cpu().numpy()
            driving_reconstructed = driving_reconstructed.detach().cpu().numpy()
            driving_save = driving_save.detach().cpu().numpy()
            mean_mask = mean_mask.squeeze(1).detach().cpu().numpy() # Now shape [batch_size, num_bands]

            # plot results with the first batch of the validation set
            for num in range(num_recovery_plots):
                plot_recovery(xs[num], true_LC[num], _xs[num], mean_mask[num], num, epoch, extra_name="", extra_folder=extra_folder, use_ylim=False) # Zoom in on each band to see the differences better
                plot_recovery_with_driving(xs[num], true_LC[num], _xs[num], mean_mask[num], driving_reconstructed[num], driving_save[num].squeeze(-1), kernel_num_days, cadence, num, epoch, extra_folder=extra_folder, use_ylim=False)
                plot_recovery_with_driving_and_kernels(xs[num], true_LC[num], _xs[num], mean_mask[num], driving_reconstructed[num], driving_save[num].squeeze(-1), transfer_function_save[num], kernels[num], kernel_num_days, kernel_size, cadence, num, epoch, extra_folder=extra_folder)
                
                # Plot the recovery with just the driving signal and the i-band, in case we need a smaller plot
                mean_mask_i_band = np.zeros_like(mean_mask[num])
                mean_mask_i_band[reference_band] = 1
                plot_recovery(xs[num], true_LC[num], _xs[num], mean_mask_i_band, num, epoch, extra_name="_i_band_only", extra_folder=extra_folder, use_ylim=False)
                plot_recovery_with_driving(xs[num], true_LC[num], _xs[num], mean_mask_i_band, driving_reconstructed[num], driving_save[num].squeeze(-1), kernel_num_days, cadence, num, epoch, extra_folder=extra_folder, use_ylim=False, extra_name="_i_band_only")
                plot_recovery_with_driving_and_kernels(xs[num], true_LC[num], _xs[num], mean_mask_i_band, driving_reconstructed[num], driving_save[num].squeeze(-1), transfer_function_save[num], kernels[num], kernel_num_days, kernel_size, cadence, num, epoch, extra_folder=extra_folder, extra_name="_i_band_only")

                if compare_to_GPR and epoch == num_epoch-1:
                    plot_recovery(xs[num], true_LC[num], light_curve_GPR[num], mean_mask[num], num, epoch, extra_name="_GPR_no_ylim", extra_folder=extra_folder, use_ylim=False) # Zoom in on each band to see the differences better

                plot_kernel_recovery(transfer_function_save[num], kernels[num], num, epoch, kernel_num_days, kernel_size, extra_folder=extra_folder)
                plot_kernel_recovery_reconstruction(kernels[num], num, epoch, kernel_num_days, kernel_size, reference_band=reference_band,extra_folder=extra_folder)
                plot_kernel_recovery_sample(transfer_function_save[num], kernels[num],  kernel_pred_samples[:, num], num, epoch, kernel_num_days, kernel_size, extra_folder=extra_folder)
                plot_driving_recovery_new(driving_reconstructed[num, :, 0], driving_save[num].squeeze(-1), num, extra_folder=extra_folder)

                if num_iterations > 1:
                    plot_kernel_recovery_iterative(transfer_function_save[num], kernel_list[:, num], num, epoch, kernel_num_days, kernel_size, extra_folder=extra_folder)
                    plot_recovery_iterative(xs[num], true_LC[num], _xs_list[:, num], _xs[num], reference_band, num, epoch, extra_name="_no_ylim", extra_folder=extra_folder, use_ylim=False)

                plot_time_delay_corner(mean_time_pred[num], mean_time_true[num], L_time_delay[num], mean_mask[num], num, epoch, kernel_num_days, kernel_resolution, 
                                    relative_mean_time=relative_mean_time, reference_band=reference_band, extra_folder=extra_folder)
            
                plot_corner_plot(sample[:, num],param_true[num],num,epoch, extra_folder=extra_folder)


        with torch.no_grad():
            # Test the recovery on different driving signals and cadences

            if local:
                # Just test the normal recovery when using locally
                test_names = ["recovery/",]
                custom_driving_signal_function_list = [None,]
                use_LSST_cadence_list = [0,]
            else:
                test_names = ["recovery/", "recovery_test1_high_cadence_sections/", "recovery_test1_DRW1/", "recovery_test2_sine/", "recovery_test3_sine_BPL/", "recovery_test4_two_sine/", "recovery_test5_sawtooth/", "recovery_test6_square_wave/"] #, "recovery_test7_sawtooth_high_cadence_sections/", "recovery_test8_two_sine_high_cadence_sections/"]
                custom_driving_signal_function_list = [None, None, custom_driving_signal_DRW, custom_driving_signal_sine, custom_driving_signal_sine_with_BPL, 
                                                    custom_driving_signal_two_sine, custom_driving_signal_sawtooth, custom_driving_signal_square_wave] #, custom_driving_signal_sawtooth, custom_driving_signal_two_sine]
                
                use_LSST_cadence_list = [0, 2, 0, 0, 0, 0, 0, 0] 

            
            use_rank_list = np.arange(len(use_LSST_cadence_list)) % world_size
            for use_rank, test_name, custom_driving_signal, use_LSST_cadence in zip(use_rank_list, test_names, custom_driving_signal_function_list, use_LSST_cadence_list):
                
                if rank == use_rank:
                    # Test using a sine wave as the driving signal
                    test_dataset = Dataset_Loader(kernel_num_days, kernel_resolution, seed_list_val, cadence, model_spectrum, cosmo, emline_template, galaxy_template, reddening_curve, 
                                                    min_magnitude, max_magnitude, reference_band, custom_driving_signal=custom_driving_signal, augment=False,save_true_LC=True, use_LSST_cadence=use_LSST_cadence)
                    
                    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=num_plot, num_workers=max(max(num_cpus_use-1,num_plot),1), pin_memory=False)

                    for batch in test_loader:
                        break
                
                    del test_dataset, test_loader

                    # creat folder for the test results
                    os.makedirs(f"{save_path}/recovery/{test_name}", exist_ok=True)
                    test_batch(batch, extra_folder=test_name)

                    # run garbage collector to free up memory just in case
                    gc.collect()
                    torch.cuda.empty_cache()

            del batch, test_names, custom_driving_signal_function_list

            # run garbage collector to free up memory just in case
            gc.collect()
            torch.cuda.empty_cache()


        if distributed:
            # synchronize the processes to make sure that all the processes have finished the epoch
            dist.barrier() 

    # After the training is done, destroy the distributed process group
    if distributed:
        dist.destroy_process_group()

# This was not used.
# Function to train an RNN baseline to compare parameter predictions to the latent SDE model
def main_RNN_baseline(
    batch_size=64,
    hidden_size=2, # 256
    num_layers=1, # 5
    num_heads=1, # 8
    num_epoch=1,
    lr_init=4.0e-4,
    lr_gamma=0.9,
    num_Gaussian_parameterization=5,
    min_magnitude=min_magnitude,
    max_magnitude=max_magnitude,
    model_spectrum=model_spectrum,
    cosmo=cosmo,
    use_GPU=True,
    kernel_num_days=800,
    kernel_resolution=cadence,
    reference_band=3,
    n_params=n_params,
    grad_clip_value=100.0,
    load_model=False,
    load_path="",
):
    distributed = torch.distributed.is_available() and use_GPU and torch.cuda.device_count() > 1
    print(f'distributed: {distributed}')

    # print the number of GPUs
    print(f'number of GPUs counted: {torch.cuda.device_count()}')

    if distributed:

        dist.init_process_group("nccl", timeout=datetime.timedelta(days=1))
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        device_to_use = f"cuda:{rank}"
        device_ids = [rank]

        # get number of gpu's
        world_size = dist.get_world_size()
        print(f'world size: {world_size}')
        # change the learning rate based on the number of GPUs
        lr_init = lr_init * np.sqrt(world_size)

    else:
        if torch.cuda.is_available() and use_GPU:
            device = torch.device("cuda")
            device_to_use = "cuda"
        else:
            device = torch.device("cpu")
            device_to_use = "cpu"
        # rank is 0 if not using distributed, just used for validation and testing
        rank = 0 
        world_size = 1

    print(f'using device type: {device_to_use}')

    if distributed:
        max_cpus = 100
        # Divide the number of cpus across the different GPUs for DDP
        total_cpus = min(multiprocessing.cpu_count(),max_cpus)
        #total_cpus = min(60, multiprocessing.cpu_count())
        num_cpus_use = total_cpus // world_size
        
    else: 
        max_cpus = 100
        total_cpus = min(multiprocessing.cpu_count(), max_cpus)
        num_cpus_use = total_cpus
    
    print(f'number of cpus used: {num_cpus_use}')

    #create a new training set each epoch on the fly. The validation and test sets are fixed by their random seed.
    train_dataset = Dataset_Loader(kernel_num_days, kernel_resolution, seed_list_train, cadence, model_spectrum, cosmo, emline_template, galaxy_template, reddening_curve, 
                                   min_magnitude, max_magnitude, reference_band, augment=True,save_true_LC=True, custom_driving_signal=None, use_LSST_cadence=0)
    val_dataset = Dataset_Loader(kernel_num_days, kernel_resolution, seed_list_val, cadence, model_spectrum, cosmo, emline_template, galaxy_template, reddening_curve, 
                                 min_magnitude, max_magnitude, reference_band, augment=False,save_true_LC=True, custom_driving_signal=None, use_LSST_cadence=0)

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = None
    else:
        train_sampler = None
        val_sampler = None

    prefetch_factor = 4
    train_loader = DataLoader(train_dataset, shuffle=not distributed,batch_size=batch_size,num_workers=max(num_cpus_use-1,1),pin_memory=True,sampler=train_sampler,prefetch_factor=prefetch_factor)
    # We can use all the cpus for the validation set since we only do it for the rank 0 process
    val_loader = DataLoader(val_dataset, shuffle=False,batch_size=batch_size,num_workers=max(num_cpus_use-1,1),pin_memory=True,sampler=val_sampler,prefetch_factor=prefetch_factor)

    input_size = 2*num_bands
    rnn = RNN_baseline(input_size, hidden_size, num_bands, num_layers, num_Gaussian_parameterization, n_params, num_heads=num_heads).to(device)

    if distributed:
        rnn = DDP(rnn, device_ids=device_ids, find_unused_parameters=True)

    optimizer = optim.Adam(params=rnn.parameters(), lr=lr_init)

    if load_model:
        load_model_function(rnn, optimizer, load_path, device, distributed)

        if distributed:
            dist.barrier()

    print_metrics = True if rank == 0 else False

    if print_metrics:
        print(f'number of parameters: {sum(p.numel() for p in rnn.parameters()):,}')

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,lr_gamma)

    # train the model
    ngll_train_epoch = []
    rmse_train_epoch = []
    mae_train_epoch = []

    ngll_val_epoch = []
    rmse_val_epoch = []
    mae_val_epoch = []

    for epoch in tqdm(range(num_epoch)):
        ngll_train = 0.0
        rmse_train = 0.0
        mae_train = 0.0
        
        ngll_val = 0.0
        rmse_val = 0.0
        mae_val = 0.0

        rnn.train()

        if distributed:
            dist.barrier()

        for i, batch in enumerate(tqdm(train_loader, disable=not print_metrics)):
            xs = batch['y']
            true_LC = batch['true_LC']
            param_true = batch['params'].float()
            transfer_function_save = batch['transfer_function_save'] 
            driving_save = batch['driving_save']

            optimizer.zero_grad()
            param_loss, rmse, mae, params_mean, samples, NGLL_batch, rmse_batch, mae_batch  = rnn(xs.to(device), 
                                                                                                param_true.to(device))
            param_loss.backward()

            if rank == 0:
                ngll = param_loss.detach().cpu().item()    
                rmse = rmse.detach().cpu().item()
                mae = mae.detach().cpu().item()

                ngll_train += ngll
                rmse_train += rmse
                mae_train += mae

            gradient_norm = torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=float('inf'))
            # clip the gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), grad_clip_value)

            if print_metrics:
                print(f'NGLL: {ngll:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, gradient_norm: {gradient_norm:.4f}')

            optimizer.step()
        
            del param_loss, rmse, mae, params_mean, samples, NGLL_batch, rmse_batch, mae_batch

        scheduler.step()
        rnn.zero_grad()
        rnn.eval()

        if distributed:
            dist.barrier()

        if rank == 0:
            with torch.no_grad():

                # save the model
                save_model(rnn, optimizer, save_path)
                
                num_quantiles = 50 #resolution of the quantiles for graphing the coverage probabilities
                all_quantiles = np.linspace(0.0,1.0,num_quantiles+1)

                coverage_prob_parameters = {}
                for param in parameters_keys:
                    coverage_prob_parameters[param] = {}
                    for quantile in all_quantiles:
                        coverage_prob_parameters[param][quantile] = 0.0

                for i, batch in enumerate(val_loader):
                    xs = batch['y']
                    true_LC = batch['true_LC']
                    param_true = batch['params'].float()
                    transfer_function_save = batch['transfer_function_save'] 
                    driving_save = batch['driving_save']

                    if distributed:
                        param_loss, rmse, mae, params_mean, samples, NGLL_batch, rmse_batch, mae_batch = rnn.module.predict(xs.to(device),
                                                                                                                            param_true.to(device),
                                                                                                                            )
                    else:
                        param_loss, rmse, mae, params_mean, samples, NGLL_batch, rmse_batch, mae_batch = rnn.predict(xs.to(device), 
                                                                                                                    param_true.to(device),
                                                                                                                    )
                        
                    samples = samples.detach().cpu().numpy()
                    param_true = param_true.detach().cpu().numpy()
                    params_mean = params_mean.detach().cpu().numpy()

                    NGLL_batch = NGLL_batch.detach().cpu().numpy()
                    rmse_batch = rmse_batch.detach().cpu().numpy()
                    mae_batch = mae_batch.detach().cpu().numpy()

                    ngll_val += param_loss.detach().cpu().item()
                    rmse_val += rmse.detach().cpu().item()
                    mae_val += mae.detach().cpu().item()

                    if i == 0:
                        param_true_batch = param_true
                        param_mean_batch = params_mean
                    else:
                        #param_true_batch = torch.cat((param_true_batch, param_true), dim=0)
                        #param_mean_batch = torch.cat((param_mean_batch, params_mean), dim=0)
                        param_true_batch = np.concatenate((param_true_batch, param_true), axis=0)
                        param_mean_batch = np.concatenate((param_mean_batch, params_mean), axis=0)

                    if i == 0:
                        eval_lower_bound = np.quantile(samples,0.5-0.68/2.,axis=0)
                        eval_upper_bound = np.quantile(samples,0.5+0.68/2.,axis=0)
                        eval_median = np.quantile(samples,0.5,axis=0)
                    else:
                        eval_lower_bound = np.concatenate((eval_lower_bound,np.quantile(samples,0.5-0.68/2.,axis=0)),axis=0)
                        eval_upper_bound = np.concatenate((eval_upper_bound,np.quantile(samples,0.5+0.68/2.,axis=0)),axis=0)
                        eval_median = np.concatenate((eval_median,np.quantile(samples,0.5,axis=0)),axis=0)

                    if i == 0:
                        ngll_val_batch = NGLL_batch
                        rmse_val_batch = rmse_batch
                        mae_val_batch = mae_batch
                    else:
                        ngll_val_batch = np.concatenate((ngll_val_batch, NGLL_batch), axis=0)
                        rmse_val_batch = np.concatenate((rmse_val_batch, rmse_batch), axis=0)
                        mae_val_batch = np.concatenate((mae_val_batch, mae_batch), axis=0)

                    for quantile in all_quantiles:
                        for param in parameters_keys:
                            
                            #sample = param_dist.sample((num_samples,))[:, :, parameters_keys.index(param)]
                            #sample = expit(sample) # convert from [-inf, inf] to [0,1] using the expit function
                            sample_val = samples[:, :, parameters_keys.index(param)]

                            param_true_val = param_true[:, parameters_keys.index(param)]
                            quantile_L = np.quantile(sample_val, 0.5-quantile/2., axis=0)
                            quantile_H = np.quantile(sample_val, 0.5+quantile/2., axis=0)

                            value = ((quantile_L <= param_true_val) & (param_true_val <= quantile_H)).mean()

                            coverage_prob_parameters[param][quantile] += value

                for quantile in all_quantiles:
                    for param in parameters_keys:
                        coverage_prob_parameters[param][quantile] /= len(val_loader)

                plot_mean_versus_truth(param_mean_batch,param_true_batch,num_bins=20,log_mass_limit=None)
                plot_scatter_plots(param_true_batch,eval_median,eval_lower_bound,eval_upper_bound,log_mass_limit=None,max_num_samples=50)
                plot_scatter_plots(param_true_batch,eval_median,eval_lower_bound,eval_upper_bound,log_mass_limit=None,max_num_samples=100)
                plot_coverage_probability_params(all_quantiles, coverage_prob_parameters, epoch)

                # save coverage probabilities dictionary using pickle
                with open(f'{save_path}/coverage_probabilities_baseline.pkl', 'wb') as f:
                    pickle.dump(coverage_prob_parameters, f)

                ngll_train /= len(train_loader)
                rmse_train /= len(train_loader)
                mae_train /= len(train_loader)

                ngll_val /= len(val_loader)
                rmse_val /= len(val_loader)
                mae_val /= len(val_loader)

                print(f'Epoch: {epoch}, NGLL train: {ngll_train:.4f}, RMSE train: {rmse_train:.4f}, MAE train: {mae_train:.4f}')
                print(f'Epoch: {epoch}, NGLL val: {ngll_val:.4f}, RMSE val: {rmse_val:.4f}, MAE val: {mae_val:.4f}')
                print()

                # Evaluate with uncertainty

                ngll_val_median = np.median(ngll_val_batch)
                rmse_val_median = np.median(rmse_val_batch)
                mae_val_median = np.median(mae_val_batch)

                ngll_val_median_abs_dev = np.median(np.abs(ngll_val_batch - ngll_val_median)) / np.sqrt(len(ngll_val_batch))
                rmse_val_median_abs_dev = np.median(np.abs(rmse_val_batch - rmse_val_median)) / np.sqrt(len(rmse_val_batch))
                mae_val_median_abs_dev = np.median(np.abs(mae_val_batch - mae_val_median)) / np.sqrt(len(mae_val_batch))

                print(f"validation median NGLL: {ngll_val_median:.4f} +/- {ngll_val_median_abs_dev:.4f}, RMSE: {rmse_val_median:.6f} +/- {rmse_val_median_abs_dev:.6f}, MAE: {mae_val_median:.6f} +/- {mae_val_median_abs_dev:.6f}")

                ngll_val_mean = np.mean(ngll_val_batch)
                rmse_val_mean = np.mean(rmse_val_batch)
                mae_val_mean = np.mean(mae_val_batch)

                ngll_val_std = np.std(ngll_val_batch) / np.sqrt(len(ngll_val_batch))
                rmse_val_std = np.std(rmse_val_batch) / np.sqrt(len(rmse_val_batch))
                mae_val_std = np.std(mae_val_batch) / np.sqrt(len(mae_val_batch))

                print(f"validation mean NGLL: {ngll_val_mean:.4f} +/- {ngll_val_std:.4f}, RMSE: {rmse_val_mean:.6f} +/- {rmse_val_std:.6f}, MAE: {mae_val_mean:.6f} +/- {mae_val_std:.6f}") 

                ngll_train_epoch.append(ngll_train)
                rmse_train_epoch.append(rmse_train)
                mae_train_epoch.append(mae_train)
                
                ngll_val_epoch.append(ngll_val)
                rmse_val_epoch.append(rmse_val)
                mae_val_epoch.append(mae_val)

                epoch_list = np.arange(len(ngll_train_epoch))+1 # start from 1

                plt.plot(epoch_list, ngll_train_epoch, label='train')
                plt.plot(epoch_list, ngll_val_epoch, label='val')
                plt.minorticks_on()
                plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                plt.xlabel('epoch')
                plt.ylabel('NGLL')
                plt.xlim(epoch_list[0], epoch_list[-1])
                plt.legend()
                plt.savefig(f"{save_path}/ngll.pdf", bbox_inches='tight')
                plt.close()

                plt.plot(epoch_list, rmse_train_epoch, label='train')
                plt.plot(epoch_list, rmse_val_epoch, label='val')
                plt.minorticks_on()
                plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                plt.xlabel('epoch')
                plt.ylabel('RMSE')
                plt.xlim(epoch_list[0], epoch_list[-1])
                plt.legend()
                plt.savefig(f"{save_path}/rmse.pdf", bbox_inches='tight')
                plt.close()

                plt.plot(epoch_list, mae_train_epoch, label='train')
                plt.plot(epoch_list, mae_val_epoch, label='val')
                plt.minorticks_on()
                plt.tick_params(which='major',direction='in',top=True, right=True,length=tick_length_major,width=tick_width)
                plt.tick_params(which='minor',direction='in',top=True, right=True,length=tick_length_minor,width=tick_width)
                plt.xlabel('epoch')
                plt.ylabel('MAE')
                plt.xlim(epoch_list[0], epoch_list[-1])
                plt.legend()
                plt.savefig(f"{save_path}/mae.pdf", bbox_inches='tight')
                plt.close()

        if distributed:
            # synchronize the processes to make sure that all the processes have finished the epoch
            dist.barrier() 

    # save metrics to a text file
    if rank == 0:
        with open(f'{save_path}/metrics.txt', 'w') as f:
            f.write(f'ngll_train_epoch: {ngll_train_epoch}\n')
            f.write(f'rmse_train_epoch: {rmse_train_epoch}\n')
            f.write(f'mae_train_epoch: {mae_train_epoch}\n')

            f.write(f'ngll_val_epoch: {ngll_val_epoch}\n')
            f.write(f'rmse_val_epoch: {rmse_val_epoch}\n')
            f.write(f'mae_val_epoch: {mae_val_epoch}\n')

    # After the training is done, destroy the distributed process group
    if distributed:
        dist.destroy_process_group()   

if __name__ == '__main__':
    # Baseline not currently used
    baseline = False
    
    start_time = time.time()

    if not baseline:
        main()
    else:     
        print("Running the baseline model") 
        main_RNN_baseline()
    
    # pint training time in hours
    print(f"Training time: {(time.time()-start_time)/3600.0:.2f} hours")
    # print training time in days
    print(f"Training time: {(time.time()-start_time)/3600.0/24.0:.2f} days")
