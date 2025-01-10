# Authors: Joshua Fagin, James Chan

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import constants as const
from astropy import constants as const_astropy
import astropy.units as u
from time import time
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import gc

c = const.c # Speed of light, m/s
G = const.G # Gravitational constant, m^3/kg/s^2
m_p = const.m_p # Proton mass, kg
sigma_sb = const.Stefan_Boltzmann # Stefan-Boltzmann constant, W/m^2/K^4
h = const.h # Planck constant, J*s
k_B = const.k # Boltzmann constant, J/K
pc = const.parsec # Parsec, m
Msun = float(const_astropy.M_sun.value) # Solar mass, kg
sigma_T = float(const_astropy.sigma_T.value) # Thomson cross section, m^2
AA = 1e-10 # Angstrom, m
Jy = 1e-26 # Jansky, W/m^2/Hz
seconds_per_day = 60*60*24 # seconds/day
seconds_per_year = 365.*seconds_per_day # seconds/year
pi = np.pi # pi


def ISCO_size_numpy(a):
    """
    returns the ISCO radius in units of r_g = GM/c^2
    
    a is the dimensionless spin parameter that ranges from -1 to 1
    
    return the ISCO radius in units of r_g = GM/c^2
    """
    Z1 = 1 + (1 - np.abs(a)**2)**(1/3.) * ((1 + np.abs(a))**(1/3.) + (1 - np.abs(a))**(1/3.))
    Z2 = (3 * np.abs(a)**2 + Z1**2)**0.5
    
    return 3 + Z2 - np.sign(a)*((3 - Z1) * (3 + Z1 + 2*Z2))**0.5

def histogram_numpy(input, bins, weights=None):
    """
    Compute histograms for each feature in input using numpy's bincount function.

    input: 2D numpy array of shape (T, features)
    bins: 1D numpy array of shape (max_bins,)
    weights: 2D numpy array of shape (T, features), default is None

    returns: 2D numpy array of shape (max_bins, features)
    """
    # Compute the bin indices
    bin_indices = np.digitize(input.ravel(), bins, right=False) 
    
    # If weights are not provided, make them ones with the same shape as input but with the feature dimension
    if weights is None:
        weights = np.ones((*input.shape, 1), dtype=input.dtype)

    # Reshape weights to have a shape of [T, features]
    weights_reshaped = weights.reshape(-1, weights.shape[-1])
    
    # Offset bin indices for each feature using broadcasting
    max_bins = len(bins)
    offsets = np.arange(weights_reshaped.shape[1]) * max_bins
    bin_indices_offsets = bin_indices[:, None] + offsets
    
    # Compute bin counts using numpy's bincount function for each feature
    flattened_indices = bin_indices_offsets.ravel()
    flattened_weights = weights_reshaped.ravel()
    bin_counts_flat = np.bincount(flattened_indices, weights=flattened_weights, minlength=offsets[-1] + max_bins)
    
    # Reshape bin_counts_flat to separate the features
    bin_counts = bin_counts_flat.reshape(-1, max_bins)[:weights_reshaped.shape[1]]

    bin_counts = bin_counts.T

    return bin_counts


def interpolate(x, xp, fp):
    """
    Linear interpolation with an extra feature dimension.
    
    Parameters:
    - x : 1D numpy array of shape (n,)
    - xp : 1D numpy array of shape (m)
    - fp : 2D numpy array of shape (n, features) 
    
    Returns:
    - yp: 2D numpy array of shape (m, features), interpolated values at xp
    """
    ind = np.searchsorted(xp, x, side='left')
    ind = np.clip(ind, 1, len(xp)-1)
    
    x0 = np.expand_dims(xp[ind - 1], axis=-1)
    x1 = np.expand_dims(xp[ind], axis=-1)
    y0 = fp[ind - 1]
    y1 = fp[ind]

    return y0 + (np.expand_dims(x,axis=-1) - x0) * (y1 - y0) / (x1 - x0)

def compute_kerr_redshift(r, phi, theta, a):
    """
    Function that computes the redshift of a photon emitted from a Kerr black hole.
    see for example: https://iopscience.iop.org/article/10.3847/1538-4357/abe305/pdf equation 26

    r: numpy array, radius in units of r_g
    phi: numpy array, azimuthal angle in radians
    theta: numpy array, polar angle in radians
    a: float, dimensionless spin parameter
    """
    Sigma = r**2 + a**2 * np.cos(theta)**2

    g_tt = -(1 - 2 * r / Sigma)
    g_phi_phi = (r**2 + a**2 + 2 * r * a**2 * np.sin(theta)**2 / Sigma) * np.sin(theta)**2
    g_t_phi = - 2 * r * a * np.sin(theta)**2 / Sigma
    Omega_K = 1.0/(r**(3.0/2.0)+a).clip(min=1e-25) 
    
    z = (1 + Omega_K * r * np.sin(theta) * np.sin(phi)) / np.sqrt((-g_tt - 2 * Omega_K * g_t_phi - Omega_K**2 * g_phi_phi).clip(min=1e-25)) - 1
    return z

def get_RC(R, Rin, r_g, a, s, M_dot, eta_x_times_f_col, eps, lamb, GR=True):
    """
    Function that computes R_c, the characteristic radius of the disk, used to determine how far out we should compute the transfer function.

    R: numpy array, radius in units of meters typically
    Rin: float, inner radius of the disk in units of meters typically
    r_g: float, gravitational radius in meters
    a: float, dimensionless spin parameter, must be between -1 and 1
    s: float, power law index for the accretion rate and temperature profile
    M_dot: float, mass accretion rate in Msun/s
    eta_x_times_f_col: float, determines the lampost strength
    eps: float, height of the disk above the corona in units of r_g
    lamb: numpy array, wavelength in meters typically
    GR: bool, whether to use general relativity or not, default is True since why not

    returns: normalization factor of the power law, R_c in meters
    """
    coeff = 3.0*c**2/(8.0*pi*sigma_sb*r_g) * (Msun/r_g) * M_dot

    if GR:
        x = np.sqrt((R/r_g).clip(min=1e-8))
        x0 = np.sqrt(Rin/r_g)

        x1 = 2*np.cos(1.0/3.0*np.arccos(a)-pi/3)
        x2 = 2*np.cos(1.0/3.0*np.arccos(a)+pi/3)
        x3 = -2*np.cos(1.0/3.0*np.arccos(a))

        T4_visc = coeff / (x**7-3*x**5+2*a*x**4).clip(min=1e-25) * (x-x0-(3.0/2.0)*a*np.log((x/x0).clip(min=1e-25)) \
                                            - 3*(x1-a)**2/(x1*(x1-x2)*(x1-x3)) * np.log(((x-x1)/(x0-x1)).clip(min=1e-25)) \
                                            - 3*(x2-a)**2/(x2*(x2-x1)*(x2-x3)) * np.log(((x-x2)/(x0-x2)).clip(min=1e-25)) \
                                            - 3*(x3-a)**2/(x3*(x3-x1)*(x3-x2)) * np.log(((x-x3)/(x0-x3)).clip(min=1e-25)))
        T4_visc[R<Rin] = 0.0
    else:
        T4_visc = coeff * (1.0 - np.sqrt(Rin/R)) * (r_g/R)**3.0 
        T4_visc[R<Rin] = 0.0
    power_norm =  np.sum(T4_visc * (R/r_g).clip(min=1e-25)) / np.sum(T4_visc * (R/r_g).clip(min=1e-25)*((R/r_g).clip(min=1e-25,max=1e25))**s)
    T4_visc = power_norm * (R/r_g).clip(min=1e-25,max=1e25)**s * T4_visc
    T4_visc[R<Rin] = 0.0

    T4_lamp = eta_x_times_f_col*coeff*(4.0/3.0)*eps*((R/r_g)**2+eps**2)**(-3.0/2.0)
    T4_lamp[R<Rin] = 0.0 

    Teff = (T4_visc + T4_lamp)**0.25

    # Find R_c when k_B*Teff(R_c) = h*c/(lamb*AA)
    diff = np.abs(Teff - h*c/k_B/lamb)
    diff[R < 1.5*Rin] = 1e10
    min_idx = np.argmin(diff)
    R_c = R[min_idx]

    return power_norm, R_c

def generate_tf_numpy(params, lamb, kernel_num_days, kernel_resolution, parameters_keys, cosmo, GR=True, just_get_spectrum=False, plot=False):
    """
    Main function that generates the transfer function for a given set of parameters.
    Can also be used to just get the continuum spectrum of the quasar if just_get_spectrum is True.

    parameters: numpy array of shape (num_params,), accretion disk / black hole parameters to generate the transfer functions
    lamb: numpy array of shape (num_bands,), wavelength in Angstrom
    kernel_num_days: int, maximum time of the transfer function in days
    kernel_resolution: float, resolution of the transfer function in days
    parameters_keys: list of strings, keys for the parameters
    cosmo: astropy cosmology object, cosmology object to compute the luminosity distance
    GR: bool, whether to use general relativity or not, default is True since why not
    just_get_spectrum: bool, whether to just get the spectrum or not, default is False, generate the transfer functions.
    plot: bool, whether to plot the transfer functions or not, default is False, used for debugging

    if just_get_spectrum is True, the function returns the spectrum of the disk without the transfer function. shape ~ (len(lamb))
    otherwise, the function returns the transfer function. shape ~ (kernel_num_days/kernel_resolution, len(lamb))
    """

    log_mass = params[parameters_keys.index('log_mass')]
    # This is the log Eddington ratio divided by the radiative efficiency
    log_eddington_ratio = params[parameters_keys.index('log_edd')]
    eps      = params[parameters_keys.index('height')] # This is the height of the disk above the corona in units of r_g
    incl     = params[parameters_keys.index('theta_inc')]
    beta     = params[parameters_keys.index('beta')]
    a        = params[parameters_keys.index('spin')]
    z        = params[parameters_keys.index('redshift')]
    f_lamp = params[parameters_keys.index('f_lamp')]

    del params

    Mbh = 10.0**log_mass     # Msun
    eddington_ratio = 10.0**log_eddington_ratio
    del log_mass, log_eddington_ratio

    alpha = ISCO_size_numpy(a)
    eps = eps + alpha # the height of the disk above the corona

    r_g = (G/c**2 * Msun)*Mbh
    Rin = alpha*r_g  # inner radius of the disk in units of r_g
    eta = 1.0-np.sqrt(1.0-2.0/(3.0*alpha))
    M_dot_Edd = 4*pi*G*Mbh*m_p/(sigma_T*c*eta) # in Msun/s
    M_dot = eddington_ratio*M_dot_Edd # in Msun/s 
    eta_x_times_f_col = f_lamp * eta / eddington_ratio 

    s = 3-4*beta 

    # redshift the wavelength
    # cosmological redshift
    lamb = AA*lamb/(1.0+z) 

    Nsrc = 1000  

    R_radius1D = r_g*np.linspace(0, 10_000, 100_000)
    power_norm, R_c = get_RC(R_radius1D, Rin, r_g, a, s, M_dot, eta_x_times_f_col, eps, lamb[-1], GR=GR)
    del R_radius1D
    pix_scale = 100*R_c/Nsrc

    lamb = np.expand_dims(lamb, axis=(0,1)) # make it size (1,1,6)

    ##########################################################################################

    Height = eps*r_g # height of the disk in units of r_g

    ############################
    ti_tau = 0                      # [day]
    dt_tau = kernel_resolution      # [day], resolution of the transfer function
    tf_tau = kernel_num_days        # [day], maximum time of the transfer function
    ############################
    
    theta_inc = incl*pi/180.0

    x_edges = np.linspace(-Nsrc/2,Nsrc/2,Nsrc+1)
    y_edges = np.linspace(-Nsrc/2,Nsrc/2,Nsrc+1)

    x_bins  = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_bins  = 0.5 * (y_edges[1:] + y_edges[:-1])
    dx_bins = x_edges[1:]-x_edges[:-1]
    dy_bins = y_edges[1:]-y_edges[:-1]

    del x_edges, y_edges
    
    x_bins_mesh, y_bins_mesh = np.meshgrid(x_bins, y_bins)
    dx_bins_mesh, dy_bins_mesh = np.meshgrid(dx_bins, dy_bins)

    x_bins_mesh = np.expand_dims(x_bins_mesh, axis=-1)
    y_bins_mesh = np.expand_dims(y_bins_mesh, axis=-1)
    dx_bins_mesh = np.expand_dims(dx_bins_mesh, axis=-1)
    dy_bins_mesh = np.expand_dims(dy_bins_mesh, axis=-1)

    del x_bins, y_bins, dx_bins, dy_bins

    R_radius2D = ( x_bins_mesh**2 + y_bins_mesh**2 )**0.5

    dA_area2D  = dx_bins_mesh*dy_bins_mesh
    phi2D  = np.arctan2(y_bins_mesh,x_bins_mesh)

    del x_bins_mesh, y_bins_mesh, dx_bins_mesh, dy_bins_mesh
    gc.collect()

    R_radius2D = R_radius2D*pix_scale
    dA_area2D  = dA_area2D*pix_scale**2

    ISCO_inx = R_radius2D<=Rin

    if GR:
        z_kerr = compute_kerr_redshift(R_radius2D/r_g, phi2D, theta_inc, a)
        z_kerr[ISCO_inx] = 0.0
    else:
        z_kerr = 0.0
    #dOmega = dA_area2D*np.cos(theta_inc)
    # get the luminosity distance in meters
    d_L = cosmo.luminosity_distance(z).to(u.m).value
    dOmega = dA_area2D*np.cos(theta_inc) / d_L**2

    del dA_area2D

    coeff = 3.0*c**2/(8.0*pi*sigma_sb*r_g) * (Msun/r_g) * M_dot

    if GR:
        x = np.sqrt((R_radius2D/r_g).clip(min=1e-25))
        x0 = np.sqrt(Rin/r_g)

        x1 = 2*np.cos(1.0/3.0*np.arccos(a)-pi/3)
        x2 = 2*np.cos(1.0/3.0*np.arccos(a)+pi/3)
        x3 = -2*np.cos(1.0/3.0*np.arccos(a))

        T4_visc = coeff / (x**7-3*x**5+2*a*x**4).clip(min=1e-25) * (x-x0-(3.0/2.0)*a*np.log((x/x0).clip(min=1e-25)) \
                                            - 3*(x1-a)**2/(x1*(x1-x2)*(x1-x3)) * np.log(((x-x1)/(x0-x1)).clip(min=1e-25)) \
                                            - 3*(x2-a)**2/(x2*(x2-x1)*(x2-x3)) * np.log(((x-x2)/(x0-x2)).clip(min=1e-25)) \
                                            - 3*(x3-a)**2/(x3*(x3-x1)*(x3-x2)) * np.log(((x-x3)/(x0-x3)).clip(min=1e-25)))
        del x, x0, x1, x2, x3

        T4_visc = power_norm * ((R_radius2D/r_g).clip(min=1e-25,max=1e25))**s * T4_visc
        T4_visc[ISCO_inx] = 0.0 
    else:
        T4_visc = coeff * (1.0 - np.sqrt(Rin/R_radius2D)) * (r_g/R_radius2D)**3.0 
        T4_visc = power_norm * ((R_radius2D/r_g).clip(min=1e-25,max=1e25))**s * T4_visc
        T4_visc[ISCO_inx] = 0.0

    T4_lamp = eta_x_times_f_col*coeff*(4.0/3.0)*eps*((R_radius2D/r_g)**2+eps**2)**(-3.0/2.0)
    T4_lamp[ISCO_inx] = 0.0

    T0_temp =  (T4_visc + T4_lamp)**0.25

    Ratio_T4 = T4_lamp/(T4_visc+T4_lamp)
    Ratio_T4[ISCO_inx] = 0.0

    del T4_visc, T4_lamp
    gc.collect()

    #ISCO_inx = ISCO_inx.unsqueeze(-1).repeat(1,1,1,xi0.shape[-1])
    xi0 = h*c/k_B/(lamb*T0_temp) 
    xi0[ISCO_inx.repeat(xi0.shape[-1], axis=-1)] = 0.0

    if just_get_spectrum:
        # This is the spectrum of the disk without the transfer function. 
        # We can use the spectrum to see how bright the quasar should be in each broad band filter.
        # If just_get_spectrum is True, we don't need to compute the transfer function and just return the spectrum F_lambda in units of W/m^3.
        # That we we can save time and memory when we just want to get the spectrum.

        # Note that the g^4 from the relativistic beaming and redshift of lambda^5 leads to one overall factor of (1+z_kerr)
        B_lambda = (1.0+z_kerr) * 2*h*c**2/lamb**5 /(np.exp(xi0 * (1.0 + z_kerr))-1)
        B_lambda[ISCO_inx.repeat(xi0.shape[-1], axis=-1)] = 0.0

        F_lambda = np.sum(B_lambda*dOmega, axis=(0,1))

        return F_lambda

    tau_time2D = ( (R_radius2D**2+Height**2)**0.5 + Height*np.cos(theta_inc) - R_radius2D*np.cos(phi2D)*np.sin(theta_inc) )/ c

    del phi2D
    # convert the time to days
    tau_time2D = tau_time2D/seconds_per_day
    # redshift the time
    tau_time2D = tau_time2D*(1.0+z) # Do I need to redshift the time with the kerr black hole?
    # Shift the time so it starts at zero. The start time of the transfer function is arbitrary for our purposes
    tau_time2D = tau_time2D - np.min(tau_time2D)


    dB_dT = (2*h*c/lamb)/lamb/lamb/T0_temp * xi0 / (2*np.sinh(xi0/2 * (1.0+z_kerr)))**2  #The GR beaming and redshift of lambda cancels out excep in the sinh term

    dB_dT[ISCO_inx.repeat(xi0.shape[-1], axis=-1)] = 0.0

    del xi0

    dB0_nu = dB_dT * T0_temp/4 * Ratio_T4
    del dB_dT, T0_temp, Ratio_T4

    dflux0_nu = np.sum(dB0_nu*dOmega, axis=(0,1))

    Weight = dB0_nu*dOmega/dflux0_nu
    del dB0_nu, dflux0_nu
    gc.collect()

    # Only consider the pixels that are within the outer radius we choose, even though we use a square grid.
    Weight[R_radius2D.repeat(Weight.shape[-1],axis=-1) > Nsrc/2*pix_scale] = 0.0

    # This effectively integrates the weight over the 2D time axis
    
    # We do interpolation to make the transfer function smoother
    smooth = True
    if smooth:
        tau_time2D_clone = np.copy(tau_time2D)
        tau_time2D_clone[R_radius2D > Nsrc/2*pix_scale] = 0.0
        max_tau = np.max(tau_time2D_clone)
        del tau_time2D_clone

        dtau = 2.0*pix_scale*(1.0+z)/seconds_per_day/c # 2.5 is smoother

        tau_time2D = tau_time2D / max_tau
        
        N_tau_max = Nsrc

        bins = N_tau_max/(max_tau/dtau) * np.linspace(0.0,1.0,N_tau_max)

        Psi_nu = histogram_numpy(tau_time2D[:, :, 0],
                                bins=bins,
                                weights=Weight)

        Psi_nu = interpolate(np.linspace(ti_tau,tf_tau,int((tf_tau-ti_tau)/dt_tau)), 
                            max_tau*bins, 
                            Psi_nu)
    else:
        Psi_nu = histogram_numpy(tau_time2D[:,:,0], bins=np.linspace(ti_tau,tf_tau,int((tf_tau-ti_tau)/dt_tau)), weights=Weight)

    # Normalize the transfer function such that it sums to 1 for each band
    Psi_nu = Psi_nu/np.sum(Psi_nu, axis=0)

    if plot:
        tau_avg = np.sum(Psi_nu*np.expand_dims(np.linspace(ti_tau,tf_tau,int((tf_tau-ti_tau)/dt_tau)),axis=-1).repeat(Psi_nu.shape[-1],axis=-1),axis=0)
        num = 0

        plt.figure(figsize=(8,6))

        linewidth = 2
        xscale = 'linear'
        yscale = 'linear'
        xmin,xmax = 0,500
        ymin,ymax = 0,0.05  
        bands = ['u','g','r','i','z','y']
        colors = ['b','g','r','orange','m','k']
        for i in range(Psi_nu.shape[-1]):
            label = f'{bands[i]}-band, '+r'$\tau_{\rm avg}=%.2f$ d'%tau_avg[i]
            plt.plot(np.linspace(ti_tau,tf_tau,int((tf_tau-ti_tau)/dt_tau)),Psi_nu[:, i],label=label,lw=linewidth,color=colors[i])
            plt.axvline(x=tau_avg[i],ls=':'  ,lw=linewidth,color=colors[i])
        plt.xlabel(r'$\tau$ [d]',fontsize=14)
        plt.ylabel(r'$\psi_\nu$',fontsize=14)
        plt.xlim(ti_tau,tf_tau)
        plt.ylim(ymin)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.legend(loc=1,fontsize=14)
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=7,width=1)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=3,width=1)
        plt.tight_layout()
        plt.show()
 
    return Psi_nu
