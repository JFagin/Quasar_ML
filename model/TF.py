# Author Joshua Fagin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from scipy import constants as const
from astropy import constants as const_astropy
import warnings
warnings.filterwarnings("ignore")
import gc

# Can be used to debug autodiff issues, but is very slow
#torch.autograd.set_detect_anomaly(True)

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


def ISCO_size(a, autodiff=True):
    """
    returns the ISCO radius in units of r_g = GM/c^2
    
    a is the dimensionless spin parameter that ranges from -1 to 1
    
    return the ISCO radius in units of r_g = GM/c^2
    """
    # This is not differentiable at a=0
    if autodiff:
        # For autodiff reasons we require a not to be near zero
        a = a.clamp(min=-0.999, max=0.999)
        a = torch.where(torch.abs(a) < 1e-3, torch.tensor(1e-3).type(a.dtype).to(a.device), a)
    
    Z1 = 1 + (1 - torch.abs(a)**2)**(1/3.) * ((1 + torch.abs(a))**(1/3.) + (1 - torch.abs(a))**(1/3.))
    Z2 = (3 * torch.abs(a)**2 + Z1**2)**0.5
    
    return 3.0 + Z2 - torch.sign(a)* ((3 - Z1) * (3 + Z1 + 2*Z2))**0.5  


def batch_histogram(input, bins, weights=None):
    """
    Function to compute histograms of Torch tensors including batch support.

    input: Tensor of shape [B, N, features] or [N, features]
    bins: Tensor of shape [B, T] or [T]
    weights: Tensor of shape [B, N, N, features] or [N, N, features] or None

    return weighted_hist_values: Tensor of shape [B, T, features] or [T, features]
    """
    # Now bins can be different across the batch

    if len(bins.shape) == 1:
        bins = bins.unsqueeze(0)

    B, N, _ = input.shape
    T = bins.shape[1]

    if weights is None:
        features_dim = 1
        weights = torch.ones(B, N, N, features_dim, dtype=input.dtype, device=input.device)
    else:
        features_dim = weights.shape[-1]

    # Flatten and sort input
    input_flattened = input.view(B, -1)
    sorted_vals, sorted_idx = input_flattened.sort(dim=-1)

    # Compute counts of elements less than each bin edge
    counts = torch.searchsorted(sorted_vals, bins, right=False)

    # Flatten and gather weights corresponding to the sorted input
    weights_flattened = weights.view(B, N*N, features_dim)
    sorted_weights = torch.gather(weights_flattened, 1, sorted_idx.unsqueeze(-1).expand(B, N*N, features_dim))

    # Compute cumulative sums for the sorted weights
    cumsum_sorted_weights = sorted_weights.cumsum(dim=1)

    # Use the counts to find cumulative sum values at bin edges
    expanded_counts = torch.clamp(counts - 1, min=0).unsqueeze(-1).expand(B, T, features_dim)
    bin_cumsum_values = torch.gather(cumsum_sorted_weights, 1, expanded_counts)

    # Pad bin_cumsum_values with zeros at the start to compute histogram values using diff
    bin_cumsum_values_padded = torch.cat([torch.zeros(B, 1, features_dim, dtype=input.dtype, device=input.device), bin_cumsum_values], dim=1)

    # Compute histogram values as the difference between consecutive bin_cumsum_values
    weighted_hist_values = bin_cumsum_values_padded.diff(dim=1)

    return weighted_hist_values

def interpolate(x, xp, fp):
    """
    One-dimensional linear interpolation for monotonically increasing sample points.

    Adapted from: https://github.com/pytorch/pytorch/issues/50334

    x: the :math:`x`-coordinates at which to evaluate the interpolated values. Shape [B, T2] or [T2]
    xp: the :math:`x`-coordinates of the data points, must be increasing. Shape [B, T1] or [T1]
    fp: the :math:`y`-coordinates of the data points, same length as `xp`. Shape [B, T1, features], [B, T1], or [T1]

    returns: the interpolated values, same size as `x`. shape [B, T2, features], [B, T2], or [T2]
    """
    assert len(xp.shape) in (1,2)
    assert len(x.shape) in (1,2)
    assert len(fp.shape) in (1,2,3)

    # Reshape inputs if necessary
    x = x if len(x.shape) == 2 else x.unsqueeze(0)
    xp = xp if len(xp.shape) == 2 else xp.unsqueeze(0)
    if len(fp.shape) == 1:
        fp = fp.unsqueeze(0).unsqueeze(-1)
    elif len(fp.shape) == 2:
        fp = fp.unsqueeze(-1)

    m = (fp[:, 1:, :] - fp[:, :-1, :]) / ((xp[:, 1:] - xp[:, :-1]).clip(min=1e-8)).unsqueeze(-1)
    b = fp[:, :-1, :] - m * xp[:, :-1].unsqueeze(-1)

    indices = torch.sum(x[:, :, None] >= xp[:, None, :], dim=-1) - 1
    indices = torch.clamp(indices, 0, m.shape[1] - 1)

    output = m[torch.arange(m.shape[0]).unsqueeze(-1), indices]*x[:, :, None] + b[torch.arange(b.shape[0]).unsqueeze(-1), indices]
    output = output.squeeze(-1)

    return output


def compute_kerr_redshift(r, phi, theta, a):
    """
    Function that computes the redshift of a photon emitted from a Kerr black hole.
    see for example: https://iopscience.iop.org/article/10.3847/1538-4357/abe305/pdf equation 26

    r: torch tensor, radius in units of r_g
    phi: torch tensor, azimuthal angle in radians
    theta: torch tensor, polar angle in radians
    a: torch tensor, dimensionless spin parameter
    """
    Sigma = r**2 + a**2 * torch.cos(theta)**2

    g_tt = -(1 - 2 * r / Sigma)
    g_phi_phi = (r**2 + a**2 + 2 * r * a**2 * torch.sin(theta)**2 / Sigma) * torch.sin(theta)**2
    g_t_phi = - 2 * r * a * torch.sin(theta)**2 / Sigma
    Omega_K = 1.0/(r**(3.0/2.0)+a).clamp(min=1e-6) 
    
    z = (1 + Omega_K * r * torch.sin(theta) * torch.sin(phi)) / torch.sqrt((-g_tt - 2 * Omega_K * g_t_phi - Omega_K**2 * g_phi_phi).clamp(min=1e-6)) - 1
    return z


def get_RC(R, Rin, r_g, a, s, M_dot, eta_x_times_f_col, eps, lamb, GR=True, plot=False):
    """
    Function that computes R_c, the characteristic radius of the disk, used to determine how far out we should compute the transfer function.

    R: torch tensor, radius in units of meters typically, shape~[B, N] 
    Rin: torch tensor, inner radius of the disk in units of meters typically, shape~[B, 1]
    r_g: torch tensor, gravitational radius in meters, shape~[B, 1]
    a: torch tensor, dimensionless spin parameter, must be between -1 and 1, shape~[B, 1]
    s: torch tensor, power law index for the accretion rate and temperature profile, shape~[B, 1]
    M_dot: torch tensor, mass accretion rate in Msun/s, shape~[B, 1]
    eta_x_times_f_col: torch tensor, determines the lampost strength, shape~[B, 1]
    eps: torch tensor, height of the disk above the corona in units of r_g, shape~[B, 1]
    lamb: torch tensor, wavelength in meters typically, shape~[B, num_bands]
    GR: bool, whether to use general relativity or not, default is True since why not
    plot: bool, whether to plot the temperature profile or not, default is False, only for debugging

    returns: normalization factor of the power law~[B, 1], characteristic radius of the disk~[B, 1]
    """
    coeff = 3.0*c**2/(8.0*pi*sigma_sb*r_g) * (Msun/r_g) * M_dot

    if GR:
        x = torch.sqrt((R/r_g).clamp(min=1e-6))
        x0 = torch.sqrt(Rin/r_g)

        x1 = 2*torch.cos(1.0/3.0*torch.arccos(a)-pi/3)
        x2 = 2*torch.cos(1.0/3.0*torch.arccos(a)+pi/3)
        x3 = -2*torch.cos(1.0/3.0*torch.arccos(a))

        T4_visc = coeff / (x**7-3*x**5+2*a*x**4).clamp(min=1e-6) * (x-x0-(3.0/2.0)*a*torch.log((x/x0).clamp(min=1e-6)) \
                                            - 3*(x1-a)**2/(x1*(x1-x2)*(x1-x3)) * torch.log(((x-x1)/(x0-x1)).clamp(min=1e-6)) \
                                            - 3*(x2-a)**2/(x2*(x2-x1)*(x2-x3)) * torch.log(((x-x2)/(x0-x2)).clamp(min=1e-6)) \
                                            - 3*(x3-a)**2/(x3*(x3-x1)*(x3-x2)) * torch.log(((x-x3)/(x0-x3)).clamp(min=1e-6)))
        T4_visc[R<Rin] = 1e-6
    else:
        T4_visc = coeff * (1.0 - torch.sqrt(Rin/R)) * (r_g/R)**3.0 
        T4_visc[R<Rin] = 1e-6
    power_norm =  (torch.sum(T4_visc * (R/r_g).clamp(min=1e-6), dim=1) / torch.sum(T4_visc * (R/r_g).clamp(min=1e-6)*((R/r_g).clamp(min=1e-6,max=1e6))**s, dim=1)).unsqueeze(-1)
    T4_visc = power_norm * ((R/r_g).clamp(min=1e-6,max=1e6))**s * T4_visc
    T4_visc[R<Rin] = 1e-6

    #T4_lamp = (1.0-albedo)*coeff*(4.0/3.0)*eta_x*eps*((R/r_g)**2+eps**2)**(-3.0/2.0)
    T4_lamp = eta_x_times_f_col*coeff*(4.0/3.0)*eps*((R/r_g)**2+eps**2)**(-3.0/2.0)
    T4_lamp[R<Rin] = 1e-6 # Cannot be zero for automatic differentiation

    Teff = (T4_visc + T4_lamp)**0.25

    # Find R_c when k_B*Teff(R_c) = h*c/lamb
    diff = torch.abs(Teff - h*c/k_B/lamb)

    diff[R < 1.5*Rin] = 1e8

    min_idx = torch.argmin(diff, dim=1)
    R_c = torch.gather(R, 1, min_idx.unsqueeze(-1))

    if plot:
        plt.loglog(R[0].squeeze().detach().cpu().numpy(),Teff[0].squeeze().detach().cpu().numpy(),ls='-' ,color='black',label=r'$T_0$')
        plt.loglog(R[0].squeeze().detach().cpu().numpy(),(T4_visc[0].squeeze()**0.25).detach().cpu().numpy(),ls='--' ,color='r',label=r'$T_{\rm visc}$')
        plt.loglog(R[0].squeeze().detach().cpu().numpy(),(T4_lamp[0].squeeze()**0.25).detach().cpu().numpy(),ls='--',color='g',label=r'$T_{\rm lamp}$')
        plt.axvline(R_c[0].squeeze().detach().cpu().numpy(),ls='--',color='k',label=r'$R_c$')
        plt.axvline(Rin[0].squeeze().detach().cpu().numpy(),ls='--',color='b',label=r'$R_{\rm ISCO}$')
        plt.axhline((h*c/k_B/lamb)[0].squeeze().detach().cpu().numpy(),ls='--',color='m',label=r'$T_{\rm obs}$')
        plt.legend()
        plt.show()

    return power_norm, R_c

def generate_tf(params, lamb, kernel_num_days, kernel_resolution, parameters_keys, dtype=torch.float32, plot=False, GR=True):
    """
    Main function that generates the transfer function for a given set of parameters.
    Can also be used to just get the continuum spectrum of the quasar if just_get_spectrum is True.

    parameters: torch tensor, shape~[B, num_params], accretion disk / black hole parameters to generate the transfer functions
    lamb: torch tensor, shape~[B, num_bands], wavelength in Angstrom
    kernel_num_days: int, maximum time of the transfer function in days
    kernel_resolution: float, resolution of the transfer function in days
    parameters_keys: list of strings, keys for the parameters
    dtype: torch dtype, default is torch.float32
    plot: bool, whether to plot the transfer functions or not, default is False, used for debugging
    GR: bool, whether to use general relativity or not, default is True since why not

    returns: transfer function, torch tensor, shape~[B, kernel_num_days, num_bands]
    """
    
    device = params.device

    # set dtype
    params = params.type(dtype)
    lamb   = lamb.type(dtype)

    log_mass = params[:, parameters_keys.index('log_mass')].unsqueeze(-1).unsqueeze(-1)
    # This is the log Eddington ratio divided by the radiative efficiency
    log_eddington_ratio = params[:, parameters_keys.index('log_edd')].unsqueeze(-1).unsqueeze(-1)
    #eta      = params[:, parameters_keys.index('ETA')].unsqueeze(-1).unsqueeze(-1)
    eps      = params[:, parameters_keys.index('height')].unsqueeze(-1).unsqueeze(-1)
    incl     = params[:, parameters_keys.index('theta_inc')].unsqueeze(-1).unsqueeze(-1)
    beta     = params[:, parameters_keys.index('beta')].unsqueeze(-1).unsqueeze(-1)
    a        = params[:, parameters_keys.index('spin')].unsqueeze(-1).unsqueeze(-1)
    z        = params[:, parameters_keys.index('redshift')].unsqueeze(-1).unsqueeze(-1)
    f_lamp   = params[:, parameters_keys.index('f_lamp')].unsqueeze(-1).unsqueeze(-1) # fraction of the total luminosity that is in the lamppost times f_cov

    del params

    lamb = lamb.unsqueeze(0).unsqueeze(0)

    Mbh = 10.0**log_mass     # Msun
    eddington_ratio = 10.0**log_eddington_ratio # This is the log Eddington ratio divided by the radiative efficiency!
    alpha = ISCO_size(a) # ISCO radius in units of r_g
    eps = eps + alpha # The height we measure is the height above the ISCO in units of r_g
    r_g = (G/c**2 * Msun)*Mbh
    Rin = alpha*r_g # inner radius of the disk 
    eta = 1.0-torch.sqrt(1.0-2.0/(3.0*alpha)) # radiative efficiency in the Novikov-Thorne model
    del log_mass, log_eddington_ratio

    # Radiative efficiency
    M_dot_Edd = 4*pi*G*Mbh*m_p/(sigma_T*c*eta) # in Msun/s
    M_dot = eddington_ratio*M_dot_Edd # in Msun/s 
    
    # Efficiency of the lamppost term
    eta_x_times_f_col = f_lamp * eta / eddington_ratio # eta_x * (1-Albedo)

    # Deviation of temperature profile slope: T ~ r^-beta, T ~ r^-(n+3)/4 
    s = 3-4*beta 

    Height = eps*r_g # height of the disk in units of r_g
    
    # cosmological redshift
    lamb = AA*lamb/(1.0+z) 
    Nsrc = 1000  
    
    # go out to 10,000 R_g
    R_radius1D = r_g*torch.linspace(0.0001, 10000, 100000, dtype=dtype, device=device).unsqueeze(0).unsqueeze(2)
    power_norm, R_c = get_RC(R_radius1D, Rin, r_g, a, s, M_dot, eta_x_times_f_col, eps, lamb[:, :, -1].unsqueeze(-1), GR=GR, plot=plot) # Use the reddest band
    del R_radius1D
    pix_scale = 100*R_c/Nsrc
   
    ############################
    ti_tau = 0                      # [day]
    dt_tau = kernel_resolution      # [day], resolution of the transfer function
    tf_tau = kernel_num_days        # [day], maximum time of the transfer function
    ############################
    
    theta_inc = incl*pi/180.0

    x_edges = torch.linspace(-Nsrc/2,Nsrc/2,Nsrc+1, dtype=dtype, device=device)
    y_edges = torch.linspace(-Nsrc/2,Nsrc/2,Nsrc+1, dtype=dtype, device=device)

    x_bins  = 0.5 * (x_edges[1:] + x_edges[:-1])
    y_bins  = 0.5 * (y_edges[1:] + y_edges[:-1])
    dx_bins = x_edges[1:]-x_edges[:-1]
    dy_bins = y_edges[1:]-y_edges[:-1]

    del x_edges, y_edges

    x_bins_mesh, y_bins_mesh = torch.meshgrid(x_bins, y_bins)
    dx_bins_mesh, dy_bins_mesh = torch.meshgrid(dx_bins, dy_bins)

    del x_bins, y_bins, dx_bins, dy_bins

    x_bins_mesh = x_bins_mesh.unsqueeze(0)
    y_bins_mesh = y_bins_mesh.unsqueeze(0)
    dx_bins_mesh = dx_bins_mesh.unsqueeze(0)
    dy_bins_mesh = dy_bins_mesh.unsqueeze(0)
    
    R_radius2D = ( x_bins_mesh**2 + y_bins_mesh**2 )**0.5

    dA_area2D  = dx_bins_mesh*dy_bins_mesh
    phi2D  = torch.arctan2(y_bins_mesh,x_bins_mesh)

    del x_bins_mesh, y_bins_mesh, dx_bins_mesh, dy_bins_mesh
    gc.collect()

    R_radius2D = R_radius2D*pix_scale
    dA_area2D  = dA_area2D*pix_scale**2

    ISCO_inx = R_radius2D<=Rin

    lamb = lamb.unsqueeze(1)
    
    # Gravitational redshift and Doppler shift
    if GR:
        z_kerr = compute_kerr_redshift(R_radius2D/r_g, phi2D, theta_inc, a)
        z_kerr = z_kerr.unsqueeze(-1)
        z_kerr[ISCO_inx] = 1e-6
    else:
        z_kerr = 0.0

    dOmega = dA_area2D*torch.cos(theta_inc)

    del dA_area2D

    tau_time2D = ((R_radius2D**2+Height**2)**0.5 + Height*torch.cos(theta_inc) - R_radius2D*torch.cos(phi2D)*torch.sin(theta_inc))/ c

    del phi2D
    # convert to days for the kernel output
    tau_time2D = tau_time2D/seconds_per_day 
    # redshift the time
    tau_time2D = tau_time2D*(1.0+z) 
    # Shift the time so it starts at zero. The start time of the transfer function is arbitrary for our purposes
    tau_time2D = tau_time2D-(torch.min(tau_time2D.reshape(tau_time2D.shape[0],tau_time2D.shape[1]*tau_time2D.shape[2]), dim=(1)).values).unsqueeze(1).unsqueeze(1)  #- dt_tau/2 -dt_tau ?? IDK

    coeff = 3.0*c**2/(8.0*pi*sigma_sb*r_g) * (Msun/r_g) * M_dot

    if GR:
        x = torch.sqrt((R_radius2D/r_g).clamp(min=1e-6))
        x0 = torch.sqrt(Rin/r_g)

        x1 = 2*torch.cos(1.0/3.0*torch.arccos(a)-pi/3)
        x2 = 2*torch.cos(1.0/3.0*torch.arccos(a)+pi/3)
        x3 = -2*torch.cos(1.0/3.0*torch.arccos(a))

        T4_visc = coeff / (x**7-3*x**5+2*a*x**4).clamp(min=1e-6) * (x-x0-(3.0/2.0)*a*torch.log((x/x0).clamp(min=1e-6)) \
                                            - 3*(x1-a)**2/(x1*(x1-x2)*(x1-x3)) * torch.log(((x-x1)/(x0-x1)).clamp(min=1e-6)) \
                                            - 3*(x2-a)**2/(x2*(x2-x1)*(x2-x3)) * torch.log(((x-x2)/(x0-x2)).clamp(min=1e-6)) \
                                            - 3*(x3-a)**2/(x3*(x3-x1)*(x3-x2)) * torch.log(((x-x3)/(x0-x3)).clamp(min=1e-6)))
        del x, x0, x1, x2, x3

        T4_visc = power_norm * ((R_radius2D/r_g).clamp(min=1e-6,max=1e6))**s * T4_visc
        T4_visc[ISCO_inx] = 1e-6 # Cannot be zero for automatic differentiation
    else:
        T4_visc = coeff * (1.0 - torch.sqrt(Rin/R_radius2D)) * (r_g/R_radius2D)**3.0 
        T4_visc = power_norm * ((R_radius2D/r_g).clamp(min=1e-6,max=1e6))**s * T4_visc
        T4_visc[ISCO_inx] = 1e-6 # Cannot be zero for automatic differentiation

    T4_lamp = eta_x_times_f_col*coeff*(4.0/3.0)*eps*((R_radius2D/r_g)**2+eps**2)**(-3.0/2.0)
    T4_lamp[ISCO_inx] = 1e-6 # Cannot be zero for automatic differentiation

    T0_temp =  (T4_visc + T4_lamp)**0.25

    if plot:
        cmap = 'Reds'
        from matplotlib.colors import LogNorm
        norm = LogNorm()
        label = r'$T_0(R)$ [K]'
        data = T0_temp[0].clone().detach().cpu().numpy()
        data[ISCO_inx[0]] = np.nan
        pix_scale_np = pix_scale[0,0,0].clone().detach().cpu().numpy() / r_g[0,0,0].clone().detach().cpu().numpy()
        plt.imshow(data,origin='lower',cmap=cmap, extent=[-Nsrc/2*pix_scale_np, Nsrc/2*pix_scale_np, -Nsrc/2*pix_scale_np, Nsrc/2*pix_scale_np])
        plt.xlim(-300, 300)
        plt.ylim(-300, 300)
        plt.colorbar(label=label)
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=7,width=1)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=4,width=1)
        plt.savefig('example_temperature_map.pdf', bbox_inches='tight')
        plt.show()

    Ratio_T4 = T4_lamp/(T4_visc+T4_lamp)
    Ratio_T4[ISCO_inx] = 1e-6 # Cannot be zero for automatic differentiation

    if plot:
        num = 0
        R_radius1D = R_radius2D[num, int(Nsrc/2),int(Nsrc/2):]

        plt.plot((R_radius1D/r_g[num].squeeze()).detach().cpu().numpy(),T0_temp[num, int(Nsrc/2),int(Nsrc/2):].detach().cpu().numpy(),ls='-' ,color='black',label=r'$T_0$')

        plt.plot((R_radius1D/r_g[num].squeeze()).detach().cpu().numpy(),(T4_visc[num, int(Nsrc/2),int(Nsrc/2):]**0.25).detach().cpu().numpy(),ls='--' ,color='r',label=r'$T_{\rm visc}$')

        plt.plot((R_radius1D/r_g[num].squeeze()).detach().cpu().numpy(),(T4_lamp[num, int(Nsrc/2),int(Nsrc/2):]**0.25).detach().cpu().numpy(),ls='--',color='g',label=r'$T_{\rm lamp}$')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    del T4_visc, T4_lamp, Height, eps, coeff
    gc.collect()    

    xi0 = h*c/k_B/(lamb*T0_temp.unsqueeze(-1)) 
    xi0[ISCO_inx.unsqueeze(-1).repeat(1,1,1,xi0.shape[-1])] = 1e-6

    dB_dT = (2*h*c/lamb)/lamb/lamb/T0_temp.unsqueeze(-1) * xi0 / (2*torch.sinh((xi0/2 * (1.0+z_kerr)).clamp(max=85.0)))**2  # The GR beaming and redshift of lambda cancels out excep in the sinh term

    dB_dT[ISCO_inx.unsqueeze(-1).repeat(1,1,1,xi0.shape[-1])] = 1e-8

    del ISCO_inx, xi0
    gc.collect()    
    
    dB0_nu = dB_dT * T0_temp.unsqueeze(-1) /4 * Ratio_T4.unsqueeze(-1)
    del dB_dT, T0_temp, Ratio_T4
    gc.collect()    

    dflux0_nu = torch.sum(dB0_nu*dOmega.unsqueeze(-1), dim=(1,2), keepdim=True)

    Weight = dB0_nu*dOmega.unsqueeze(-1)/dflux0_nu

    # Only consider the pixels that are within the outer radius we choose, even though we use a square grid.
    Weight[R_radius2D.unsqueeze(-1).repeat(1,1,1,Weight.shape[-1]) > Nsrc/2*pix_scale.unsqueeze(-1)] = 0.0

    if plot:


        cmap = 'Greens'
        from matplotlib.colors import LogNorm
        norm = LogNorm()
        label = r'$W(R)=\Delta B_\nu\cdot {\rm d}\Omega$'
        data = (dB0_nu*dOmega.unsqueeze(-1))[0,:,:,1].clone().detach().cpu().numpy()
        #data = Weight[0,:,:,0].clone().detach().cpu().numpy()
        pix_scale_np = pix_scale[0,0,0].clone().detach().cpu().numpy() / r_g[0,0,0].clone().detach().cpu().numpy()
        plt.imshow(data,origin='lower',cmap=cmap, extent=[-Nsrc/2*pix_scale_np, Nsrc/2*pix_scale_np, -Nsrc/2*pix_scale_np, Nsrc/2*pix_scale_np])
        plt.xlim(-200, 200)
        plt.ylim(-200, 200)
        plt.colorbar(label=label)
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=7,width=1)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=4,width=1)
        plt.xlabel(r'$x$ [$R_g$]')
        plt.ylabel(r'$y$ [$R_g$]')
        plt.title(r'Weight map $g$-band')
        plt.savefig('example_weight_map.pdf', bbox_inches='tight')
        plt.show()

        del data
    
    del dB0_nu, dflux0_nu, dOmega
    gc.collect()

    # We do interpolation to make the transfer function smoother
    smooth=True
    if smooth:
        tau_time2D_clone = tau_time2D.clone()
        tau_time2D_clone[R_radius2D > Nsrc/2*pix_scale] = 0.0
        tau_time2D_clone = tau_time2D_clone.reshape(tau_time2D_clone.shape[0],tau_time2D_clone.shape[1]*tau_time2D_clone.shape[2])
        max_tau = (torch.max(tau_time2D_clone, dim=1).values).unsqueeze(1).unsqueeze(1)


        del tau_time2D_clone

        # larger will give smoother transfer function but can increase the time lags artificially
        dtau = 2.0*pix_scale*(1.0+z)/seconds_per_day/c  

        tau_time2D = tau_time2D / max_tau
        
        N_tau_max = Nsrc

        bins = N_tau_max/(max_tau.squeeze(2)/dtau.squeeze(2)) * torch.linspace(0.0,1.0,N_tau_max,dtype=dtype,device=device).unsqueeze(0) #.repeat(Psi_nu.shape[0],1)

        Psi_nu = batch_histogram(tau_time2D,
                                            bins=bins,
                                            weights=Weight)

        Psi_nu[:, 0] = 0.0*Psi_nu[:, 0]

        Psi_nu = interpolate(torch.linspace(ti_tau,tf_tau,int((tf_tau-ti_tau)/dt_tau),dtype=dtype,device=device),
                                            max_tau.squeeze(2)*bins, 
                                            Psi_nu)

        Psi_nu[:, 1] = Psi_nu[:, 1] + 1e-7

    else:
        Psi_nu = batch_histogram(tau_time2D,
                                bins=torch.linspace(ti_tau,tf_tau,int((tf_tau-ti_tau)/dt_tau),dtype=dtype,device=device),
                                weights=Weight)

    # Make sure the transfer function is normalized such that it sums to 1 for each band
    Psi_nu = Psi_nu / torch.sum(Psi_nu, dim=1, keepdim=True)
    
    if plot:
        tau_avg = torch.sum(Psi_nu*torch.linspace(ti_tau,tf_tau,int((tf_tau-ti_tau)/dt_tau),dtype=dtype,device=device).unsqueeze(0).unsqueeze(-1), dim=1)
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
            label = f'{bands[i]}-band, '+r'$\tau_{\rm avg}=%.2f$ d'%tau_avg[num,i]
            plt.plot(np.linspace(ti_tau,tf_tau,int((tf_tau-ti_tau)/dt_tau)),Psi_nu[num, :, i].detach().cpu().numpy(),label=label,lw=linewidth,color=colors[i])
            plt.axvline(x=tau_avg[num, i].detach().cpu().numpy(),ls=':'  ,lw=linewidth,color=colors[i])
        plt.xlabel(r'$\tau$ [days]',fontsize=14)
        plt.ylabel(r'$\Psi$',fontsize=14)
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

        plt.figure(figsize=(8,6))

        linewidth = 2
        xscale = 'linear'
        yscale = 'linear'
        xmin,xmax = 0,500
        ymin,ymax = 0,0.05  
        bands = ['u','g','r','i','z','y']
        colors = ['b','g','r','orange','m','k']
        for i in range(Psi_nu.shape[-1]):
            label = f'${bands[i]}$-band, '+r'$\tau_{\rm avg}=%.2f$ d'%tau_avg[num,i]
            plt.plot(np.linspace(ti_tau,tf_tau,int((tf_tau-ti_tau)/dt_tau)),Psi_nu[num, :, i].detach().cpu().numpy(),label=label,lw=linewidth,color=colors[i])
            plt.axvline(x=tau_avg[num, i].detach().cpu().numpy(),ls=':'  ,lw=linewidth,color=colors[i])
        plt.xlabel(r'$\tau$ [days]',fontsize=14)
        plt.ylabel(r'$\Psi$',fontsize=14)
        plt.xlim(ti_tau,20)
        plt.ylim(ymin)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.legend(loc=1,fontsize=14)
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=7,width=1)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=3,width=1)
        plt.tight_layout()
        plt.savefig('example_transfer_function.pdf', bbox_inches='tight')
        plt.close()

        # plot with wavelength Axis instead of bands
        plt.figure(figsize=(8,6))
        xscale = 'linear'
        yscale = 'linear'
        xmin,xmax = 0,500
        ymin,ymax = 0,0.05
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
        wavelengths = [2000, 5000, 5500, 7000, 8500, 9500]
        for i in range(Psi_nu.shape[-1]):
            label = r'$\lambda=%.f$ $\mathrm{\AA}$'%wavelengths[i]
            plt.plot(np.linspace(ti_tau,tf_tau,int((tf_tau-ti_tau)/dt_tau)),Psi_nu[num, :, i].detach().cpu().numpy(),label=label,lw=linewidth, color=colors[i])
            plt.axvline(x=tau_avg[num, i].detach().cpu().numpy(),ls=':'  ,lw=linewidth,color=colors[i])
        plt.xlabel(r'$\tau$ [days]',fontsize=14)
        plt.ylabel(r'$\Psi$',fontsize=14)
        plt.xlim(0.0,10.0)
        plt.ylim(ymin)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.legend(loc=1,fontsize=14)
        plt.minorticks_on()
        plt.tick_params(which='major',direction='in',top=True, right=True,length=7,width=1)
        plt.tick_params(which='minor',direction='in',top=True, right=True,length=3,width=1)
        plt.tight_layout()
        plt.savefig('example_transfer_function_wavelength.pdf', bbox_inches='tight')
        plt.close()

    return Psi_nu

class GenerateTFModule(nn.Module):
    """
    Class to put the generate_tf function into a torch module. Neater to add to the neural network.
    """
    def __init__(self, parameters_keys, lamb, kernel_num_days, kernel_resolution):
        """
        parameters_keys: list of strings, keys for the parameters
        lamb: torch tensor, shape~[B, num_bands], wavelength in Angstrom
        kernel_num_days: int, maximum time of the transfer function in days
        kernel_resolution: float, resolution of the transfer function in days
        """
        super(GenerateTFModule, self).__init__()
        self.parameters_keys = parameters_keys
        self.lamb = lamb
        self.kernel_num_days = kernel_num_days
        self.kernel_resolution = kernel_resolution

    def forward(self, params, batch_fraction_sim=1.0):
        """
        params: torch tensor, shape~[B, num_params], accretion disk / black hole parameters to generate the transfer functions
        batch_fraction_sim: float, fraction of the batch to simulate at once, default is 1.0

        returns: transfer function, torch tensor, shape~[B, kernel_num_days, num_bands]
        """
        assert batch_fraction_sim <= 1.0 and batch_fraction_sim > 0.0, 'batch_fraction must be between 0 and 1'

        with autocast(enabled=False):
            if batch_fraction_sim == 1.0:
                return generate_tf(params, self.lamb, self.kernel_num_days, self.kernel_resolution, self.parameters_keys, plot=False)
            else:
                num_samples = params.size(0)
                split_size = max(int(batch_fraction_sim * num_samples),1)
                
                results_list = []
                for start_idx in range(0, num_samples, split_size):
                    end_idx = min(start_idx + split_size, num_samples)
                    smaller_batch = params[start_idx:end_idx]
                    partial_result = generate_tf(smaller_batch, self.lamb, self.kernel_num_days, self.kernel_resolution, self.parameters_keys, plot=False)
                    results_list.append(partial_result)

                # Concatenate the results to get the full batch result
                results = torch.cat(results_list, dim=0)
                
                return results

if __name__ == '__main__':
    # Just for testing the code

    do = False
    if do:
        from tqdm import tqdm
        from time import time

        log_mass = 10.0  # log10(Mbh/Msun)
        log_eddington_ratio = -1. # Eddington ratio per radiative efficiency
        f_lamp = 0.005 # Strength of the lamppost term f_lamp = (1-A)*L_x/L_Edd
        eps = 10.0        # coronal height in units of r_g
        incl = 70.0        # inclination angle in degrees
        beta = 0.8    # temperature slope, 3/4 for Shakura-Sunyaev disk
        a = 0.99           # spin parameter
        z = 2.1         # redshift

        batch_size = 1

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        lambda_effective = np.array([3671, 4827, 6223, 7546, 8691, 9712]) #in Angstrom
        #lambda_effective = np.array([10000, 10000]) #in Angstrom

        log_mass = torch.tensor(log_mass).repeat(batch_size).to(device)
        log_eddington_ratio = torch.tensor(log_eddington_ratio).to(device)
        f_lamp = torch.tensor(f_lamp).to(device)
        eps = torch.tensor(eps).repeat(batch_size).to(device)
        incl = torch.tensor(incl).repeat(batch_size).to(device)
        beta = torch.tensor(beta).repeat(batch_size).to(device)
        a = torch.tensor(a).repeat(batch_size).to(device)
        z = torch.tensor(z).repeat(batch_size).to(device)

        parameters_keys = ['spin','log_edd','f_lamp','height','theta_inc','redshift','beta','log_mass', 'log_nu_b', 'alpha_L', 'alpha_H_minus_L', 'standard_deviation'] 

        params = torch.zeros(batch_size, len(parameters_keys)).to(device)
        params[:,0] = a
        params[:,1] = log_eddington_ratio
        params[:,2] = f_lamp
        params[:,3] = eps
        params[:,4] = incl
        params[:,5] = z
        params[:,6] = beta
        params[:,7] = log_mass

        # require gradient for all parameters
        params.requires_grad = True

        lamb = torch.tensor(lambda_effective).to(device)

        #print('compiling...')
        #generate_tf = torch.compile(generate_tf)
        back_prop = False
        t0 = time()

        for i in tqdm(range(10_000)):
            tf = generate_tf(params,
                                lamb,
                                kernel_num_days=800,
                                kernel_resolution=1.0,
                                parameters_keys=parameters_keys,
                                GR=True,
                                plot=False)  
            # Entropy loss, this is just a random example loss function to test the gradients work
            if back_prop:
                loss = torch.sum(tf*torch.log(tf+1e-6), dim=1).mean() 
                with torch.autograd.detect_anomaly():
                    loss.backward(retain_graph=True)
                print(params.grad)
        print('Time taken: %.2f s'%(time()-t0))


    do = True
    if do:

        from time import time
        from tqdm import tqdm

        for i in tqdm(range(10_000)):
            #log_mass = 10.0  # log10(Mbh/Msun)
            #log_eddington_ratio = -0.5 # Eddington ratio per radiative efficiency
            #f_lamp = 0.005 # Strength of the lamppost term f_lamp = (1-A)*L_x/L_Edd
            #eps = 20.0        # coronal height in units of r_g
            #incl = 30.0        # inclination angle in degrees
            #beta = 0.8    # temperature slope, 3/4 for Shakura-Sunyaev disk
            #a = 0.5           # spin parameter
            #z = 3.0         # redshift

            log_mass = np.random.uniform(7.0, 10.0)
            log_eddington_ratio = np.random.uniform(-2.0, 0.0)
            f_lamp = np.random.uniform(0.002,0.007)
            eps = np.random.uniform(0.0,40.0)
            incl = np.random.uniform(0.0, 70.0)
            beta = np.random.uniform(0.5,1.0)
            a = np.random.uniform(-1.0,1.0)
            z = np.random.uniform(0.1,5.0)

            batch_size = 1

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            lambda_effective = np.array([3671, 4827, 6223, 7546, 8691, 9712]) #in Angstrom
            #lambda_effective = np.array([2000, 5000]) #in Angstrom

            log_mass = torch.tensor(log_mass).repeat(batch_size).to(device)
            log_eddington_ratio = torch.tensor(log_eddington_ratio).to(device)
            f_lamp = torch.tensor(f_lamp).to(device)
            eps = torch.tensor(eps).repeat(batch_size).to(device)
            incl = torch.tensor(incl).repeat(batch_size).to(device)
            beta = torch.tensor(beta).repeat(batch_size).to(device)
            a = torch.tensor(a).repeat(batch_size).to(device)
            z = torch.tensor(z).repeat(batch_size).to(device)

            parameters_keys = ['spin','log_edd','f_lamp','height','theta_inc','redshift','beta','log_mass', 'log_nu_b', 'alpha_L', 'alpha_H_minus_L', 'standard_deviation'] 

            params = torch.zeros(batch_size, len(parameters_keys)).to(device)
            params[:,0] = a
            params[:,1] = log_eddington_ratio
            params[:,2] = f_lamp
            params[:,3] = eps
            params[:,4] = incl
            params[:,5] = z
            params[:,6] = beta
            params[:,7] = log_mass


            #print('params')
            #print(params)

            # require gradient for all parameters
            params.requires_grad = True

            lamb = torch.tensor(lambda_effective).to(device)

            tf = generate_tf(params,
                                lamb,
                                kernel_num_days=800,
                                kernel_resolution=1.0,
                                parameters_keys=parameters_keys,
                                plot=False)  
            
            # Entropy loss, this is just a random example loss function to test the gradients work
            loss = torch.sum(tf*torch.log(tf+1e-6), dim=1).mean() 

            try:
                with torch.autograd.detect_anomaly():
                    loss.backward(retain_graph=True)
            except:
                print('Failed')
                print(params)
                
                for i in range(tf.shape[-1]):
                    plt.plot(np.linspace(0,800,800), tf[0, :, i].detach().cpu().numpy())
                plt.xlim(0,800)
                plt.ylim(0)
                plt.show()


                break
