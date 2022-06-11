
# -- misc --
import copy
from easydict import EasyDict as edict

# -- torch imports --
from torchvision import transforms as tvT

# -- project imports --
from data_hub.common import optional
from .impl import AddGaussianNoiseSetN2N,GaussianBlur,AddGaussianNoise,AddPoissonNoiseBW,AddLowLightNoiseBW,AddHeteroGaussianNoise,ScaleZeroMean,QIS

__all__ = ['get_noise_transform']


def get_noise_transform(noise_info,noise_only=False,
                        use_to_tensor=True,zero_mean=True):
    """
    The exemplar function for noise getting info
    """
    # -- get transforms --
    to_tensor = tvT.ToTensor()
    szm = ScaleZeroMean()
    noise = choose_noise_transform(noise_info)

    # -- create composition --
    comp = []
    if noise_only: comp = [noise]
    else:
        if use_to_tensor: comp += [to_tensor]
        comp += [noise]
        if zero_mean: comp += [szm]
    transform = tvT.Compose(comp)

    return transform

def choose_noise_transform(noise_info, verbose=False):
    if verbose: print("[data_hub/transforms/parse_noise_info.py]: ",noise_info)
    ntype = noise_info.ntype
    if ntype == "g":
        return get_g_noise(noise_info)
    if ntype == "hg":
        return get_hg_noise(noise_info)
    elif ntype == "ll":
        return get_ll_noise(noise_info)
    elif ntype == "pn":
        return get_pn_noise(noise_info)
    elif ntype == "qis":
        return get_qis_noise(noise_info)
    elif ntype == "msg":
        return get_msg_noise(noise_info)
    elif ntype == "msg_simcl":
        return get_msg_simcl_noise(noise_info)
    elif ntype in ["none","clean"]:
        def null(image): return image
        return null
    else:
        raise ValueError(f"Unknown noise_type [{ntype}]")

def get_g_noise(params):
    """
    Noise Type: Gaussian
    """
    sigma = params['sigma']
    gaussian_noise = AddGaussianNoise(0.,sigma)
    return gaussian_noise

def get_hg_noise(params):
    """
    Noise Type: Heteroskedastic Gaussian N(x, \sigma_r + \sigma_s * x)
    """
    gaussian_noise = AddHeteroGaussianNoise(params['mean'],
                                            params['read'],params['shot'])
    return gaussian_noise

def get_ll_noise(params):
    """
    Noise Type: Low-Light  (LL)
    - Each N images is a low-light image with same alpha parameter
    """
    low_light_noise = LowLight(params['alpha'])
    return low_light_noise

def get_pn_noise(params):
    alpha,std = params['alpha'],params['std']
    if std > 1: std /= 255. # rescale is necessary
    pn_noise = AddLowLightNoiseBW(alpha,std,-1,False)
    return pn_noise

def get_qis_noise(params):
    # alpha,readout = params['alpha'],params['readout']
    # nbits,use_adc = params['nbits'],params['use_adc']
    alpha,readout = params['alpha'],params['readout']
    nbits = params['nbits']
    qis_noise = QIS(alpha,read_noise,nbits)
    # qis_noise = AddLowLightNoiseBW(alpha,readout,nbits,use_adc)
    return qis_noise

def get_msg_noise(params):
    """
    Noise Type: Multi-scale Gaussian  (MSG)
    - Each N images has it's own noise level
    """
    std_range = (params['std_min'],params['std_max'])
    gaussian_n2n = AddGaussianNoiseSetN2N(params['N'],std_range)
    return gaussian_n2n

# --------------------------------
#
#    From String to Config
#
# --------------------------------

def noise_from_cfg(cfg):
    ns = edict()
    ns.sigma = cfg.sigma
    ntype = optional(cfg,'ntype','g')
    ns.ntype = ntype
    # -- additional fields --
    if ntype == "g":
        fields = ["sigma"]
    elif ntype == "pn":
        fields = ["alpha"]
    elif ntype == "qis":
        fields = QIS.fields
    else:
        raise ValueError(f"Uknown noisy type [{ntype}] for fields")

    # -- assignment --
    for field in fields:
        ns[field] = cfg[field]

    return ns

def get_noise_config(name):
    if name.split("-")[0] == "g": # gaussian
        config = get_gaussian_config_from_name(name)
    elif name.split("-")[0] == "pn":
        config = get_poisson_config_from_name(name)
    elif name.split("-")[0] == "qis":
        config = get_qis_config_from_name(name)
    else:
        raise ValueError(f"Uknown noise config [{name}]")
    return config

def get_gaussian_config_from_name(name):
    noise_type,noise_str = name.split("-")
    noise_level = float(noise_str.replace('p','.'))
    ns = edict()
    ns['ntype'] = noise_type
    ns['sigma'] = noise_level
    ns['name'] = name
    return ns

def get_poisson_config_from_name(name):
    noise_type,rate_str,std_str = name.split("-")
    rate_level = float(rate_str.replace('p','.'))
    std_level = float(std_str.replace('p','.'))
    ns = edict()
    ns['ntype'] = noise_type
    ns['alpha'] = rate_level
    ns['sigma'] = std_level
    ns['name'] = name
    return ns

def get_qis_config_from_name(name):
    noise_type,rate_str,read_str,nbits_str = name.split("-")
    rate = float(rate_str.replace('p','.'))
    read_noise = float(read_str.replace('p','.'))
    nbits = int(nbits_str)
    ns = edict()
    ns['ntype'] = noise_type
    ns['alpha'] = rate
    ns['read_noise'] = read_noise
    ns['nbits'] = nbits
    ns['name'] = name
    return ns



