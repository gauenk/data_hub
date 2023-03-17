"""

Parse the options from the cfg

"""

from easydict import EasyDict as edict
from .common import optional,get_isize

def parse_cfg(cfg,modes,fields):
    parse = edict({m:edict() for m in modes})
    for field,default in fields.items():
        def_field = optional(cfg,field,default)
        def_field = default_xform(field,def_field)
        for mode in modes:
            val = optional(cfg,"%s_%s"%(field,mode),def_field)
            parse[mode][field] = post_xform(field,val)
    return parse

def default_xform(field,default):
    if field == "isize":
        if default == "-1_-1": default = None
    return default

def post_xform(field,value):
    if field == "isize":
        return get_isize(value)
    else:
        return value


"""
    # -- bw --
    def_bw = optional(cfg,"bw",False)
    bw = edict()
    for mode in modes:
        bw[mode] = optional(cfg,"bw_%s"%mode,def_bw)

    # -- frames --
    def_nframes = optional(cfg,"nframes",0)
    nframes = edict()
    for mode in modes:
        nframes[mode] = optional(cfg,"nframes_%s"%mode,def_nframes)

    # -- fstride [amount of overlap for subbursts] --
    def_fstride = optional(cfg,"fstride",1)
    fstride = edict()
    for mode in modes:
        fstride[mode] = optional(cfg,"%s_fstride"%mode,def_fstride)

    # -- frame sizes --
    def_isize = optional(cfg,"isize",None)
    if def_isize == "-1_-1": def_size = None
    isizes = edict()
    for mode in modes:
        isizes[mode] = get_isize(optional(cfg,"isize_%s"%mode,def_isize))

    # -- samples --
    def_nsamples = optional(cfg,"nsamples",-1)
    nsamples = edict()
    for mode in modes:
        nsamples[mode] = optional(cfg,"nsamples_%s"%mode,def_nsamples)

    # -- crop mode --
    def_cropmode = optional(cfg,"cropmode","region")
    cropmode = edict()
    for mode in modes:
        cropmode[mode] = optional(cfg,"cropmode_%s"%mode,def_cropmode)

    # -- random order --
    def_rand_order = optional(cfg,'rand_order',False)
    rand_order = edict()
    for mode in modes:
        rand_order[mode] = optional(cfg,"rand_order_%s"%mode,def_rand_order)

    # -- skipping index [for testing mostly] --
    def_index_skip = optional(cfg,'index_skip',1)
    index_skip = edict()
    for mode in modes:
        index_skip[mode] = optional(cfg,"index_skip_%s"%mode,def_index_skip)

    # -- batch size --
    def_batch_size = optional(cfg,'batch_size',1)
    batch_size = edict()
    for mode in modes:
        batch_size[mode] = optional(cfg,'batch_size_%s'%mode,def_batch_size)
"""
