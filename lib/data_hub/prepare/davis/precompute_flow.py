"""

Pre-compute the optical flow for a given set of directories

"""


# -- basics --
import dev_basics
import argparse
from dev_basics import flow
from dev_basics import vid_io
from dev_basics import flow_io
from dev_basics import metrics
from dev_basics.utils.misc import set_seed

# -- sim noise --
from data_hub.transforms.noise import choose_noise_transform

# -- helpers --
import tqdm
import copy
dcopy = copy.deepcopy
from pathlib import Path
from easydict import EasyDict as edict

# -- numpy --
import numpy as np
import torch as th

def run_exp(cfg):

    # -- noise fxn --
    sim_noise = choose_noise_transform(cfg)
    noise_str = str(cfg.noise_label)
    seed_str = "seed-%s" % cfg.seed
    name = "%s_%s" % (noise_str,seed_str)

    # -- compute paths --
    path_list  = [p for p in cfg.in_root.iterdir()]
    # print(path_list[0])
    # path_list = list(reversed(path_list))
    # print(path_list[0])
    # filter_path_list(path_list)

    # -- compute flow for each directory --
    for in_root in tqdm.tqdm(path_list):

        # -- set-up paths --
        out_root = cfg.out_root / in_root.stem
        if not(out_root.exists()):
            out_root.mkdir(parents=True)

        # -- skip if complete --
        fn = out_root / ("%s.npz" % name)
        if fn.exists(): continue
        fn = out_root / ("%s_fflow.npy" % name)
        if fn.exists(): continue

        # -- load video --
        vid = vid_io.read_video(in_root)*1.
        # vid = vid[:5,:,:128,:128].contiguous()

        # -- simulate noise --
        set_seed(cfg.seed)
        noisy = sim_noise(vid)

        # -- run flows --
        flows = flow.run(noisy,True,ftype="svnlb")
        flows = edict({k:v.type(th.half) for k,v in flows.items()})
        flow_io.save_flows(flows,out_root,name,itype="npy")

def parser():
    parser = argparse.ArgumentParser(
        prog = "Precompute optical flow.",
        description = "",epilog = 'Happy Hacking')
    parser.add_argument('--in_path',default="flows/480p",
                        help="Input Image Path")
    args = parser.parse_known_args()[0]
    args = edict(vars(args))
    return args

def run(base):

    # -- parse --
    args = parser()
    if "crop" in str(args.in_path):
        out_subdir = "cropped_flows/train"
    else:
        out_subdir = "flows/480p"

    # -- cfg --
    # base = Path(__file__).parents[0]
    cfg = edict()
    cfg.seed = 123
    cfg.in_root = base / args.in_path
    cfg.out_root = base / out_subdir
    cfg.device = "cuda:0"
    assert cfg.in_root.exists(),"Input path must exist."

    # -- noise grids --
    noise_types = ["g"]
    noise_params = {"g":{"sigma":[30,50],"noise_label":["g-30","g-50"]}}
    # noise_types = ["pg"]
    # noise_params = {"pg":{"sigma":[10,10],"rate":[30,10],
    #                       "noise_label":["pg-30","pg-10"]}}
    # g(30) <-> pg(30,10)
    # g(50) <-> pg(10,10)
    # stardeno

    # -- run over grid --
    for noise_type in noise_types:
        params = noise_params[noise_type]
        L = len(params[list(params.keys())[0]])
        for l in range(L):

            # -- set params --
            _cfg = dcopy(cfg)
            _cfg.ntype = noise_type
            for key,vals in params.items():
                _cfg[key] = vals[l]

            # -- run ins/outs --
            run_exp(_cfg)

if __name__ == "__main__":
    main()
