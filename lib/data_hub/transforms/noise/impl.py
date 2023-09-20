# -- python imports --
import uuid,time
import torch as th
import numpy as np
from joblib import Parallel, delayed
from functools import partial

# -- pytorch imports --
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms as thT
import torchvision.transforms.functional as tvF
import torchvision.utils as tvUtils

# -- submillilux noise --
try:
    import stardeno
except:
    pass

# -- project imports --
# from pyutils.timer import Timer

class SuperResolutionNoise:

    def __init__(self,scale):
        self.scale = scale

    def __call__(self,image):
        osize = list(image.shape[-2:])
        isize = [int(s/self.scale) for s in osize]
        smaller = tvF.resize(image,isize,antialias=False)
        noisy = tvF.resize(smaller,osize,antialias=False)
        return noisy

class QIS:

    fields = ["alpha","read_noise","nbits"]
    def __init__(self,alpha,read_noise,nbits,seed=None):
        self.alpha = alpha
        self.read_noise = read_noise
        self.seed = seed
        self.nbits = nbits

    def __call__(self,image):
        pix_max = 2**self.nbits-1
        frame = np.random.poisson(self.alpha*image)
        frame += self.read_noise*np.random.randn(*image.shape)
        frame = np.round(frame)
        frame = np.clip(frame, 0, pix_max)
        noisy = frame.astype(np.float32) / self.alpha
        return noisy

class LowLight:

    def __init__(self,alpha,seed=None):
        self.alpha = alpha
        self.seed = seed

    def __call__(self,pic):
        low_light_pic = torch.poisson(self.alpha*pic,generator=self.seed)/self.alpha
        return low_light_pic

class AddHeteroGaussianNoise():
    def __init__(self, mean=0., read=25., shot=1.):
        self.mean = mean
        self.read = read/255.
        self.shot = shot/255.

    def __call__(self, tensor):
        pic = torch.normal(tensor.add(self.mean),self.read)
        shot_noise = torch.normal(0,self.shot * tensor)
        pic.add(shot_noise)
        return pic
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1e-2):
        self.mean = mean
        self.std = std# / 255.
        self.sigma = std
        # self.counter = 0
        # print("Creating a new gaussian noise.")

    def __call__(self, tensor):
        # milli_time = round(time.time() * 1000)
        # name = str(uuid.uuid4()) + f"_{milli_time}"
        # fn = f"./rands/rng_{name}_{self.counter}.txt"
        # np.savetxt(fn,torch.get_rng_state().numpy())
        # fn = f"./rands/img_{name}_{self.counter}.txt"
        # np.savetxt(fn,tensor.numpy().ravel())
        pic = torch.normal(tensor.add(self.mean),self.std)
        # fn = f"./rands/noisy_{name}_{self.counter}.txt"
        # np.savetxt(fn,pic.numpy().ravel())
        # self.counter += 1
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddLowLightNoiseBW(object):

    def __init__(self,alpha,read_noise,nbits=3,use_adc=True,seed=None):
        self.alpha = alpha
        self.read_noise = read_noise
        self.nbits = nbits
        self.use_adc = use_adc
        self.seed = seed

    def __call__(self,pic):
        """
        :params pic: input image shaped [...,C,H,W]

        we assume C = 3 and then we convert it to BW.
        """
        # if pic.max() <= 1: pic *= 255.
        # print("noise",torch.get_rng_state())
        device = pic.device
        pix_max = 2**self.nbits-1
        pic_bw = tvF.rgb_to_grayscale(pic,1)
        ll_pic = torch.poisson(self.alpha*pic_bw,generator=self.seed)
        ll_pic += self.read_noise*torch.randn(ll_pic.shape,device=device)
        if pic.shape[-3] == 3: ll_pic = self._add_color_channel(ll_pic)
        if self.use_adc:
            ll_pic = torch.round(ll_pic)
            ll_pic = torch.clamp(ll_pic, 0, pix_max)
        ll_pic /= self.alpha
        return ll_pic

    def __repr__(self):
        return self.__class__.__name__ + '(alpha={0})'.format(self.alpha)

    def _add_color_channel(self,ll_pic):
        repeat = [1 for i in ll_pic.shape]
        repeat[-3] = 3
        ll_pic = ll_pic.repeat(*(repeat))
        return ll_pic

class AddPoissonNoiseBW(object):

    def __init__(self,alpha,seed=None):
        self.alpha = alpha
        self.seed = seed

    def __call__(self,pic):
        pic_bw = tvF.rgb_to_grayscale(pic,1)
        poisson_pic = torch.poisson(self.alpha*pic_bw,generator=self.seed)/self.alpha
        if pic.shape[-3] == 3:
            repeat = [1 for i in pic.shape]
            repeat[-3] = 3
            poisson_pic = poisson_pic.repeat(*(repeat))
        return poisson_pic

    def __repr__(self):
        return self.__class__.__name__ + '(alpha={0})'.format(self.alpha)

class AddPoissonNoise(object):

    def __init__(self,alpha,seed=None):
        self.alpha = alpha
        self.seed = seed

    def __call__(self,pic):
        poisson_pic = torch.poisson(self.alpha*pic,generator=self.seed)/self.alpha
        return poisson_pic

    def __repr__(self):
        return self.__class__.__name__ + '(alpha={0})'.format(self.alpha)


class AddGaussianNoiseRandStd(object):
    def __init__(self, mean=0., min_std=0,max_std=50):
        self.mean = mean
        self.min_std = min_std
        self.max_std = max_std

    def __call__(self, tensor):
        std = th_uniform(self.min_std,self.max_std,1)
        pic = torch.normal(tensor.add(self.mean),std)
        return pic

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddGaussianNoiseSet(object):
    def __init__(self, N, mean= 0., std=50):
        self.N = N
        self.mean = mean
        self.std = std / 255.

    def __call__(self, tensor):
        pics = tensor.add(self.mean)
        pics = torch.stack(self.N*[pics])
        return torch.normal(pics,self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddMultiScaleGaussianNoise(object):

    def __init__(self, sigma_min, sigma_max, return_sigma=False):
        self.sigma_min = torch.tensor([1.*sigma_min])
        self.sigma_max = torch.tensor([1.*sigma_max])
        Uniform = torch.distributions.uniform.Uniform
        self.unif = Uniform(self.sigma_min,self.sigma_max)
        self.sigma = -1
        self.return_sigma = return_sigma

    def __call__(self, tensor, sigma=None):

        # -- sample noise intensity --
        if sigma is None:
            sigma = self.unif.sample().item()
        self.sigma = sigma

        # -- sample noise --
        noise_level = torch.ones((1, 1, 1, 1)) * self.sigma
        pic = torch.normal(mean=tensor, std=noise_level.expand_as(tensor))

        # -- combine with noise --
        t,_,h,w = tensor.shape
        pic = torch.cat([pic, noise_level.expand(t,1,h,w)], 1)

        if self.return_sigma:
            return pic,sigma
        else:
            return pic

    def __repr__(self):
        smin = self.sigma_min.item()
        smax = self.sigma_max.item()
        print(smin,smax)
        return self.__class__.__name__ + ' ({%2.2f,%2.2f})' % (smin,smax)

class GaussianBlur:

    def __init__(self,size,min_sigma=0.1,max_sigma=2.0,dim=2,channels=3):
        self.size = size
        self.sigma_range = (min_sigma,max_sigma)
        self.dim = dim
        self.channels = channels

    def __call__(x):
        kernel = self.gaussian_kernel()
        kernel_size = 2*self.size + 1

        x = x[None,...]
        padding = int((kernel_size - 1) / 2)
        x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        x = torch.squeeze(F.conv2d(x, kernel, groups=3))

        return x

    def gaussian_kernel(self):
        """
        The gaussian kernel is the product of the
        gaussian function of each dimension.
        """
        # unpacking
        size = self.size
        dim = self.dim
        channels = self.channels
        sigma = th_uniform(*self.sigma_range,1)[0].to(tensor.device)

        # kernel_size should be an odd number.
        kernel_size = 2*size + 1

        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32)
                                    for size in kernel_size])

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        return kernel

class BlockGaussian:

    def __init__(self,N,mean=0.,std=1e-1, mean_size=.4):
        self.N = N # number of transforms
        self.mean = mean
        self.std = std
        self.mean_size = mean_size

    def __call__(self, pic):
        (h,w) = pic.shape
        b_h = int(self.mean_size * h)
        b_w = int(self.mean_size * w)
        pics = []
        for n in range(self.N):
            pic_n = pic.clone()
            y = torch.randint(h)
            x = torch.randint(w)
            y1 = torch.clamp(y - b_h // 2, 0, h)
            y2 = torch.clamp(y + b_h // 2, 0, h)
            x1 = torch.clamp(x - b_w // 2, 0, w)
            x2 = torch.clamp(x + b_w // 2, 0, w)
            mask = torch.normal(self.mean,self.std,(y2-y1,x2-x1))
            mask = torch.Tensor(mask).to(pic.device)
            mask = torch.clamp(mask,0,1.)
            pic_n[y1:y2, x1:x2] = mask
            pics.append(pic_n)
        return pics

class Submillilux:

    def __init__(self,device="cuda:1"):
        self.device = device
        self.gan = stardeno.load_noise_sim(device,True).to(device)

    def __call__(self,vid):
        assert vid.shape[-3] in [3,4],"three or four color channels."
        if vid.shape[-3] == 3:
            empty = th.zeros_like(vid[...,[0],:,:])
            vid = th.cat([vid,empty],-3)
        with th.no_grad():
            vid = self.gan(vid.to(self.device)/255.).cpu()*255.
        vid = vid[...,:3,:,:].contiguous()
        return vid

class PoissonGaussianNoise:

    def __init__(self, sigma, rate):
        self.sigma = sigma
        self.rate = rate

    def __call__(self, tensor):
        ratio = self.rate/255.
        pic = torch.poisson(ratio*tensor)/ratio
        pic = torch.normal(pic,self.sigma)
        return pic

    def __repr__(self):
        msg = '(sigma={0},rate={1})'.format(self.sigma,self.rate)
        return self.__class__.__name__ + msg

class ScaleZeroMean:

    def __init__(self):
        pass

    def __call__(self,pic):
        return pic - 0.5
