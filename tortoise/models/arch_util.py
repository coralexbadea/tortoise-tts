import os #working with directories
import functools # higher order functions
import math #math operations

import torch #ML library
import torch.nn as nn #base class for all newral network modules
import torch.nn.functional as F #functions related to NN
import torchaudio #torch library for audio handling
from tortoise.models.xtransformers import ContinuousTransformerWrapper, RelativePositionBias #undefined
#undefined
#undefined
def zero_module(module):#zero out the parameters of a module
    """#undefined
    Zero out the parameters of a module and return it.#undefined
    """#undefined
    for p in module.parameters(): #for each parameters in module
        p.detach().zero_()#The gradients of the tensor are no longer computed and values set to 0
    return module#return that module
#undefined    
#undefined
class GroupNorm32(nn.GroupNorm):#perform GN on float32
    def forward(self, x):#forward function
        return super().forward(x.float()).type(x.dtype)#perform gorup_norm using floats and returns x original type
#undefined
#undefined
def normalization(channels):#normalization function
    """#undefined
    Make a standard normalization layer.

    :param channels: number of input channels. #number of input channels
    :return: an nn.Module for normalization. #return a module
    """
    groups = 32 #32 groups
    if channels <= 16: #if channels <= 16
        groups = 8 #groups is now 8
    elif channels <= 64: #else if channels <=64
        groups = 16 #groups is now 16
    while channels % groups != 0: #while if we cant divide exactly channles by groups
        groups = int(groups / 2) #divide groups by 2, eventualy it can be 1 which is instance normalization
    assert groups > 2 #we make sure that groups is bigger than 2 so that it is still GN and not IN
    return GroupNorm32(groups, channels) #return a module of GN that takes groups and channels size
#undefined
#undefined
class QKVAttentionLegacy(nn.Module): #we will perform QKV attention in this module
    """#undefined
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """#undefined

    def __init__(self, n_heads):#init
        super().__init__()#call super init
        self.n_heads = n_heads #number of heads

    def forward(self, qkv, mask=None, rel_pos=None):#forward function taking qvk vector
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.#so the 3 comes from the 3 tensors
        :return: an [N x (H * C) x T] tensor after attention. #so one transformer token is [N x (H * C) x T]
        """
        bs, width, length = qkv.shape #batchsize, width, length of the tensor
        assert width % (3 * self.n_heads) == 0 #assert if width can be divided exactly by 3 * numheads
        ch = width // (3 * self.n_heads) #get channels size and the width divided by 3 * numheads
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1) #so we reshape putting heads first, then split into three by taking section size as ch
        scale = 1 / math.sqrt(math.sqrt(ch)) #the scale is 1/sqrt(sqrt(chanelsize))
        weight = torch.einsum(#perform einstein sum with q and k scaled
            "bct,bcs->bts", q * scale, k * scale #we transpose the cannels t and c from q and them multiply with k and keep batch dimension
        )  # More stable with f16 than dividing afterwards # it seems that we are using f16
        if rel_pos is not None: #if rel_pos is not none
            weight = rel_pos(weight.reshape(bs, self.n_heads, weight.shape[-2], weight.shape[-1])).reshape(bs * self.n_heads, weight.shape[-2], weight.shape[-1]) #perform le position on the weights reshaped as b,heads,t,s
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype) #perform softmax on the last dimension
        if mask is not None: #if mask is not None
            # The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs. #apply mask after softmax
            mask = mask.repeat(self.n_heads, 1).unsqueeze(1)#repeat the mask for all the heads then unsqeeze since heads are not delimited explicitly
            weight = weight * mask #perform masking
        a = torch.einsum("bts,bcs->bct", weight, v) #multiply with v to obtain again the bct shape

        return a.reshape(bs, -1, length)# finnaly reshape again from heads first to heads inside


class AttentionBlock(nn.Module): #AttentionBlock as module
    """
    An attention block that allows spatial positions to attend to each other.# we are using spatial positions

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.#unet from a diffusion model
    """

    def __init__(#init function
        self,
        channels,#channels
        num_heads=1,#num_heads
        num_head_channels=-1,#num_head_channels_size as many as posible
        do_checkpoint=True,# do checkpoint
        relative_pos_embeddings=False, # relative position elbedings
    ):
        super().__init__() #super call
        self.channels = channels #channels
        self.do_checkpoint = do_checkpoint #do checkpoint
        if num_head_channels == -1: #if as many as possible nun_head_channels size
            self.num_heads = num_heads #num heads
        else:       
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64)
        else:
            self.relative_pos_embeddings = None

    def forward(self, x, mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, mask, self.relative_pos_embeddings)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None, factor=4):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.factor = factor
        if use_conv:
            ksize = 5
            pad = 2
            self.conv = nn.Conv1d(self.channels, self.out_channels, ksize, padding=pad)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=self.factor, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None, factor=4, ksize=5, pad=2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        stride = factor
        if use_conv:
            self.op = nn.Conv1d(
                self.channels, self.out_channels, ksize, stride=stride, padding=pad
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(
            self,
            channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            up=False,
            down=False,
            kernel_size=3,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = 1 if kernel_size == 3 else 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size, padding=padding)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv1d(
                channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AudioMiniEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 base_channels=128,
                 depth=2,
                 resnet_blocks=2,
                 attn_blocks=4,
                 num_attn_heads=4,
                 dropout=0,
                 downsample_factor=2,
                 kernel_size=3):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv1d(spec_dim, base_channels, 3, padding=1)
        )
        ch = base_channels
        res = []
        for l in range(depth):
            for r in range(resnet_blocks):
                res.append(ResBlock(ch, dropout, kernel_size=kernel_size))
            res.append(Downsample(ch, use_conv=True, out_channels=ch*2, factor=downsample_factor))
            ch *= 2
        self.res = nn.Sequential(*res)
        self.final = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            nn.Conv1d(ch, embedding_dim, 1)
        )
        attn = []
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads,))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        h = self.init(x)
        h = self.res(h)
        h = self.final(h)
        h = self.attn(h)
        return h[:, :, 0]


DEFAULT_MEL_NORM_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/mel_norms.pth')


class TorchMelSpectrogram(nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, mel_fmin=0, mel_fmax=8000,
                 sampling_rate=22050, normalize=False, mel_norm_file=DEFAULT_MEL_NORM_FILE):
        super().__init__()
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=self.filter_length, hop_length=self.hop_length,
                                                             win_length=self.win_length, power=2, normalized=normalize,
                                                             sample_rate=self.sampling_rate, f_min=self.mel_fmin,
                                                             f_max=self.mel_fmax, n_mels=self.n_mel_channels,
                                                             norm="slaney")
        self.mel_norm_file = mel_norm_file
        if self.mel_norm_file is not None:
            self.mel_norms = torch.load(self.mel_norm_file)
        else:
            self.mel_norms = None

    def forward(self, inp):
        if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        self.mel_stft = self.mel_stft.to(inp.device)
        mel = self.mel_stft(inp)
        # Perform dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if self.mel_norms is not None:
            self.mel_norms = self.mel_norms.to(mel.device)
            mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
        return mel


class CheckpointedLayer(nn.Module):
    """
    Wraps a module. When forward() is called, passes kwargs that require_grad through torch.checkpoint() and bypasses
    checkpoint for all other args.
    """
    def __init__(self, wrap):
        super().__init__()
        self.wrap = wrap

    def forward(self, x, *args, **kwargs):
        for k, v in kwargs.items():
            assert not (isinstance(v, torch.Tensor) and v.requires_grad)  # This would screw up checkpointing.
        partial = functools.partial(self.wrap, **kwargs)
        return partial(x, *args)


class CheckpointedXTransformerEncoder(nn.Module):
    """
    Wraps a ContinuousTransformerWrapper and applies CheckpointedLayer to each layer and permutes from channels-mid
    to channels-last that XTransformer expects.
    """
    def __init__(self, needs_permute=True, exit_permute=True, checkpoint=True, **xtransformer_kwargs):
        super().__init__()
        self.transformer = ContinuousTransformerWrapper(**xtransformer_kwargs)
        self.needs_permute = needs_permute
        self.exit_permute = exit_permute

        if not checkpoint:
            return
        for i in range(len(self.transformer.attn_layers.layers)):
            n, b, r = self.transformer.attn_layers.layers[i]
            self.transformer.attn_layers.layers[i] = nn.ModuleList([n, CheckpointedLayer(b), r])

    def forward(self, x, **kwargs):
        if self.needs_permute:
            x = x.permute(0,2,1)
        h = self.transformer(x, **kwargs)
        if self.exit_permute:
            h = h.permute(0,2,1)
        return h