from models.archs.CARN_arch import CARN_M
from models.archs.MSRN_arch import MSRN
from models.archs.FSRCNN_arch import FSRCNN_net
from models.archs.SRResNet_arch import SRResNet
from models.archs.SwinIR_arch import SwinIR

def define(opt):
    opt_net = opt['network']
    which_model = opt_net['which_model']

    if which_model == 'SRResNet':
        netG = SRResNet(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
            nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt['scale']
        )
    elif which_model == 'MSRN':
        netG = MSRN(
            scale=opt['scale'], rgb_range=opt_net['rgb_range'], n_colors=opt_net['n_colors'],
            n_feats=opt_net['n_feats'], n_blocks=opt_net['n_blocks'], kernel_size=opt_net['kernel_size']
        )
    elif which_model == 'CARN_M':
        netG = CARN_M(
            in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            scale=opt['scale'], group=opt_net['group']
        )
    elif which_model == 'FSRCNN':
        netG = FSRCNN_net(
            input_channels=opt_net['in_nc'],upscale=opt['scale'],
            d=opt_net['d'], s=opt_net['s'],m=opt_net['m']
        )
    elif which_model == 'SwinIR':
        netG = SwinIR(
            upscale=opt['scale'], in_chans=opt_net['in_chans'], img_size=opt_net['img_size'],
            window_size=opt_net['window_size'], img_range=opt_net['img_range'], depths=opt_net['depths'],
            embed_dim=opt_net['embed_dim'], num_heads=opt_net['num_heads'], mlp_ratio=opt_net['mlp_ratio'],
            upsampler=opt_net['upsampler'], resi_connection=opt_net['resi_connection']
        )
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG