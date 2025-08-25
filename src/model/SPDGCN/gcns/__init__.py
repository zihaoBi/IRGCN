
from re import M
from .RGCN import RGCN_Module


def create(args, **kwargs):
    return RGCN_Module(args.train_args['input_dims'], args.train_args['window_size'], args.train_args['step'], args.train_args['k'],  args.train_args['kk'],args, **kwargs)