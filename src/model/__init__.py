from . import TPGCN

__models = {
    'IRGCN': TPGCN,
}

def create(model_type, **kwargs):
    model_name = model_type.split('-')[0]
    return __models[model_name].create(model_type, **kwargs)
