import torch
from torch import nn
from ..spd import functional, modules


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        outputs = self.predict_node_classes(x)
        return outputs

    def predict_node_classes(self, log_mats):
        raise NotImplementedError("predict_node_classes is not implemented in Classifier")

    @classmethod
    def get(cls, type_str, final_channels, k, num_class, cls_pow):
        classifs = {
            "linear": LinearClassifier,
        }
        return classifs[type_str](final_channels, k, num_class, cls_pow)


class LinearClassifier(Classifier):
    def __init__(self, final_channels, k, num_class, cls_pow):
        super().__init__()
        self.dims = final_channels
        # self.dims = self.hidden_dims * (self.hidden_dims + 1) // 2 + args.train_args['kk']
        self.rows, self.cols = torch.tril_indices(self.dims, self.dims)
        self.proj = nn.Linear(self.dims * (self.dims + 1) // 2, num_class)
        # self.dropout = nn.Dropout(args.dropout)
        self.power = modules.Pow_Log_I(cls_pow)

    def predict_node_classes(self, x):
        x = self.power(x)
        # x = functional.sym_logm.apply(x)
        node_feats = x[..., self.rows, self.cols]
        # node_feats = node_feats.mean(1).mean(1).reshape(node_feats.shape[0], -1)
        # node_feats = self.dropout(node_feats)
        predictions = self.proj(node_feats)
        return predictions
