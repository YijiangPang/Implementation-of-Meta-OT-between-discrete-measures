import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
#cite https://github.com/facebookresearch/meta-ot/blob/main/meta_ot/models.py
#cite https://github.com/AmirTag/OT-ICNN/blob/master/High_dim_experiments/optimal_transport_modules/icnn_modules.py
#cite https://github.com/iamalexkorotin/Wasserstein2GenerativeNetworks/blob/master/src/icnn.py



class PotentialMLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_num = 3, hidden_dim = 1024):
        super().__init__()
        self.model_base = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim_in, hidden_dim, dtype=torch.float64) if i == 0 else nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64),
            nn.ReLU()])
            for i in range(hidden_num)
        ])
        self.lin = nn.Linear(hidden_dim, dim_out, dtype=torch.float64)

    def forward(self, a, b):
        x = torch.cat((a, b), dim = -1)
        for L, A in self.model_base:
            x = A(L(x))
        return self.lin(x)


class ObjMLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_rank, hidden_num = 3, hidden_dim = 1024):
        super().__init__()
        self.model_base = nn.ModuleList([
            nn.ModuleList([nn.Linear(dim_in, hidden_dim, dtype=torch.float64) if i == 0 else nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64),
            nn.ReLU()])
            for i in range(hidden_num)
        ])
        self.opt_l = nn.Linear(hidden_dim, int(dim_out*dim_rank), dtype=torch.float64)
        self.opt_r = nn.Linear(hidden_dim, int(dim_out*dim_rank), dtype=torch.float64)
        self.eigen_v = nn.Linear(hidden_dim, dim_rank, dtype=torch.float64)

    def forward(self, a, b):
        x = torch.cat((a, b), dim = -1)
        for L, A in self.model_base:
            x = A(L(x))
        return self.opt_l(x), self.opt_l(x), self.eigen_v(x)

class PosLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(PosLinear, self).__init__(*kargs, **kwargs)
    def forward(self, input):
        # self.weight.data = F.softplus(self.weight, beta = 10)
        # self.weight.data.clamp_(0)
        # self.weight.be_positive = 1.0
        out = F.linear(input, self.weight, self.bias)
        return out

class PosConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(PosConv2d, self).__init__(*kargs, **kwargs)
    def forward(self, input):
        # self.weight = F.softplus(self.weight)
        # self.weight.data.clamp_(0)
        # self.weight.be_positive = 1.0
        out = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class DenseICNN(nn.Module):
    strong_convexity= 1.0
    def __init__(self, input_dim, hidden_dim, hidden_num):
        super(DenseICNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num

        self.linear_first = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_first = nn.LeakyReLU(0.2)
        self.hidden_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            PosLinear(self.hidden_dim, self.hidden_dim, bias=False),
            nn.LeakyReLU(0.2)])
            for _ in range(hidden_num)
        ])
        self.pos_last = PosLinear(self.hidden_dim, 1, bias=False)
        self.linear_last = nn.Linear(self.input_dim, 1, bias=True)

    def forward(self, inputs, params):
        if isinstance(params, list):
            outputs = []
            for X, P in zip(inputs, params):
                out = self._forward(X, P)
                outputs.append(out)
            outputs = torch.stack(outputs)
            return outputs
        else:
            return self._forward(inputs, params)


    def _forward(self, inputs, params_dic = None):
        z = F.linear(inputs, params_dic["linear_first.weight"], params_dic["linear_first.bias"])
        z = F.leaky_relu(z)
        for i, [_, _, _] in enumerate(self.hidden_layers):
            z_in = F.linear(inputs, params_dic["hidden_layers.%d.0.weight"%(i)], params_dic["hidden_layers.%d.0.bias"%(i)])
            p_pos = params_dic["hidden_layers.%d.1.weight"%(i)].data.clamp_(0)  #pos-weights
            z = F.linear(z, p_pos, bias = None)  #pos-weights
            z = F.leaky_relu(z + z_in)
        z_in = F.linear(inputs, params_dic["linear_last.weight"], params_dic["linear_last.bias"])
        p_pos = params_dic["pos_last.weight"].data.clamp_(0)    #pos-weights
        z = F.linear(z, p_pos, bias = None)      #pos-weights
        output = z + z_in

        l2_reg = torch.sum(.5 * self.strong_convexity * (inputs ** 2), dim = -1, keepdim= True)
        return output + l2_reg
        
    
    def push(self, inputs, params):
        if isinstance(params, list):
            outputs = []
            for X, P in zip(inputs, params):
                out = self._push(X, P)
                outputs.append(out)
            outputs = torch.stack(outputs)
            return outputs
        else:
            return self._push(inputs, params)

    def _push(self, X, P):
        X.requires_grad_(True)
        # Y = self._forward(X, P)
        # g = [torch.autograd.grad(outputs=out, inputs=X, retain_graph=True)[0][i] for i, out in enumerate(Y)]
        g = torch.autograd.grad(self._forward(X, P).sum(), X, retain_graph=True)[0]
        return g     


class MetaICNN(nn.Module):

    fc_num_hidden_layers: int = 1
    fc_num_hidden_units: int = 512

    def __init__(self, icnn_para_dim, backbone, input_channel, pretrained):
        super().__init__()
        if backbone == "resnet18":
            self.model_base = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            self.model_base = models.resnet34(pretrained=pretrained)
        self.model_base_feature_dim = self.model_base.fc.in_features
        # self.model_base.conv1 = nn.Conv2d(input_channel, 64, 3, 1, 1, bias=False)
        # self.model_base.maxpool = nn.Identity()
        self.model_base.fc = nn.Identity()
        self.icnn_para_dim = icnn_para_dim
        self.fc_hidden_layers = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.model_base_feature_dim*2, self.fc_num_hidden_units) if i == 0 else nn.Linear(self.fc_num_hidden_units, self.fc_num_hidden_units),
            nn.ReLU()])
            for i in range(self.fc_num_hidden_layers)
        ])
        self.linear_last = nn.Linear(self.fc_num_hidden_units, 2*self.icnn_para_dim)

    def forward(self, x, y):

        zx = self.model_base(x)
        zy = self.model_base(y)

        z = torch.concat((zx, zy), axis=-1)
        for L, A in self.fc_hidden_layers:
            z = A(L(z))
        z = self.linear_last(z)

        D_params_flat, D_conj_params_flat = torch.chunk(z, 2, dim=-1)
        return D_params_flat, D_conj_params_flat