import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import from_scipy_sparse_matrix



from torch_geometric.nn import GCNConv,SAGEConv,GATConv,ResGatedGraphConv

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        if mask is None:
            mask = torch.eye(emb.size(0)).to(emb.device)
        vsum = torch.mm(mask, emb)
    # 其他代码

        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 
    




from torch_geometric.nn import LayerNorm,BatchNorm,GCNConv

class GraphCVAEEncoder(nn.Module):
    def __init__(self, dim_input, dim_output, edge_index):
        super(GraphCVAEEncoder, self).__init__()
        self.edge_index = edge_index
        hidden_dim1 = dim_output * 2
        hidden_dim2 = dim_output * 4
        hidden_dim3 = dim_output * 8

        self.gcn_conv1 = GCNConv(dim_input, hidden_dim1)
        self.gcn_conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.gcn_conv3 = GCNConv(hidden_dim2, hidden_dim3)

        self.ln1 = BatchNorm(hidden_dim1)
        self.ln2 = BatchNorm(hidden_dim2)
        self.ln3 = BatchNorm(hidden_dim3)

        self.gcn_conv_mean = GCNConv(hidden_dim3, dim_output)
        self.gcn_conv_logvar = GCNConv(hidden_dim3, dim_output)

        self.ln_mean = BatchNorm(dim_output)
        self.ln_logvar = BatchNorm(dim_output)

        self.fc2 = nn.Linear(dim_output, dim_input)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(0.1)

        self.disc = Discriminator(dim_output)
        self.read = AvgReadout()
        
        # Add projection layers to adjust the dimensions of the residuals
        self.proj1 = nn.Linear(dim_input, hidden_dim1)  # Projection for the first layer
        self.proj2 = nn.Linear(hidden_dim1, hidden_dim2)  # Projection for the second layer
        self.proj3 = nn.Linear(hidden_dim2, hidden_dim3)  # Projection for the third layer


    
    def encode(self, x, edge_index):
        # First layer with residual connection
        h0 = self.proj1(x)  # Project the input x to match the dimension of h
        h = self.gcn_conv1(x, edge_index)
        h = self.ln1(h) + h0  # Add the residual
        h = F.elu(h)
    
        # Second layer with residual connection
        h0 = self.proj2(h)  # Project the output of the previous layer to match the dimension of the next layer's output
        h = self.gcn_conv2(h, edge_index)
        h = self.ln2(h) + h0  # Add the residual
        h = F.elu(h)
    
        # Third layer with residual connection
        h0 = self.proj3(h0)  # Project the residual to match the dimension of the third layer's output
        h = self.gcn_conv3(h, edge_index)
        h = self.ln3(h) + h0  # Add the residual
        h = F.elu(h)
        
        # Compute mean and log variance for the Gaussian distribution
        mu = self.gcn_conv_mean(h, edge_index)
        mu = self.ln_mean(mu)
        logvar = self.gcn_conv_logvar(h, edge_index)
        logvar = self.ln_logvar(logvar)
    
        return mu, logvar, h

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, x_a):
        mu, logvar, hidden_emb = self.encode(x, self.edge_index)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.fc2(self.dropout(z))

        global_emb = self.read(z, None)
        mu_a, logvar_a, _ = self.encode(x_a, self.edge_index)
        global_emb_a = self.read(self.reparameterize(mu_a, logvar_a), None)

        ret = self.disc(global_emb, z, global_emb_a)
        ret_a = self.disc(global_emb_a, global_emb_a, z)

        return mu, logvar, hidden_emb, reconstructed, ret, ret_a
