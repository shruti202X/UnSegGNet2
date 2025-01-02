import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

# self.model = GNNpool(384, 64, 32, 2, self.device, activation, loss_type, conv_type).to(self.device)

class GNNpool(nn.Module):
    def __init__(self, input_dim, conv_hidden, mlp_hidden, num_clusters, device, activ="silu", loss_type="DMON", conv_type="ARMA"):
        """
        implementation of mincutpool model from: https://arxiv.org/pdf/1907.00481v6.pdf
        @param input_dim: Size of input nodes features
        @param conv_hidden: Size Of conv hidden layers
        @param mlp_hidden: Size of mlp hidden layers
        @param num_clusters: Number of cluster to output
        @param device: Device to run the model on
        @param activ: Activation function to use. Enum: ["deepcut_activation", "relu", "silu", "gelu", "selu"]
        @param loss_type: Loss function to use. Enum: ["DMON", "NCUT"]
        """
        super(GNNpool, self).__init__()
        self.device = device
        self.activ = activ
        self.num_clusters = num_clusters
        self.mlp_hidden = mlp_hidden
        if loss_type not in ["DMON", "NCUT"]:
            raise ValueError(f'Loss type: {loss_type} is not supported')
        self.loss_type = loss_type

        if activ == "deepcut_activation":
            act = 'relu'
            nn_activ = nn.ReLU()
            self.f_act = F.elu
        elif activ == "relu":
            act = 'relu'
            nn_activ = nn.ReLU()
            self.f_act = F.relu
        elif activ == "silu":
            act = 'silu'
            nn_activ = nn.SiLU()
            self.f_act = F.silu
        elif activ == "gelu":
            act = 'gelu'
            nn_activ = nn.GELU()
            self.f_act = F.gelu
        elif activ == "selu":
            act = 'selu'
            nn_activ = nn.SELU()
            self.f_act = F.selu
        else:
            raise ValueError("Activation function not supported")

        # GNN conv
        self.conv_type = conv_type
        if self.conv_type == "ARMA":
            self.convs = pyg_nn.ARMAConv(input_dim, conv_hidden, num_stacks=2, num_layers=4, act=nn_activ,\
                dropout=0.4,shared_weights=False)
        elif self.conv_type == "GCN":
            self.convs = pyg_nn.GCN(input_dim, conv_hidden, 1, act=act)
        elif self.conv_type == "GAT":
            self.conv1 = pyg_nn.GCN(input_dim, 128, 2, act=act)
            self.conv2 = pyg_nn.GATConv(128, 64, heads=2, concat=False, dropout=0.4, negative_slope=0.2)
            self.conv3 = pyg_nn.GCN(64, 64, 2, act=act)
        else:
            raise ValueError("Conv type not supported")

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(conv_hidden, mlp_hidden), nn_activ, nn.Dropout(0.25),
            nn.Linear(mlp_hidden, self.num_clusters))

    def forward(self, data, A):
        """
        forward pass of the model
        @param data: Graph in Pytorch geometric data format
        @param A: Adjacency matrix of the graph
        @return: Adjacency matrix of the graph and pooled graph (argmax of S)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # print(x.shape) # torch.Size([3025, 384])
        
        if self.conv_type == "GAT":
            x = self.conv1(x, edge_index, edge_attr)
            x = self.conv2(x, edge_index, edge_attr)
            x = self.conv3(x, edge_index, edge_attr)
        else:
            x = self.convs(x, edge_index, edge_attr)
        x = self.f_act(x)

        # pass feats through mlp
        H = self.mlp(x)
        # cluster assignment for matrix S
        S = F.softmax(H)

        # print(S.shape) # torch.Size([3025, 2])

        return A, S

    def loss(self, As, Ss, Ls=None):
        """
        loss calculation, relaxed form of Normalized-cut
        @param As: Adjacency matrices of the graph
        @param Ss: Polled graphs (argmax of Ss)
        @return: loss value
        """

        if Ls == None:
            modularity_term = None
            collapse_reg_term = None

            for A, S in zip(As, Ss):
                C = S
                d = torch.sum(A, dim=1)
                m = torch.sum(A)
                B = A - torch.ger(d, d) / (2 * m)
                
                I_S = torch.eye(self.num_clusters, device=self.device)
                k = torch.norm(I_S)
                n = S.shape[0]
                
                if modularity_term is None:
                    modularity_term = (-1/(2*m)) * torch.trace(torch.mm(torch.mm(C.t(), B), C))
                    collapse_reg_term = (torch.sqrt(k)/n) * (torch.norm(torch.sum(C.t(), dim=0), p='fro')) - 1
                else:
                    modularity_term_current = (-1/(2*m)) * torch.trace(torch.mm(torch.mm(C.t(), B), C))    
                    collapse_reg_term_current = (torch.sqrt(k)/n) * (torch.norm(torch.sum(C.t(), dim=0), p='fro')) - 1

                    if modularity_term.shape!= modularity_term_current.shape:
                        print(modularity_term.shape, modularity_term_current.shape)
                        print("modularity_term")
                    if collapse_reg_term.shape!= collapse_reg_term_current.shape:
                        print(collapse_reg_term.shape, collapse_reg_term_current.shape)
                        print("collapse_reg_term")
            
            modularity_term /= len(As)
            collapse_reg_term /= len(As)

            return modularity_term + collapse_reg_term
        else:
          
            modularity_term = None
            collapse_reg_term = None
            cross_entropy_term = None

            for A, S, L in zip(As, Ss, Ls):
                C = S
                d = torch.sum(A, dim=1)
                m = torch.sum(A)
                B = A - torch.ger(d, d) / (2 * m)
                
                I_S = torch.eye(self.num_clusters, device=self.device)
                k = torch.norm(I_S)
                n = S.shape[0]
                
                if modularity_term is None:
                    modularity_term = (-1/(2*m)) * torch.trace(torch.mm(torch.mm(C.t(), B), C))
                    collapse_reg_term = (torch.sqrt(k)/n) * (torch.norm(torch.sum(C.t(), dim=0), p='fro')) - 1
                    cross_entropy_term = np.sum(S[L == 1, 1])
                else:
                    modularity_term_current = (-1/(2*m)) * torch.trace(torch.mm(torch.mm(C.t(), B), C))    
                    collapse_reg_term_current = (torch.sqrt(k)/n) * (torch.norm(torch.sum(C.t(), dim=0), p='fro')) - 1
                    cross_entropy_term_current = np.sum(S[L == 1, 1])

                    if modularity_term.shape!= modularity_term_current.shape:
                        print(modularity_term.shape, modularity_term_current.shape)
                        print("modularity_term")
                    if collapse_reg_term.shape!= collapse_reg_term_current.shape:
                        print(collapse_reg_term.shape, collapse_reg_term_current.shape)
                        print("collapse_reg_term")
                    if cross_entropy_term.shape!=cross_entropy_term_current.shpae:
                        print(cross_entropy_term.shape, cross_entropy_term_current.shape)
                        print("cross_entropy_term")
            
            modularity_term /= len(As)
            collapse_reg_term /= len(As)
            cross_entropy_term /= len(As)

            return (modularity_term + collapse_reg_term)*0.5 + cross_entropy_term*0.5

    def loss2(self, A, S):
        """
        loss calculation, relaxed form of Normalized-cut
        @param A: Adjacency matrix of the graph
        @param S: Polled graph (argmax of S)
        @return: loss value
        """
        if self.loss_type == "NCUT":
            # cut loss
            A_pool = torch.matmul(torch.matmul(A, S).t(), S)
            num = torch.trace(A_pool)

            D = torch.diag(torch.sum(A, dim=-1))
            D_pooled = torch.matmul(torch.matmul(D, S).t(), S)
            den = torch.trace(D_pooled)
            mincut_loss = -(num / den)
            # orthogonality loss
            St_S = torch.matmul(S.t(), S)
            I_S = torch.eye(self.num_clusters, device=self.device)
            ortho_loss = torch.norm(St_S / torch.norm(St_S) - I_S / torch.norm(I_S))

            return mincut_loss + ortho_loss
        elif self.loss_type == "DMON":
            C = S
            d = torch.sum(A, dim=1)
            m = torch.sum(A)
            B = A - torch.ger(d, d) / (2 * m)
            
            I_S = torch.eye(self.num_clusters, device=self.device)
            k = torch.norm(I_S)
            n = S.shape[0]
            
            modularity_term = (-1/(2*m)) * torch.trace(torch.mm(torch.mm(C.t(), B), C))
            
            collapse_reg_term = (torch.sqrt(k)/n) * (torch.norm(torch.sum(C.t(), dim=0), p='fro')) - 1
            
            return modularity_term + collapse_reg_term
