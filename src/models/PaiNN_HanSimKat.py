import nn
import torch
import torch.nn as nn
import numpy

class PaiNN(nn.Module):

    def __init__(self, r_cut, n_blocks = 3, embedding_size= 128,device: torch.device = 'cpu'): #rbf_size= 20, device: torch.device = 'cpu'):

        # Instantiate as a module of PyTorch
        super(PaiNN, self).__init__()

        # Parameters of the model
        self.r_cut = r_cut
        # self.rbf_size = rbf_size #TODO - tag stilling til denne
        n_embedding = 100 # number of all elements in the periodic table
        self.embedding_size = embedding_size # 128
        self.device = device

            # Embedding layer for our model
        self.embedding_layer = nn.Embedding(n_embedding, self.embedding_size)

        # Creating the instances for the iterations of message passing and updating
        self.message_blocks = nn.ModuleList([MessageBlock(embedding_size=self.embedding_size, r_cut=self.r_cut) for _ in range(n_blocks)]) #rbf_size=self.rbf_size, r_cut=self.r_cut) for _ in range(n_blocks)])
        self.update_blocks = nn.ModuleList([UpdateBlock(embedding_size=self.embedding_size) for _ in range(n_blocks)])


        # den lyseblå til slut
        self.blue_block = nn.Sequential(
            nn.Linear(embedding_size, embedding_size), #shape(128,128)
            nn.SiLU(),
            nn.Linear(embedding_size, 1)
            )



    def forward(self, input):

        # Every input into device
        edges = input['edges'].to(self.device)
        r_ij = input['r_ij'].to(self.device)
        r_ij_normalized = input['r_ij_normalized'].to(self.device) #
        unique_atm_mat = input['graph_idx'].to(self.device) # den store matrice : atomer unikke til molekyle
        z = input['z'].to(self.device) # atomic numbers



        # Outputs from the atomic numbers
        s = self.embedding_layer(z)

        # Initializing the v0
        v = torch.zeros((unique_atm_mat.shape[0], 3, self.embedding_size), # tidligere navn: v_j
                                  device = r_ij.device,
                                  dtype = r_ij.dtype
                                  ).to(self.device)

        for message_block, update_block in zip(self.message_blocks, self.update_blocks):
            s, v = message_block(
                s = s,
                v = v,
                edges = edges,
                r_ij = r_ij,
                r_ij_normalized = r_ij_normalized
            )
            s, v = update_block(
                s = s,
                v = v
            )

        blue_outputs = self.blue_block(s)

        outputs = torch.zeros_like(torch.unique(unique_atm_mat)).float().unsqueeze(dim=1)

        outputs.index_add_(0, unique_atm_mat, blue_outputs)

        return outputs


#### hertil er det godt

### define radial basis function
def RBF(r_ij, r_cut: float):
    r_RBF = []
    n_values = torch.arange(1, 21, dtype=torch.float32)
    for i in n_values:
        r_RBF_n = (torch.sin((i * torch.pi / r_cut) * r_ij)) / r_ij
        r_RBF.append(r_RBF_n)

    r_RBF = torch.cat(r_RBF, dim=1)
    return r_RBF
def fcut(r_ij, r_cut: float):
        f_c = 0.5 * (torch.cos(torch.pi * r_ij / r_cut) + 1)
        return f_c
#https://github.com/Yangxinsix/painn-sli/blob/main/PaiNN/model.py
class MessageBlock(nn.Module):
    def __init__(self, embedding_size: int, r_cut = float):
        super(Message, self).__init__()
        self.r_cut = r_cut
        self.net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size, bias=True),
            nn.Silu(),
            nn.Linear(embedding_size, 3*embedding_size, bias=True)
        )
        self.rbf_layer = nn.Linear(20, 3*embedding_size, bias=True)
    def forward(self, s, v, edges, r_ij, r_ij_normalized):
        rbf_pass = self.rbf_layer(RBF(r_ij, self.r_cut))
        rbf_pass = rbf_pass * fcut(r_ij, self.r_cut).unsqueeze(-1)
        s_pass = self.net(s)
        pass_out = rbf_pass * s_pass[edges[:,1]]
       
        delta_v, delta_s, delta_rep = pass_out.split(128, dim=-1)
        
        delta_v = v[edge[:,1]] * delta_v.unsqueeze(1) # hamard of neighbouring vectors
        
        delta_direction = r_ij_normalized.unsqueeze(dim=-1) * delta_rep.unsqueeze(dim=1) #norm af r_ij ganget med split, virker ikke pga unsqueeze
        
        delta_v = delta_v + delta_direction # plusser ovenstående med residualerne fra v
        
        s_empty = torch.zeros_like(s)
        v_empty = torch.zeros_like(v)
        s_empty.index_add_(0, edge[:, 0], delta_s)
        v_empty.index_add_(0, edge[:, 0], delta_v)

        s = s + s_empty
        v = v + v_empty
        return s,v

class UpdateBlock(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        
        self.U = nn.Linear(embedding_size,embedding_size)
        self.V = nn.Linear(embedding_size,embedding_size)

        self.net = nn.Sequential(nn.Linear(embedding_size*2, embedding_size),
                                 nn.Silu(),
                                 nn.Linear(embedding_size, embedding_size*3))
        def forward(self, s, v):
            U = self.U(v)
            V = self.V(v)
            V_norm = torch.linalg.norm(V,dim=1)
            sv_stack = torch.cat((V_norm, s), dim=1)
            sv_stack_pass = self.net(sv_stack)
            avv, asv, ass = torch.split(sv_stack_pass, v.shape[-1], dim = 1)
            d_v = avv.unsqueeze(1)*U
            product = torch.sum(U * V, dim=1) # den underlige vi troede var scalar
            d_s = product * asv + ass
            s = s + d_s
            v = v + d_v
            return s, v
        
        



if __name__=="__main__":
    train_set = DataLoader(batch_size=100)
    model = PaiNN(r_cut = getattr(train_set, 'r_cut'))
    val_set = train_set.get_val()
    test_set = train_set.get_test()
    for i, batch in enumerate(train_set):
        output = model(batch)
        print(output)
