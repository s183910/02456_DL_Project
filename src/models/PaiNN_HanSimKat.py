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




class phi(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        activation_fn = nn.SiLU
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=True),
            activation_fn(),
            nn.Linear(input_dim, 3*input_dim, bias=True)
        )

    def forward(self, s_j):
        return self.net(s_j)

class RBF(nn.Module):
    def __init__(self, r_cut=5.0):
        super().__init__()
        self.r_cut = r_cut
        self.n_values = torch.arange(1, 21, dtype=torch.float32)

    def forward(self, r_ij):
        r_RBF_list = []

        for n_value in self.n_values:
            r_RBF_n = (torch.sin((n_value * 3.14 / self.r_cut) * r_ij)) / r_ij
            r_RBF_list.append(r_RBF_n)

        r_RBF = torch.stack(r_RBF_list, dim=1)
        return r_RBF

class F_cut(nn.Module):
    def __init__(self, r_cut=5.0):
        super().__init__()
        self.r_cut = r_cut

    def forward(self, r_ij):
        f_c = 0.5 * torch.cos(torch.pi * r_ij / self.r_cut) + 1
        return f_c

class w(nn.Module):
    def __init__(self, r_cut=5.0):
        super().__init__()
        self.RBF = RBF(r_cut)
        self.F_cut = F_cut(r_cut)
        self.net = nn.Linear(20, 384, bias=True)

    def forward(self, r_ij):
        New_RBF = self.RBF(r_ij)
        New_F_cut = self.F_cut(r_ij).unsqueeze(1)
        Total = New_RBF * New_F_cut
        Output = self.net(Total)
        return Output

class MessageBlock(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.phi = phi(input_dim)
        self.w = w()
        self.v_j = nn.Parameter(torch.zeros(input_dim))

    def forward(self, v_j, s, r_ij):
        # output_phi = self.phi(s)
        # output_w = self.w(r_ij)
        # output_conv = output_phi * output_w
        # output_split = torch.chunk(output_conv, 3, dim=1)

        # output_v = output_split[0] * v_j  # Select the first 128 elements
        # delta_s_im = output_split[1]  # Select the next 128 elements
        # output_r = output_split[2] * (r_ij / r_ij)  # Select the last 128 elements #TODO: check norm

        # delta_s_im = torch.sum(delta_s_im, dim=1)
        # delta_v_im = torch.sum(output_v + output_r, dim=1)
        # s = s + delta_s_im
        # v_j = v_j + delta_v_im
        # her er hans tester

        output_phi = self.phi(s)
        output_w = self.w(r_ij)
        convolution1 = output_phi * output_w #first convolution
        delta_v, delta_s, delta_rep = convolution1.split(128, dim=-1) #split into three
        delta_v = v_j * delta_v.unsqueeze(dim=1) # hamard of neighbouring vectors
        delta_direction = r_ij_normalized.unsqueeze(dim=-1) * delta_rep.unsqueeze(dim=1) #norm af r_ij ganget med split, virker ikke pga unsqueeze
        delta_v = delta_v + delta_direction # plusser ovenstående med residualerne fra v
        s = s + torch.zeros_like(s).index_add(0, index_atom, delta_s) # her opdaterer vi vores nodes med før
        v_j = v_j + torch.zeros_like(v_j).index_add(0, index_atom, delta_v)
        return s, v_j




class UpdateBlock(nn.Module):
    def __init__(self, size_atomwise):
        '''
        node_size = size_atomwise: size of the atomwise layers
        '''
        super(UpdateBlock).__init__()

        self.size_atomwise = size_atomwise

        # The U and V matrices
        self.U = nn.Linear(size_atomwise, size_atomwise, bias = False) # U takes v_j and outputs u_m
        self.V = nn.Linear(size_atomwise, size_atomwise, bias = False) # V takes v_j and outputs v_m


        # Atomwise layers applied to node scalars and V projections (stacked)
        # This is the S string of opearations
        self.atomwise_layers = nn.Sequential(
            nn.Linear(2 * size_atomwise, size_atomwise), # shape (256, 128)
            nn.SiLU(), # activation function
            nn.Linear(size_atomwise, 3 * size_atomwise) # shape (128, 384)
        )


    def forward(self, s, v_j):  # atom_scalars = s, atom_vector = v_j
        """ Forward pass
        Args:
            s = atom_scalars: scalar representations of the atoms
            v_j = atom_vectors: vector (equivariant) representations of the atoms
                this is the v_j vector
            graph: interactions between atoms = edges
            edges_dist: distances between neighbours
            r_cut: radius to cutoff interaction = 5 Å
        """
        # Outputs from matrix projection
        Uv_j = self.U(s)
        Vv_j = self.V(v_j)


        # Stacking V projections and node scalars
        s_Vv_j = torch.cat((s, torch.linalg.norm(Vv_j, dim=1)), dim=1)
        a = self.atomwise_layers(s_Vv_j)
        avv, asv, ass = a.split(self.node_size, dim=-1)

        # Scalar product between Uv and Vv
        s_product = torch.sum(Uv_j * Vv_j, dim=1)

        # Calculating the residual values for scalars and vectors
        delta_s = ass + asv * s_product
        delta_v = avv.unsqueeze(dim=1) * Uv_j

        # Updating the representations
        s = s + delta_s
        v_j = v_j + delta_v

        return s, v_j

if __name__=="__main__":
    train_set = DataLoader(batch_size=100)
    model = PaiNN(r_cut = getattr(train_set, 'r_cut'))
    val_set = train_set.get_val()
    test_set = train_set.get_test()
    for i, batch in enumerate(train_set):
        output = model(batch)
        print(output)
