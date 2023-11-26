#PAINN model

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.properties as properties
import schnetpack.nn as snn

# TODO: add all the functions here
__all__ = ["phi", "RBF","F_cut","w", "u", "v","s", "PaiNN"]


# v_norm = r_ij/torch.sqrt(torch.sum(r_ij**2))

# v_j = torch.zeros(128)

class phi(nn.Module):

     def __init__(self,input_dim=128):
        super().__init__()
        self.input_dim=input_dim
        activation_fn = nn.SiLU
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            activation_fn(),
            nn.Linear(128, 384),
        )
    def forward(self,s_j):
        return self.net(s_j)

class RBF(nn.Module):
    def __init__(self,r_cut=5.0):
        self.r_cut = r_cut
        self.n_values = torch.arange(1, 21, dtype=torch.float32)
    def forward(self,r_ij):
        r_RBF = torch.sin((self.n_values*torch.pi()/self.r_cut)*r_ij)/r_ij
        return r_RBF

class F_cut(nn.Module):
    def __init__(self,r_cut=5.0):
        self.r_cut = r_cut
    def forward(self,r_ij):
        f_c=0.5*torch.cos(torch.pi()*r_ij/self.r_cut)+1
        return f_c

class w(nn.Module):
    def __init__(self,r_ij,r_cut):
        super().__init__()
        self.r_ij = r_ij
        self.r_cut = r_cut

        self.RBF=RBF(r_cut=5.0)
        self.F_cut=F_cut(r_cut=5.0)
        self.net = nn.Linear(20,384)

    def forward(self,r_ij):
        New_RBF = self.RBF(r_ij)
        New_F_cut=self.F_cut(r_ij)
        Total = New_RBF*New_F_cut
        Output=self.net(Total)
        return Output


class MessageBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.phi = phi(input_dim=128)
        self.w = w(r_ij=None, r_cut=5.0)  # Initialize w with r_ij=None

    def forward(self, s_j, r_ij, v_j, v_norm):
        output1 = self.phi(s_j)
        output2 = self.w(r_ij)
        output = output1 * output2
        output_split = torch.split(output, 3, dim=1)  # Split along the second dimension

        # TODO: replace these with edge indexes from x2e
        # atom i will be updated as a function of it's j neighbors (atom j)


        # Update s_m
        s_m = torch.sum(output_split[1], dim=1, keepdim=True) + s_j

        # Update v_m
        output3 = output_split[2] * v_norm
        v_m = torch.sum(output3, dim=1, keepdim=True) + v_j

        return s_m, v_m


#Update block

class u(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(128, 128)

    def forward(self, v_m):
        return self.net(v_m)


class v(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(128, 128)

    def forward(self, v_m):
        return self.net(v_m)


class S(nn.Module):
    def __init__(self):
        super(S, self).__init__()
        activation_fn = nn.SiLU()
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            activation_fn,
            nn.Linear(128, 384)
        )

    def forward(self, v_norm, s_m):
        stack = torch.stack((v_norm, s_m))
        output = self.net(stack)
        output = torch.split(output, 128)
        return output


class UpdateBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.u = u()
        self.v = v()
        self.s = S()

    def forward(self, v_m, v_j, s_j, s_m):
        output_u = self.u(v_m)
        output_v = self.v(v_m)
        output_s = self.s(v_norm=s_m, s_m=s_j)

        V_dup = output_v.repeat(1, 2)  # Assuming v_m has shape (batch_size, 128)
        output_s1 = output_s[0] * output_u
        output_s2 = output_s[1] * V_dup
        output_s3 = output_s[2] + output_s2

        # TODO: replace these with edge indexes from x2e
        # atom i will be updated as a function of it's j neighbors (atom j)


        v_i = output_s1 + v_j
        s_i = output_s3 + s_j

        return v_i, s_i
