from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.properties as properties
import schnetpack.nn as snn

__all__ = ["phi", "w", "u", "v","s", "PaiNN"]


# v_norm = r_ij/torch.sqrt(torch.sum(r_ij**2))

# v_j = torch.zeros(128)

class phi(nn.Module):

     def __init__(self):
        super().__init__()
        activation_fn = nn.SiLU

        self.net = nn.Sequential(
            nn.Linear(s_j, 128),
            activation_fn(),
            nn.Linear(128, 384),
        )
class w(nn.Module):
    def __init__(self):
        super().__init__()
        n=20
        r_RBF = torch.sin((n*torch.pi()/r_cut)*r_ij)/r_ij
        self.r_RBF = nn.Linear(r_RBF,384)
        f_c=0.5*torch.cos(torch.pi()*r_ij/r_cut)+1
        self.net = f_c

class Message_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(384,128)

        self.phi = snn.replicate_module(
            lambda: phi(
                # variables
            )
        )
        self.w = snn.replicate_module(
            lambda: w(
                # variables
            )
        )

    def forward(self, input1, input2):
        phi = self.net(input1)
        w = self.net(input2)
        output = phi * w
        output = torch.split(output,3)
        output1 = output[0] * v_j
        s_m = torch.sum(output[1]) + s_j
        output3 = output[2]*v_norm
        v_m= torch.sum(output1+output3) + v_j
        return v_m, s_m



### insert update block here

#Update block

class u(nn.Module):
      def __init__(self):
        super().__init__()
        self.net = nn.Linear(v_m,128)
class v(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(v_m,128)
class s(nn.Module):
    def __init__(self):
        super().__init__()
        activation_fn = nn.SiLU
        self.net = nn.Sequential(
            torch.stack(v_norm,s_m),
            nn.Linear(256,128),
            activation_fn(),
            nn.Linear(128,384))

class Update_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(384,128)
        self.u = snn.replicate_module(
            lambda: u(
                # variables
            )
        )
        self.v = snn.replicate_module(
            lambda: v(
                # variables
            )
        )
        self.s = snn.replicate_module(
            lambda: s(
                # variables
            )
        )

    def forward(self,input3,input4,input5):
        U = self.net(input3)
        V = self.net(input4)
        S = self.net(input5)
        V_dup = V.repeat(128,2)
        S = torch.split(S,3)
        output4 = S[0]*U
        output5 = S[1]*V_dup
        output6 = S[2]+output5
        v_u = output4 + v_m
        s_u = output6 + s_m
        return v_u, s_u


class PaiNN(nn.Module):
    def __init__(self):
        super().__init__()


        self.message_block = snn.replicate_module(
            lambda: Message_block(
                # variables
            )
        )

        self.update_block = snn.replicate_module(
            lambda: Update_block(
                # variables
            )
        )

    def forward(self, )
        #message block

        #update block
