from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.properties as properties
import schnetpack.nn as snn

__all__ = ["phi", "w", "u", "v", "s", "Message_block", "Update_block", "PaiNN"]

class phi(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        activation_fn = nn.SiLU
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            activation_fn(),
            nn.Linear(128, 384),
        )

    def forward(self, x):
        return self.net(x)

class w(nn.Module):
    def __init__(self, r_ij, r_cut, n=20):
        super().__init__()
        # Assuming r_ij and r_cut are tensors and precomputed
        r_RBF = torch.sin((n * torch.pi / r_cut) * r_ij) / r_ij
        self.r_RBF = nn.Linear(r_RBF.size(1), 384)
        f_c = 0.5 * torch.cos(torch.pi * r_ij / r_cut) + 1
        self.f_c = f_c

    def forward(self, x):
        rbf_out = self.r_RBF(x)
        return rbf_out * self.f_c

class Message_block(nn.Module):
    def __init__(self, phi_input_size, r_ij, r_cut):
        super().__init__()
        self.phi_layer = phi(phi_input_size)
        self.w_layer = w(r_ij, r_cut)

    def forward(self, input1, input2, v_j, s_j, v_norm):
        phi_output = self.phi_layer(input1)
        w_output = self.w_layer(input2)
        output = phi_output * w_output
        output = torch.split(output, 3)
        output1 = output[0] * v_j
        s_m = torch.sum(output[1]) + s_j
        output3 = output[2] * v_norm
        v_m = torch.sum(output1 + output3) + v_j
        return v_m, s_m

class u(nn.Module):
    def __init__(self, v_m_size):
        super().__init__()
        self.net = nn.Linear(v_m_size, 128)

    def forward(self, x):
        return self.net(x)


# update block

class Update_block(nn.Module):
    def __init__(self, v_m_size, s_m_size):
        super().__init__()
        self.u_layer = u(v_m_size)
        self.v_layer = v(v_m_size)
        self.s_layer = s(s_m_size)

    def forward(self, input3, input4, input5, v_m, s_m):
        U = self.u_layer(input3)
        V = self.v_layer(input4)
        S = self.s_layer(input5)
        V_dup = V.repeat(128, 2)
        S = torch.split(S, 3)
        output4 = S[0] * U
        output5 = S[1] * V_dup
        output6 = S[2] + output5
        v_u = output4 + v_m
        s_u = output6 + s_m
        return v_u, s_u

class PaiNN(nn.Module):
    def __init__(self, phi_input_size, r_ij, r_cut, v_m_size, s_m_size):
        super().__init__()
        self.message_block = Message_block(phi_input_size, r_ij, r_cut)
        self.update_block = Update_block(v_m_size, s_m_size)


    def forward(self, input1, input2, v_j, s_j, v_norm):
        # Forward pass through the message block
        v_m, s_m = self.message_block(input1, input2, v_j, s_j, v_norm)

        # Forward pass through the update block
        v_u, s_u = self.update_block(v_m, v_m, s_m, v_m, s_m)

        # Return the updated values
        return v_u, s_u
