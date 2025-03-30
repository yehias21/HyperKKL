import torch
import numpy as np
from torch.autograd import function
from typing import cast


class Criterion(torch.nn.Module):
    def __init__(self, loss: str, method, pde_loss=None, normalizer=None):
        super().__init__()
        self.pde_loss = pde_loss
        self.method = method
        self.normalizer = normalizer
        match loss:
            case "mse":
                self.loss_fn = torch.nn.MSELoss()
            case _:
                raise NotImplementedError(f"Loss {loss} not supported")

    def forward(self, output, data) -> torch.Tensor:
        match self.method:
            case "unsupervised":
                loss = self.loss_fn(output['x_states']['x_regress'], data['x_states']['x_regress'])
            case "supervised":
                loss = self.loss_fn(output['x_states']['x_regress'], data['x_states']['x_regress'])
                loss += self.loss_fn(output['z_states']['z_regress'], data['z_states']['z_regress'])
            case "supervised_pinn":
                # normalize output, data for regress and physics points
                x_reg_out, z_reg_out = output['x_states']['x_regress'], output['z_states']['z_regress']
                x_reg_out, z_reg_out = self.normalizer.denormalize(x_reg_out, 'x_regress'), self.normalizer.denormalize(
                    z_reg_out, 'z_regress')
                # same for data
                x_reg_data, z_reg_data = data['x_states']['x_regress'], data['z_states']['z_regress']
                x_reg_data, z_reg_data = self.normalizer.denormalize(x_reg_data,
                                                                     'x_regress'), self.normalizer.denormalize(
                    z_reg_data, 'z_regress')
                # loss for regress points
                loss = self.loss_fn(x_reg_out, x_reg_data)
                loss += self.loss_fn(z_reg_out, z_reg_data)
                loss += self.pde_loss(output['z_states']['z_physics'], data['x_states']['x_physics'],
                                      data['y_out']['y_physics'])

            case _:
                raise NotImplementedError(f"Method {self.method} not supported")
        return loss


class PDELoss(torch.nn.Module):
    def __init__(self, diff_eq, forward_mapper, A, B, normalizer=None):
        super().__init__()
        self.diff_eq = diff_eq
        self.forward_mapper = forward_mapper
        self.A, self.B = torch.tensor(A,dtype=torch.float32), torch.tensor(B,dtype=torch.float32)

    def _get_jac(self, x):
        dtdx = torch.autograd.functional.jacobian(cast(function, self.forward_mapper), x)
        idx = torch.arange(x.shape[0])
        dtdx = dtdx[idx, :, idx, :]
        return dtdx

    def forward(self,
                z_pred: torch.Tensor, x_label, y_label) -> torch.Tensor:
        """
         mean squared residual : N
        """
        # assert x_label.shape == 2
        x_dot = torch.tensor(np.array([self.diff_eq(x=x) for x in x_label]))
        dttheta_dx = self._get_jac(x_label)
        dtdx_mul_f = torch.bmm(dttheta_dx, torch.unsqueeze(x_dot, 2))

        m_mul_t = torch.matmul(self.A, torch.unsqueeze(z_pred, 2))
        k_mul_h = torch.matmul(self.B.reshape(-1, y_label.shape[-1]),
                               torch.unsqueeze(torch.tensor(np.array(y_label)), 2))

        pde = dtdx_mul_f - m_mul_t - k_mul_h
        loss_batch = torch.linalg.norm(pde, dim=1)  # fixme: is order 2 or norm is enough
        loss_pde = torch.mean(loss_batch)

        return loss_pde
