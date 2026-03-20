from tokenize import group

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)


class EISAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, s=0.01, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert s > 0.0, f"Invalid s, should be positive: {s}"

        defaults = dict(rho=rho, s=s, adaptive=adaptive, **kwargs)
        super(EISAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        for group in self.param_groups:
            group['initial_s'] = group['s']
            group['initial_rho'] = group['rho']


    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            s = group["s"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p not in self.state:
                    self.state[p] = {}
                
                self.state[p]["initial_grad"] = p.grad.detach().clone()
                
                y_k = p.data - s * self.state[p]["initial_grad"]
                p.data.copy_(y_k)
                
                self.state[p]["y_k"] = y_k.clone()
                self.state[p]["old_p"] = y_k.clone()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        grad_norm_y = self._grad_norm_y()
        if grad_norm_y is None or grad_norm_y == 0:
            grad_norm_y = torch.tensor(1e-12).to(self.param_groups[0]["params"][0].device)
        
        for group in self.param_groups:
            rho = group["rho"]
            scale = rho / (grad_norm_y + 1e-12)

            for p in group["params"]:
                if p.grad is None or "initial_grad" not in self.state[p]:
                    continue
                
                y_k = self.state[p]["y_k"]
                initial_grad = self.state[p]["initial_grad"]
                
                e_w = (torch.pow(y_k, 2) if group["adaptive"] else 1.0) * initial_grad * scale.to(p.device)
                
                p.data = y_k + e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def third_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" not in self.state[p]:
                    continue
                p.data.copy_(self.state[p]["old_p"]) 
        
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def _grad_norm_y(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            adaptive = group["adaptive"]
            for p in group["params"]:
                if p.grad is None or "initial_grad" not in self.state[p] or "y_k" not in self.state[p]:
                    continue
                y_k = self.state[p]["y_k"]
                initial_grad = self.state[p]["initial_grad"]
                weight = torch.pow(y_k, 2) if adaptive else torch.ones_like(y_k)
                norms.append((weight * initial_grad).norm(p=2).to(shared_device))
        
        if not norms:
            return torch.tensor(0.0).to(shared_device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "EISAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=False)  
        self.second_step(zero_grad=True) 
        closure()  
        self.third_step()  

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
