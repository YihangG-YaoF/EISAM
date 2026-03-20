import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

# Disable BatchNorm running statistics updates
def disable_running_stats(model):
    """Disables momentum in BatchNorm layers to prevent running stats updates."""
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)

# Enable BatchNorm running statistics updates
def enable_running_stats(model):
    """Restores original momentum in BatchNorm layers to enable running stats updates."""
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
        

import math

class EISAMsScheduler:
    """
    为 EISAM 的 's' 参数提供调度器（独立实现，不继承 PyTorch 的 LRScheduler）。
    支持 'none'（默认）、'step'、'cosine'、'cosine_restart' 四种模式。
    """
    def __init__(self, optimizer, mode='none', 
                 step_size=None, gamma=0.1, 
                 T_max=None, s_min=1e-5, 
                 restart_period=None, mult_factor=2.0,  
                 last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.mode = mode.lower()
        self.step_size = step_size
        self.gamma = gamma
        self.T_max = T_max
        self.s_min = s_min
        self.restart_period = restart_period  
        self.mult_factor = mult_factor       
        self.verbose = verbose
        
        self.last_epoch = last_epoch
        self.cumulative = 0                  
        self.T_cur = 0.0                      
        
        if self.mode not in ['none', 'step', 'cosine', 'cosine_restart']:
            raise ValueError("mode 只能为 'none', 'step', 'cosine' 或 'cosine_restart'")
        if self.mode == 'step' and step_size is None:
            raise ValueError("'step' 模式需要提供 step_size")
        if self.mode in ['cosine', 'cosine_restart'] and T_max is None and restart_period is None:
            raise ValueError("'cosine' 或 'cosine_restart' 模式需要提供 T_max 或 restart_period")
        if self.mode == 'cosine_restart' and restart_period is None:
            raise ValueError("'cosine_restart' 模式需要提供 restart_period（初始周期）")

        if self.mode == 'cosine_restart':
            self.T_cur = float(self.restart_period)
            self.cumulative = 0

    def get_s(self):
        
        if self.mode == 'none':
            return [group['initial_s'] for group in self.optimizer.param_groups]

        epoch = self.last_epoch
        
        if self.mode == 'step':
            factor = self.gamma ** (epoch // self.step_size)
            return [group['initial_s'] * factor for group in self.optimizer.param_groups]

        if self.mode == 'cosine':
            if epoch >= self.T_max:
                return [self.s_min for _ in self.optimizer.param_groups]
            coeff = 0.5 * (1.0 + math.cos(math.pi * epoch / self.T_max))
            return [self.s_min + (group['initial_s'] - self.s_min) * coeff
                    for group in self.optimizer.param_groups]

        if self.mode == 'cosine_restart':
            if epoch == 0:
                return [group['initial_s'] for group in self.optimizer.param_groups]
            
            local_epoch = epoch - self.cumulative
            coeff = 0.5 * (1.0 + math.cos(math.pi * local_epoch / self.T_cur))
            return [self.s_min + (group['initial_s'] - self.s_min) * coeff
                    for group in self.optimizer.param_groups]

        return [group['initial_s'] for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        if self.mode == 'cosine_restart':
            if self.last_epoch > self.cumulative + self.T_cur:
                self.cumulative += self.T_cur
                self.T_cur *= self.mult_factor
                if self.verbose:
                    print(f"EISAMsScheduler: Cosine restart triggered! New period: {self.T_cur}")

        new_s_values = self.get_s()
        for group, new_s in zip(self.optimizer.param_groups, new_s_values):
            if self.verbose:
                print(f"EISAMSscheduler: Updating s from {group['s']:.2e} to {new_s:.2e} (epoch {self.last_epoch})")
            group['s'] = new_s


class EISAMrhoScheduler:
    """
    为 EISAM 的 'rho' 参数提供调度器（独立实现，不继承 PyTorch 的 LRScheduler）。
    支持 'none'（默认）、'step'、'cosine'、'cosine_restart' 四种模式。
    """
    def __init__(self, optimizer, mode='none', 
                 step_size=None, gamma=0.1, 
                 T_max=None, rho_min=1e-5, 
                 restart_period=None, mult_factor=2.0, 
                 last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.mode = mode.lower()
        self.step_size = step_size
        self.gamma = gamma
        self.T_max = T_max
        self.rho_min = rho_min
        self.restart_period = restart_period  
        self.mult_factor = mult_factor        
        self.verbose = verbose
        
        self.last_epoch = last_epoch
        self.cumulative = 0                   
        self.T_cur = 0.0                      
        
        if self.mode not in ['none', 'step', 'cosine', 'cosine_restart']:
            raise ValueError("mode 只能为 'none', 'step', 'cosine' 或 'cosine_restart'")
        if self.mode == 'step' and step_size is None:
            raise ValueError("'step' 模式需要提供 step_size")
        if self.mode in ['cosine', 'cosine_restart'] and T_max is None and restart_period is None:
            raise ValueError("'cosine' 或 'cosine_restart' 模式需要提供 T_max 或 restart_period")
        if self.mode == 'cosine_restart' and restart_period is None:
            raise ValueError("'cosine_restart' 模式需要提供 restart_period（初始周期）")

        if self.mode == 'cosine_restart':
            self.T_cur = float(self.restart_period)
            self.cumulative = 0

    def get_rho(self):
        if self.mode == 'none':
            return [group['initial_rho'] for group in self.optimizer.param_groups]

        epoch = self.last_epoch
        
        if self.mode == 'step':
            factor = self.gamma ** (epoch // self.step_size)
            return [group['initial_rho'] * factor for group in self.optimizer.param_groups]

        if self.mode == 'cosine':
            if epoch >= self.T_max:
                return [self.rho_min for _ in self.optimizer.param_groups]
            coeff = 0.5 * (1.0 + math.cos(math.pi * epoch / self.T_max))
            return [self.rho_min + (group['initial_rho'] - self.rho_min) * coeff
                    for group in self.optimizer.param_groups]

        if self.mode == 'cosine_restart':
            if epoch == 0:
                return [group['initial_rho'] for group in self.optimizer.param_groups]
            
            local_epoch = epoch - self.cumulative
            coeff = 0.5 * (1.0 + math.cos(math.pi * local_epoch / self.T_cur))
            return [self.rho_min + (group['initial_rho'] - self.rho_min) * coeff
                    for group in self.optimizer.param_groups]

        return [group['initial_rho'] for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        if self.mode == 'cosine_restart':
            if self.last_epoch > self.cumulative + self.T_cur:
                self.cumulative += self.T_cur
                self.T_cur *= self.mult_factor
                if self.verbose:
                    print(f"EISAMrhoScheduler: Cosine restart triggered! New period: {self.T_cur}")

        new_rho_values = self.get_rho()
        for group, new_rho in zip(self.optimizer.param_groups, new_rho_values):
            if self.verbose:
                print(f"EISAMrhoScheduler: Updating rho from {group['rho']:.2e} to {new_rho:.2e} (epoch {self.last_epoch})")
            group['rho'] = new_rho
