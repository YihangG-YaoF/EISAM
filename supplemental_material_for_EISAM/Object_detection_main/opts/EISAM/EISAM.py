from tokenize import group

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


# ====================== 新版本 EISAM (EISAM_2) ======================
class EISAM(torch.optim.Optimizer):
    """
    新版本 EISAM (EISAM_2) - 符合你描述的“新思想”实现
    
    核心流程（与你意图完全一致）：
    1. first_step：基于初始梯度 delta0 把参数**实际更新**到 y (theta1)，并保存状态
    2. second_step：基于初始梯度 delta0 在 y 的位置施加扰动，临时移动到扰动点 (theta2)
    3. closure()：在扰动点计算新梯度 delta1
    4. third_step：恢复到 y (theta1)，然后调用 base_optimizer.step() 进行最终更新 → theta3
    
    与论文 Algorithm 1（原版探测式）完全不同，这是“先 commit y 再更新”的变体。
    """
    def __init__(self, params, base_optimizer, rho=0.05, s=0.01, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert s > 0.0, f"Invalid s, should be positive: {s}"

        defaults = dict(rho=rho, s=s, adaptive=adaptive, **kwargs)
        super(EISAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        # 新增：备份初始 s，用于后续调度恢复初始值
        for group in self.param_groups:
            group['initial_s'] = group['s']
            group['initial_rho'] = group['rho']


    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """第一步：用 initial_grad 把参数实际移动到 y (theta1)，保存 y_k 和 initial_grad"""
        for group in self.param_groups:
            s = group["s"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p not in self.state:
                    self.state[p] = {}
                
                # 保存初始梯度（delta0）
                self.state[p]["initial_grad"] = p.grad.detach().clone()
                
                # 计算并实际移动到 y (theta1)
                y_k = p.data - s * self.state[p]["initial_grad"]
                p.data.copy_(y_k)
                
                # 保存 y_k（用于 adaptive scaling 和 _grad_norm_y）
                self.state[p]["y_k"] = y_k.clone()
                # 预存 old_p（用于 third_step 恢复）
                self.state[p]["old_p"] = y_k.clone()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """第二步：基于 initial_grad（而非当前 p.grad）在 y 的位置施加扰动"""
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
                
                # 关键：使用 initial_grad 计算扰动（符合你描述）
                e_w = (torch.pow(y_k, 2) if group["adaptive"] else 1.0) * initial_grad * scale.to(p.device)
                
                # 临时移动到扰动点 (theta2)
                p.data = y_k + e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def third_step(self, zero_grad=False):
        """第三步：恢复到 y (theta1)，然后用 base_optimizer 基于新梯度更新"""
        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" not in self.state[p]:
                    continue
                p.data.copy_(self.state[p]["old_p"])  # 恢复到 theta1
        
        self.base_optimizer.step()  # 用扰动点算出的新梯度更新参数

        if zero_grad:
            self.zero_grad()

    def _grad_norm_y(self):
        """使用保存的 y_k 和 initial_grad 计算范数（支持 adaptive=True）"""
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
        """
        完整优化步（新版本流程）
        """
        assert closure is not None, "EISAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=False)   # 实际移动到 y (theta1)
        self.second_step(zero_grad=True)  # 施加扰动
        closure()                         # 在扰动点计算新梯度
        self.third_step()                 # 恢复 + 最终更新

    def load_state_dict(self, state_dict):
        """加载优化器状态"""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups