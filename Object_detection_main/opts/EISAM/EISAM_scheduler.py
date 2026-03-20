import math

class ESAMScheduler:
    def __init__(self, optimizer, mode='cosine', 
                 step_size=None, gamma=0.1, 
                 T_max=None, s_min=1e-6, 
                 restart_period=None, mult_factor=2.0,
                 warmup_ratio=0.0,   
                 warmup_start_factor=0.01, 
                 last_epoch=-1, verbose=False):
        
        self.optimizer = optimizer
        self.mode = mode.lower()
        self.step_size = step_size
        self.gamma = gamma
        self.T_max = T_max
        self.s_min = s_min
        self.restart_period = restart_period
        self.mult_factor = mult_factor
        self.warmup_ratio = max(0.0, min(1.0, warmup_ratio))
        self.warmup_start_factor = warmup_start_factor
        self.verbose = verbose
        
        self.last_epoch = last_epoch
        self.cumulative = 0
        self.T_cur = 0.0
        
        self.warmup_steps = 0
        if self.T_max is not None and self.warmup_ratio > 0:
            self.warmup_steps = int(self.T_max * self.warmup_ratio)

        if self.mode not in ['none', 'step', 'cosine', 'cosine_restart']:
            raise ValueError("mode only for 'none', 'step', 'cosine' or 'cosine_restart'")
        if self.mode == 'cosine_restart' and restart_period is None:
            raise ValueError("'cosine_restart' needs restart_period")

        if self.mode == 'cosine_restart':
            self.T_cur = float(self.restart_period)
            self.cumulative = 0

        if self.verbose and self.warmup_ratio > 0:
            print(f"ESAMScheduler: Warmup enabled | ratio={self.warmup_ratio} | warmup_steps={self.warmup_steps}")

    def get_s(self):
        if self.mode == 'none':
            return [group['initial_s'] for group in self.optimizer.param_groups]

        epoch = self.last_epoch
        groups = self.optimizer.param_groups

        if epoch < self.warmup_steps and self.warmup_steps > 0:
            progress = epoch / self.warmup_steps
            return [
                group['initial_s'] * self.warmup_start_factor +
                (group['initial_s'] - group['initial_s'] * self.warmup_start_factor) * progress
                for group in groups
            ]

        effective_epoch = epoch - self.warmup_steps if self.warmup_steps > 0 else epoch

        if self.mode == 'step':
            factor = self.gamma ** (effective_epoch // self.step_size)
            return [group['initial_s'] * factor for group in groups]

        if self.mode == 'cosine':
            if effective_epoch >= (self.T_max - self.warmup_steps):
                return [self.s_min for _ in groups]
            coeff = 0.5 * (1.0 + math.cos(math.pi * effective_epoch / max(1, self.T_max - self.warmup_steps)))
            return [self.s_min + (group['initial_s'] - self.s_min) * coeff for group in groups]

        if self.mode == 'cosine_restart':
            if epoch == 0:
                return [group['initial_s'] for group in groups]
            local_epoch = epoch - self.cumulative
            coeff = 0.5 * (1.0 + math.cos(math.pi * local_epoch / self.T_cur))
            return [self.s_min + (group['initial_s'] - self.s_min) * coeff for group in groups]

        return [group['initial_s'] for group in groups]

    def step(self, step=None):
        if step is not None:
            self.last_epoch = step
        else:
            self.last_epoch += 1

        if self.mode == 'cosine_restart':
            if self.last_epoch > self.cumulative + self.T_cur:
                self.cumulative += self.T_cur
                self.T_cur *= self.mult_factor
                if self.verbose:
                    print(f"ESAMScheduler: Cosine restart triggered! New period: {self.T_cur}")

        new_s_values = self.get_s()
        for group, new_s in zip(self.optimizer.param_groups, new_s_values):
            group['s'] = new_s
            if self.verbose:
                print(f"s updated to {new_s:.2e} at step {self.last_epoch}")


class ESAMrhoScheduler:
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
            raise ValueError("mode 只能为 'none', 'step', 'cosine' or 'cosine_restart'")
        if self.mode == 'step' and step_size is None:
            raise ValueError("'step' needs step_size")
        if self.mode in ['cosine', 'cosine_restart'] and T_max is None and restart_period is None:
            raise ValueError("'cosine' or 'cosine_restart' needs T_max or restart_period")
        if self.mode == 'cosine_restart' and restart_period is None:
            raise ValueError("'cosine_restart' needs restart_period")

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
                    print(f"ESAMrhoScheduler: Cosine restart triggered! New period: {self.T_cur}")

        new_rho_values = self.get_rho()
        for group, new_rho in zip(self.optimizer.param_groups, new_rho_values):
            if self.verbose:
                print(f"ESAMrhoScheduler: Updating rho from {group['rho']:.2e} to {new_rho:.2e} (epoch {self.last_epoch})")
            group['rho'] = new_rho
