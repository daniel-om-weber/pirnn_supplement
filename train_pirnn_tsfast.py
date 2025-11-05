# %%
import torch
import torch.nn as nn
from tsfast.basics import create_dls, SkipNLoss, fun_rmse, RNNLearner, zero_loss
from fastai.basics import *

# Load data using tsfast's create_dls
dls = create_dls(
    u=['u'],  # Input signal names
    y=['x','v'],  # Output signal names
    dataset=Path("dataset"),
    win_sz=1000,  # Full sequence length
    stp_sz=1,  # Non-overlapping windows
    bs=16,
)

# Physical parameters (must match dataset generation)
MASS = 1.0
SPRING_CONSTANT = 1.0
DAMPING_COEFFICIENT = 0.1
DT = 0.01

def physics_residual(y_pred, u, m, k, c, dt):
    y_pred_aligned = y_pred[:, 1:-1, 0:1]
    u_aligned = u[:, 1:-1]
    dy_dt = (y_pred[:, 2:] - y_pred[:, :-2]) / (2 * dt)
    d2y_dt2 = (y_pred[:, 2:] - 2 * y_pred[:, 1:-1] + y_pred[:, :-2]) / (dt**2)
    return m * d2y_dt2 + c * dy_dt + k * y_pred_aligned - u_aligned 

class PhysicsLossCallback(Callback):
    def __init__(self, weight=0.1): # dt for Silverbox is 1/1024s, approximated here
        self.weight = weight

    def after_loss(self):
        if not self.training: return # Only apply during training

        # Get model predictions (y) and inputs (u)
        y_pred = self.learn.pred[:,:,0:1]
        u = self.learn.xb[0][:,:,:] * self.learn.dls.train.after_batch[0].std + self.learn.dls.train.after_batch[0].mean
        
        physics_loss = torch.mean(physics_residual(y_pred, u, MASS, SPRING_CONSTANT, DAMPING_COEFFICIENT, DT)**2)
        initial_loss = torch.mean((y_pred[:, 0, 0]**2))

        self.learn.loss = self.learn.loss + self.weight * physics_loss + initial_loss
        self.learn.loss_grad = self.learn.loss_grad + self.weight * physics_loss + initial_loss

def generate_random_input(batch_size: int, seq_length: int, 
                         device: torch.device) -> torch.Tensor:
    """Generate random input sequences for collocation points."""
    u = torch.zeros(batch_size, seq_length, device=device)
    
    for i in range(batch_size):
        choice = np.random.randint(0, 5)
        
        if choice == 0:  # Sine wave
            freq = np.random.uniform(0.3, 3.0)
            amp = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2*np.pi)
            t = torch.arange(seq_length, device=device) * DT
            u[i] = amp * torch.sin(2 * np.pi * freq * t + phase)
        
        elif choice == 1:  # Step
            step_time = np.random.randint(100, 500)
            amp = np.random.uniform(-2.0, 2.0)
            u[i, step_time:] = amp
        
        elif choice == 2:  # Ramp
            start_time = np.random.randint(50, 300)
            slope = np.random.uniform(-0.5, 0.5)
            t = torch.arange(seq_length, device=device) * DT
            ramp = torch.clamp(slope * (t - start_time * DT), -2.0, 2.0)
            ramp[:start_time] = 0
            u[i] = ramp
        
        elif choice == 3:  # Random noise
            noise = torch.randn(seq_length, device=device)
            u[i] = noise * np.random.uniform(0.3, 0.8)
        
        else:  # Multi-sine
            for _ in range(3):
                freq = np.random.uniform(0.2, 2.0)
                amp = np.random.uniform(0.2, 0.5)
                t = torch.arange(seq_length, device=device) * DT
                u[i] += amp * torch.sin(2 * np.pi * freq * t)
    
    return u.unsqueeze(-1)  # [batch, seq_len, 1]

class CollocationPointsCB(Callback):
    def __init__(self, norm_input, weight: float = 0.5):
        self.mean = norm_input.mean
        self.std = norm_input.std
        self.weight = weight
    
    def after_loss(self):
        if not self.training: return
        
        # Get batch size and sequence length from current batch
        u_real = self.xb[0]  # [batch, seq_len, 1]
        batch_size = u_real.shape[0]
        seq_length = u_real.shape[1]
        device = u_real.device
        
        # Generate random input trajectories in physical space
        u_coloc_phys = generate_random_input(batch_size, seq_length, device)
        
        # Normalize input for model (model expects normalized inputs): x_norm = (x - mean) / std
        # Ensure norm tensors are on same device
        mean = self.mean.to(device)
        std = self.std.to(device)
        u = (u_coloc_phys - mean) / std
        
        # Forward pass through model (starting from x0=v0=0)
        with torch.enable_grad():
            y_pred = self.learn.model(u)
        
        physics_loss = torch.mean(physics_residual(y_pred, u, MASS, SPRING_CONSTANT, DAMPING_COEFFICIENT, DT)**2)
        initial_loss = torch.mean((y_pred[:, 0, 0]**2))

        self.learn.loss = self.learn.loss + self.weight * physics_loss + initial_loss
        self.learn.loss_grad = self.learn.loss_grad + self.weight * physics_loss + initial_loss


def dummy_loss(pred, targ):
    return torch.tensor(0.0, device=pred.device, requires_grad=True)
# %%
# skip_n = 0
# learn = RNNLearner(dls,
#                    rnn_type='lstm',
#                    num_layers=1,
#                    hidden_size=64,
#                    loss_func=SkipNLoss(mae,skip_n), # Use a standard loss
#                    metrics=[SkipNLoss(fun_rmse,skip_n)]) # Add the callback here

learn = RNNLearner(dls,
                   rnn_type='lstm',
                   num_layers=1,
                   hidden_size=64,
                   loss_func=zero_loss, # Use a standard loss
                   metrics=[fun_rmse]) # Add the callback here
learn.add_cb(PhysicsLossCallback(weight=1))
# learn.add_cb(CollocationPointsCB(norm_input=dls.train.after_batch[0], weight=0.01))

learn.fit_flat_cos(100, 3e-4)
# learn.lr_find()
# %%
learn.show_results(max_n=3,ds_idx=0)
# %%
