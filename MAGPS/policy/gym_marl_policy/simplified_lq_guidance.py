from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union
from MAGPS.policy.gym_marl_policy.ilq_games_base import LQGame, lq_solver_batch
import torch

class MADDPG_iLQ_Guidance():
    def __init__(
        self, 
        device: str = "cpu",
        env: Optional[Any] = None, 
        batch_size: int = 64,
        **kwargs: Any):
        # define the NN dynamics
        self.device = torch.device(device)
        torch.set_default_device(self.device)
        self.env = env
        self.nx = env.total_state_dim
        self.nu = env.total_action_dim
        self.basis = torch.eye(self.nx)
        self.basis_u = torch.eye(self.nu)
        self.horizon = 20
        self.n_players = env.num_players
        self.players_u_index_list = env.players_u_index_list.to(self.device)
        self.dtype = torch.float32
        self.A = torch.zeros((self.horizon, self.nx, self.nx), device=device, dtype=self.dtype)
        self.B = torch.zeros((self.horizon, self.nx, self.nu), device=device, dtype=self.dtype)
        self.Q = torch.zeros((self.horizon+1, self.n_players, self.nx, self.nx), device=device, dtype=self.dtype) # t=0 to t = horizon
        self.R = torch.zeros((self.horizon, self.n_players, self.nu, self.nu), device=device, dtype=self.dtype) # t=0 to t = horizon-1
        self.q = torch.zeros((self.horizon+1, self.n_players, self.nx, 1), device=device, dtype=self.dtype) # offset term r.T @ x in the cost function
        self.r = torch.zeros((self.horizon, self.n_players, self.nu, 1), device=device, dtype=self.dtype) # offset term u.T @ u in the cost function
        # define the initial policy parameters
        self.P = torch.zeros((self.horizon, self.nu, self.nx), device=device, dtype=self.dtype) # P @ x + a = u
        self.a = torch.zeros((self.horizon, self.nu, 1), device=device, dtype=self.dtype)
         # the index of the control inputs that each player can access
        self.lq_approx = LQGame(self.nx, self.nu, self.horizon, self.n_players, self.A, self.B, self.Q, self.R, self.q, self.r, self.players_u_index_list)
        self.S = torch.zeros((self.nu, self.nu), device=device, dtype=self.dtype)
        self.YP = torch.zeros((self.nu, self.nx), device=device, dtype=self.dtype)
        self.Ya = torch.zeros((self.nu, 1), device=device, dtype=self.dtype)
        self.x_traj = torch.zeros((self.horizon+1, self.nx), device=device, dtype=self.dtype)
        self.u_traj = torch.zeros((self.horizon, self.nu), device=device, dtype=self.dtype)
        self.batch_size = batch_size
        self.P_batch = self.P.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        self.a_batch = self.a.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        self.A_batch = self.A.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        self.B_batch = self.B.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        self.Q_batch = self.Q.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        self.R_batch = self.R.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        self.q_batch = self.q.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        self.r_batch = self.r.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        self.S_batch = self.S.unsqueeze(0).repeat(batch_size, 1, 1)
        self.YP_batch = self.YP.unsqueeze(0).repeat(batch_size, 1, 1)
        self.Ya_batch = self.Ya.unsqueeze(0).repeat(batch_size, 1, 1)
        
    
    def simulate_a_policy_and_return_a_traj_and_lq_approx_batch_version(
        self,
        x_batch: torch.Tensor,
        u_batch: torch.Tensor,
        policy_NN: torch.nn.Module
        ):
        batch_size = min(x_batch.shape[0], self.batch_size)
        for t in range(self.horizon):
            xt_batch = x_batch[:batch_size, t, :].clone().detach().requires_grad_(True).reshape(batch_size, self.nx)
            ut_batch = policy_NN(xt_batch)
            x_batch[:batch_size, t+1, :] = self.env.dynamics(xt_batch, ut_batch) # NOTE: should change here!!
            u_batch[:batch_size, t, :] = ut_batch
            # evaluate the dynamics jacobian
            xt = x_batch[:batch_size, t, :].clone().detach().requires_grad_(True).reshape(batch_size, self.nx)
            ut = u_batch[:batch_size, t, :].clone().detach().requires_grad_(True).reshape(batch_size, self.nu)
            # var = torch.cat((xt, ut), 1)
            dyn_jac = self.env.dynamics_jacobian(xt,ut).reshape(batch_size, self.nx, self.nx+self.nu)
            self.A_batch[:batch_size, t, :, :] = dyn_jac[:batch_size, :, :self.nx].clone().detach().reshape(batch_size, self.nx, self.nx)
            self.B_batch[:batch_size, t, :, :] = dyn_jac[:batch_size, :, self.nx:].clone().detach().reshape(batch_size, self.nx, self.nu)        
            # evaluate the cost function
            x_next_tensor_t0 = x_batch[:batch_size, t+1, :].clone().detach().requires_grad_(True).reshape(batch_size, self.nx)
            var_cost_t0 = torch.cat((x_next_tensor_t0, ut), 1)
            cost_jac_t0, cost_hess_t0 = self.env.costs_jacobian_and_hessian(var_cost_t0)
            
            x_next_tensor = x_batch[:batch_size, t+1, :].clone().detach().requires_grad_(True).reshape(batch_size, self.nx)
            var_cost = torch.cat((x_next_tensor, ut), 1)
            cost_jac, cost_hess = self.env.costs_jacobian_and_hessian(var_cost)
            for ii in range(self.n_players):
                # evaluate the gradient
                if t == 0:
                    self.q_batch[:batch_size, t, ii, :, :] = cost_jac_t0[ii, :, :self.nx].detach().reshape(batch_size, self.nx, 1) # q_{t}
                    self.Q_batch[:batch_size, t, ii, :, :] = cost_hess_t0[ii, :, :self.nx, :self.nx].detach().reshape(batch_size, self.nx, self.nx) # Q_{t}
                
                self.q_batch[:batch_size, t+1, ii, :, :] = cost_jac[ii, :, :self.nx].detach().reshape(batch_size, self.nx,1) # q_{t+1}
                self.r_batch[:batch_size, t, ii, :, :] = cost_jac[ii, :, self.nx:].detach().reshape(batch_size, self.nu,1)
                # evaluate the hessian
                self.Q_batch[:batch_size, t+1, ii, :, :]= cost_hess[ii, :, :self.nx, :self.nx].detach().reshape(batch_size, self.nx, self.nx) # Q_{t+1}
                self.R_batch[:batch_size, t, ii, :, :] = cost_hess[ii, :, self.nx:, self.nx:].detach().reshape(batch_size, self.nu, self.nu)
        return x_batch, u_batch
    
    def evaluate_expert_action(self, batch_state, policy_NN):
        batch_size = min(batch_state.shape[0], self.batch_size)
        x_batch = torch.zeros((batch_size, self.horizon+1, self.nx), device=self.device)
        u_batch = torch.zeros((batch_size, self.horizon, self.nu), device=self.device)
        x_batch[:, 0, :] = batch_state[:batch_size, :]
        x_batch, u_batch = self.simulate_a_policy_and_return_a_traj_and_lq_approx_batch_version(x_batch, u_batch, policy_NN)
        # import pdb; pdb.set_trace()
        self.P_tmp, self.a_tmp = lq_solver_batch(
            self.P_batch[:batch_size, :, :, :], 
            self.a_batch[:batch_size, :, :],
            self.A_batch[:batch_size, :, :, :], 
            self.B_batch[:batch_size, :, :, :],
            self.Q_batch[:batch_size, :, :, :],
            self.R_batch[:batch_size, :, :, :],
            self.q_batch[:batch_size, :, :, :],
            self.r_batch[:batch_size, :, :, :],
            self.players_u_index_list, self.nx, self.nu, self.horizon, self.n_players,
            self.S_batch[:batch_size, :, :], 
            self.YP_batch[:batch_size, :, :],
            self.Ya_batch[:batch_size, :, :],
        )
        return u_batch[:batch_size, 0, :].reshape(batch_size, self.nu) - self.a_tmp[:batch_size, 0, :, :].reshape(batch_size, self.nu)
    





