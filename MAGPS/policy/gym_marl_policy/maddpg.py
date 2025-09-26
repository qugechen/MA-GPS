import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import numpy as np
import torch
import torch.autograd.functional as F
from MAGPS.data import Batch, ReplayBuffer, to_torch_as
from MAGPS.exploration import BaseNoise, GaussianNoise
from MAGPS.policy.MARL_base import MARL_BasePolicy
from scipy.linalg import svd

from MAGPS.policy.gym_marl_policy.simplified_lq_guidance import MADDPG_iLQ_Guidance


def compute_FNE_LQ_pure_K(A, B_list, Q_list, R_list, T):
    num_agents = len(B_list)
    n = A.shape[0]
    m_list = [B.shape[1] for B in B_list]
    
    K = [[None for _ in range(num_agents)] for _ in range(T)]
    Z = [[np.zeros((n, n)) for _ in range(num_agents)] for _ in range(T + 1)]
    F = [None] * T
    for i in range(num_agents):
        Z[T][i] = Q_list[i] # terminal

    for_debug_use = [None] * T
    for iter in range(T):
        t = T - iter - 1  

        # import pdb; pdb.set_trace()
        R_block = np.block([
            [R_list[i] + B_list[i].T @ Z[t + 1][i] @ B_list[j] if i == j else B_list[i].T @ Z[t + 1][i] @ B_list[j]
            for j in range(num_agents)]
            for i in range(num_agents)
        ])
        
        BZB_block = np.vstack([
            B_list[i].T @ Z[t + 1][i] @ A for i in range(num_agents)
        ])
        sol_K = np.linalg.inv(R_block) @ BZB_block        
        for_debug_use[iter] = svd(R_block, compute_uv=False)
        
        start_idx = 0
        for i in range(num_agents):
            end_idx = start_idx + m_list[i]
            K[t][i] = sol_K[start_idx:end_idx, :]
            start_idx = end_idx
        
        F[t] = A - sum(B_list[i] @ K[t][i] for i in range(num_agents))
        
        for i in range(num_agents):
            Z[t][i] = F[t].T @ Z[t + 1][i] @ F[t] + K[t][i].T @ R_list[i] @ K[t][i] + Q_list[i]

    return K

class MADDPGPolicy(MARL_BasePolicy):
    """Implementation of Multi-Agent Deep Deterministic Policy Gradient."""
    def __init__(
        self,
        actor_list,
        actor_optim_list,
        critic_list,
        critic_optim_list,
        num_players: int,
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
        reward_normalization: bool = False,
        estimation_step: int = 1,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        action_dim: int = 2, # total action dimension
        device: str = "cpu",
        env: Optional[Any] = None,
        expert_policy = None,
        batch_size = None,
        pure_policy_regularization: bool = False,
        # behavior_policy_weight: float = 100.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs
        )
        assert action_bound_method != "tanh", "tanh mapping is not supported" \
            "in policies where action is used as input of critic , because" \
            "raw action in range (-inf, inf) will cause instability in training"
        
        for i in range(num_players):
            setattr(self, f"actor{i}", actor_list[i])
            setattr(self, f"actor_optim{i}", actor_optim_list[i])
            setattr(self, f"critic{i}", critic_list[i])
            setattr(self, f"critic_optim{i}", critic_optim_list[i])
            setattr(self, f"actor_old{i}", deepcopy(actor_list[i]))
            getattr(self, f"actor_old{i}").eval()
            setattr(self, f"critic_old{i}", deepcopy(critic_list[i]))
            getattr(self, f"critic_old{i}").eval()
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        self.tau = tau
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
        self._gamma = gamma
        self._noise = exploration_noise
        # it is only a little difference to use GaussianNoise
        # self.noise = OUNoise()
        self._rew_norm = reward_normalization
        self._n_step = estimation_step
        self.action_dim = action_dim
        self.nu = action_dim // num_players
        self.num_players = num_players
        self.device = device
        self.env = env
        if not self.env.is_nonlinear_game: 
            tmp = compute_FNE_LQ_pure_K(self.env.A, self.env.B_list, self.env.Q_list, self.env.R_list, 100)
            print("FNE_LQ_pure_K is computed!")
            self.KK = -np.vstack([tmp[0][i] for i in range(self.num_players)])
            print(self.KK)
            self.pure_policy_regularization = pure_policy_regularization
            if self.pure_policy_regularization:
                self.KK = 0*self.KK
        else: # NOTE: for nonlinear games, we have not implemented the expert policy yet
            self.KK = None
            self.pure_policy_regularization = pure_policy_regularization
            self.expert = MADDPG_iLQ_Guidance(device = device, env = env, batch_size = batch_size)
            


    def set_exp_noise(self, noise: Optional[BaseNoise]) -> None:
        """Set the exploration noise."""
        self._noise = noise

    # checked
    def train(self, mode: bool = True) -> "MADDPGPolicy": 
        """Set the module in training mode, except for the target network."""
        self.training = mode
        for i in range(self.num_players):
            getattr(self, f"actor{i}").train(mode)
            getattr(self, f"critic{i}").train(mode)
        return self

    # checked
    def sync_weight(self) -> None: 
        """Soft-update the weight for the target network."""
        for i in range(self.num_players):
            self.soft_update(getattr(self, f"actor_old{i}"), getattr(self, f"actor{i}"), self.tau)
            self.soft_update(getattr(self, f"critic_old{i}"), getattr(self, f"critic{i}"), self.tau)
    
    # checked
    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor: 
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        bcz = batch.obs_next.shape[0]
        target_q = torch.zeros(bcz, self.num_players, device=self.device)
        for i in range(self.num_players):
            target_q[:, i] = getattr(self, f"critic_old{i}")(
                batch.obs_next,
                self(batch, model=f"actor_old{i}", input="obs_next").act
            ).flatten()

        return target_q # shape: [bsz, num_players]

    # checked
    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch: 
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._n_step,
            self._rew_norm, self.num_players
        )
        return batch

    # checked
    def wrap_policy(self, batch_state):
        return torch.cat([getattr(self, f"actor{i}")(batch_state)[0] for i in range(self.num_players)], dim = 1)
    
    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "actor",
        input: str = "obs",
        player_index: int = 0,
        **kwargs: Any,
    ) -> Batch: 
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 2 keys:

            * ``act`` the action.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        obs = batch[input]
        actions = torch.zeros(obs.shape[0], self.action_dim, device=self.device)
        hidden = None
        if model == "actor":
            for i in range(self.num_players):
                model = getattr(self, f"actor{i}")
                actions[:, i*self.nu:(i+1)*self.nu], hidden = model(obs, state=state, info=batch.info)
        if model == "actor_old":
            for i in range(self.num_players):
                model = getattr(self, f"actor_old{i}")
                actions[:, i*self.nu:(i+1)*self.nu], hidden = model(obs, state=state, info=batch.info)
        if model == "actor_loss":
            for i in range(self.num_players):
                if i == player_index:
                    actions[:, player_index*self.nu:(player_index+1)*self.nu] = getattr(self, f"actor{player_index}")(obs, state=state, info=batch.info)[0]
                else:
                    model = getattr(self, f"actor_old{i}")
                    actions[:, i*self.nu:(i+1)*self.nu], hidden = model(obs, state=state, info=batch.info)
            
        return Batch(act=actions, state=None)

    @staticmethod
    # checked
    def _mse_optimizer(
        batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer, player_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        # weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns[:, player_index].flatten()
        td = current_q - target_q
        critic_loss = (td.pow(2) *torch.abs(td.detach()) ).mean() # change to weighted mse loss by Jingqi, * weight
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return td, critic_loss

    def actor_loss(self, batch: Batch, i: int, expert_action: None, behavior_loss_weight=None) -> torch.Tensor:
       
        if not self.env.is_nonlinear_game:            
            action_error = expert_action[:, i*self.nu:(i+1)*self.nu] - self(batch).act[:, i*self.nu:(i+1)*self.nu]
            bc_loss = (action_error.pow(2)*behavior_loss_weight).mean()
            pure_actor_loss = -getattr(self, f"critic{i}")(batch.obs, self(batch, model="actor_loss", player_index=i).act)
        else: # NOTE: for nonlinear games, we have not implemented the expert policy yet
            action_error = expert_action[:, i*self.nu:(i+1)*self.nu] - self(batch).act[:, i*self.nu:(i+1)*self.nu]
            # create a vector with random binary values
            binary_vector = torch.randint(0, 2, (batch.obs.shape[0], 1), device=self.device)
            bc_loss = (action_error.pow(2)*behavior_loss_weight*binary_vector).mean()
            pure_actor_loss = -getattr(self, f"critic{i}")(batch.obs, self(batch, model="actor_loss", player_index=i).act)

        return (pure_actor_loss ).mean() + bc_loss, bc_loss# *index.float()
        # return (-getattr(self, f"critic{i}")(batch.obs, self(batch, eps=0.0).act) + behavior_loss*index.float()).mean()
        
    # checked
    def learn(self, batch: Batch, behavior_loss_weight=None, **kwargs: Any) -> Dict[str, float]:
        # critic
        print_dict = {}
        random_index = np.random.permutation(self.num_players)
        for i in random_index:
            td, critic_loss = self._mse_optimizer(batch, getattr(self, f"critic{i}"), getattr(self, f"critic_optim{i}") , i)
            print_dict[f"loss/critic{i}"] = critic_loss.item()

        if not self.env.is_nonlinear_game:
            KK = self.KK
            expert_action = torch.tensor(batch.obs @ KK.T, device= self.device)
        else: # NOTE: for nonlinear games, we have not implemented the expert policy yet
            # import pdb; pdb.set_trace()
            if self.pure_policy_regularization:
                expert_action = torch.zeros(batch.obs.shape[0], self.action_dim, device=self.device)
            else:
                expert_action = self.expert.evaluate_expert_action(torch.tensor(batch.obs,requires_grad=False), self.wrap_policy).detach()
                # print("this is done!")
            
        # actor_old_action = self(batch, model="actor_old").act
        # create actor loss list
        for i in random_index:
            actor_loss, bc_loss = self.actor_loss(batch, i, expert_action, behavior_loss_weight)
            getattr(self, f"actor_optim{i}").zero_grad()
            actor_loss.backward()
            # print out the norm of the gradient
            # print(torch.norm(getattr(self, f"actor{i}").fc1.weight.grad).item())
            getattr(self, f"actor_optim{i}").step()
            print_dict[f"loss/actor{i}"] = actor_loss.item() - bc_loss.item() # this gives us the pure actor loss
            print_dict[f"loss/behavior{i}"] = bc_loss.item()
            # import pdb; pdb.set_trace()
            # .preprocess.model.model[0].weight.grad
            print_dict[f"loss/actor{i}_grad_norm"] = torch.norm(getattr(self, f"actor{i}").last.model[0].weight.grad).item()
            
            
        self.sync_weight()
        return print_dict


    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        if self._noise is None:
            return act
        if isinstance(act, np.ndarray):
            return act + self._noise(act.shape)
        warnings.warn("Cannot add exploration noise to non-numpy_array action.")
        return act
