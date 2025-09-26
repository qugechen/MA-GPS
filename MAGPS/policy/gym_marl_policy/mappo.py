from typing import Any, Dict, List, Optional, Union, Type

import numpy as np
import torch
from torch import nn
# relationship: PPOPolicy -> A2CPolicy -> -> PGPolicy -> BasePolicy
# NOTE!!!!!!
# new functions: _compute_returns, forward, learn
# need to change compute_episodic_return in MARL_base

from MAGPS.data import Batch, ReplayBuffer, to_torch_as
from MAGPS.policy.MARL_base import MARL_BasePolicy

from MAGPS.utils.net.common import ActorCritic
from MAGPS.utils import RunningMeanStd
from MAGPS.policy.gym_marl_policy.simplified_lq_guidance import MADDPG_iLQ_Guidance


class MAPPOPolicy(MARL_BasePolicy):
    r"""Implementation of Proximal Policy Optimization with Single Critic. 
    
    This version uses a single critic network that optimizes the sum of costs for all agents,
    while maintaining multiple actor networks for each agent.

    :param torch.nn.Module actor_list: list of actor networks for each agent
    :param torch.nn.Module critic: single critic network that evaluates global state
    :param torch.optim.Optimizer optim_list: list of optimizers for each actor
    :param torch.optim.Optimizer critic_optim: optimizer for the single critic
    :param int num_players: number of agents
    :param int nu: action dimension of each player
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper. Default to 0.2.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound.
        Default to 5.0 (set None if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553v3 Sec. 4.1.
        Default to True.
    :param bool advantage_normalization: whether to do per mini-batch advantage
        normalization. Default to True.
    :param bool recompute_advantage: whether to recompute advantage every update
        repeat according to https://arxiv.org/pdf/2006.05990.pdf Sec. 3.5.
        Default to False.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param float max_grad_norm: clipping gradients in back propagation. Default to
        None.
    :param float gae_lambda: in [0, 1], param for Generalized Advantage Estimation.
        Default to 0.95.
    :param bool reward_normalization: normalize estimated values to have std close
        to 1, also normalize the advantage to Normal(0, 1). Default to False.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the model;
        should be as large as possible within the memory constraint. Default to 256.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor_list,
        critic,  # Single critic instead of critic_list
        optim_list,
        critic_optim,  # Single critic optimizer
        num_players,
        nu, # action dim of each player
        dist_fn: Type[torch.distributions.Distribution],
        discount_factor: float = 0.99,
        max_grad_norm: Optional[float] = None,
        eps_clip: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        reward_normalization: bool = False,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        deterministic_eval: bool = False,
        device: Optional[torch.device] = None,
        pure_policy_regulation: bool = False,
        no_gd_regularization: bool = False,
        env = None,
        batch_size = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(dist_fn, **kwargs)
        self._eps_clip = eps_clip
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        
        # Set up actors (same as before)
        for i in range(num_players):
            setattr(self, f"actor{i}", actor_list[i])
            setattr(self, f"optim{i}", optim_list[i])
        
        # Set up single critic
        self.critic = critic
        self.critic_optim = critic_optim
        
        # Create actor-critic pairs for each actor with the shared critic
        for i in range(num_players):
            setattr(self, f"_actor_critic{i}", ActorCritic(getattr(self, f"actor{i}"), self.critic))
        
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        self.num_players = num_players
        self.nu = nu
        self._gamma = discount_factor
        self._grad_norm = max_grad_norm
        self._weight_vf = vf_coef
        self._weight_ent = ent_coef
        self._lambda = gae_lambda
        self._batch = max_batchsize
        self._rew_norm = reward_normalization
        self._deterministic_eval = deterministic_eval
        self.device = device
        self.dist_fn = dist_fn
        
        # Single running mean std for the global critic
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self.player_list = np.arange(self.num_players)
        self.step = 0
        self.pure_policy_regulation = pure_policy_regulation
        self.no_gd_regularization = no_gd_regularization
        if self.pure_policy_regulation == False:
            self.expert = MADDPG_iLQ_Guidance(device = device, env = env, batch_size = batch_size)
    

    def _compute_returns(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Compute returns using single critic that evaluates sum of costs."""
        bsz = len(batch.obs)
        
        # Compute values using single critic (outputs scalar V(s))
        v_s_tmp, v_s__tmp = [], []
        batch.v_s = torch.zeros(bsz, device=self.device) 
        batch.v_s_ = torch.zeros(bsz, device=self.device)
        batch.returns = torch.zeros(bsz, device=self.device) 
        
        with torch.no_grad():
            for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                v_s_tmp.append(self.critic(minibatch.obs))
                v_s__tmp.append(self.critic(minibatch.obs_next))
        
        batch.v_s = torch.cat(v_s_tmp, dim=0).flatten()  # old value (scalar)
        v_s_1d = torch.cat(v_s__tmp, dim=0).flatten().cpu().numpy()  # next value (scalar)
        
        # Convert to numpy for GAE computation
        v_s = batch.v_s.cpu().numpy()
        v_s_ = v_s_1d

        # Sum costs first to a single global cost, then compute GAE once via base util
        individual_cost = batch.info.individual_cost  # (bsz, num_players)
        global_cost = individual_cost.sum(axis=1)  # (bsz,)

        tmp_batch = Batch(
            obs=batch.obs,
            obs_next=batch.obs_next,
            terminated=batch.terminated,
            truncated=batch.truncated,
            info=Batch(individual_cost=global_cost.reshape(-1, 1)),
        )
        
        v_s_1 = v_s.reshape(-1, 1)
        v_s__1 = v_s_.reshape(-1, 1)
        
        returns_1, adv_1 = MARL_BasePolicy.compute_episodic_return(
            tmp_batch, buffer, indices,
            v_s_=v_s__1, v_s=v_s_1,
            gamma=self._gamma, gae_lambda=self._lambda, num_players=1,
        )
        global_advantage = adv_1[:, 0]
        global_returns = returns_1[:, 0]
        # if not np.isfinite(global_returns).all():
        #     print("[WARN] global_returns contains non-finite values; applying nan_to_num.")
        #     global_returns = np.nan_to_num(global_returns, nan=0.0, posinf=1e6, neginf=-1e6)
        # if not np.isfinite(global_returns).all():
        #     print("[WARN] global_returns contains non-finite values; applying nan_to_num.")
        #     global_returns = np.nan_to_num(global_returns, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if self._rew_norm:
            batch.returns = torch.tensor(global_returns / np.sqrt(self.ret_rms.var + self._eps), device = self.device)
            self.ret_rms.update(global_returns)
        else:
            batch.returns = torch.tensor(global_returns, device = self.device)
        
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        # All agents use the same global advantage
        batch.adv = to_torch_as(global_advantage, batch.v_s)
        return batch
    
    
    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        if self._recompute_adv:
            # buffer input `buffer` and `indices` to be used in `learn()`.
            self._buffer, self._indices = buffer, indices
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        with torch.no_grad():
            bsz = batch.act.shape[0]
            out = self(batch)
            tmp_dist = out.dist  # Independent(Normal, event_dim=1) over all dims
            base_logp = tmp_dist.base_dist.log_prob(batch.act)  # [bsz, total_dim]

            batch.logp_old = base_logp.view(bsz, self.num_players, self.nu).sum(dim=-1)  # [bsz, num_players]
                # check if logp_old has nan
                # if torch.isnan(batch.logp_old[:,i]).any():
                #     import pdb; pdb.set_trace()
        return batch
    
    def wrap_policy(self, batch_state):
        return self(Batch(obs = batch_state)).logits[0]
    
    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        bsz = len(batch.obs)
        logits_mean = torch.zeros(bsz, self.num_players * self.nu, device=self.device)
        logits_std = torch.zeros(bsz, self.num_players * self.nu, device=self.device)
        hidden = None
        for i in range(self.num_players):
            tmp, hidden = getattr(self, f"actor{i}")(batch.obs, state=state)
            logits_mean[:, i*self.nu:(i+1)*self.nu] = tmp[0]
            logits_std[:, i*self.nu:(i+1)*self.nu] = tmp[1]   
        logits = (logits_mean, logits_std)
        
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logits.argmax(-1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)
    
    def learn(  # type: ignore
        self, batch: Batch, behavior_loss_weight, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        print_dict = {}
        losses, clip_losses, vf_losses, ent_losses = [[] for i in range(self.num_players)], [[] for i in range(self.num_players)], [[] for i in range(self.num_players)], [[] for i in range(self.num_players)]
        bc_mean_losses, bc_var_losses = [[] for i in range(self.num_players)], [[] for i in range(self.num_players)]

        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                # calculate loss for actor
                # randomly shuffle the players
                np.random.shuffle(self.player_list)
                if self.pure_policy_regulation == False and self.no_gd_regularization == False:
                    with torch.no_grad():
                        expert_action = self.expert.evaluate_expert_action(torch.tensor(minibatch.obs,requires_grad=False), self.wrap_policy).detach()
                        expert_action = torch.clamp(expert_action, min=-1.0, max=1.0)
                
                # Update single critic first
                # Critic learns to predict global returns (sum of all agents' cost returns)
                critic_value = self.critic(minibatch.obs).flatten()
                if self._value_clip:
                    v_clip = minibatch.v_s + \
                        (critic_value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - critic_value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    critic_vf_loss = torch.max(vf1, vf2).mean()
                else:
                    critic_vf_loss = (minibatch.returns - critic_value).pow(2).mean()
                
                # Update critic
                self.critic_optim.zero_grad()
                critic_vf_loss.backward()
                
                # Update actors
                for j in range(self.num_players):
                    i = self.player_list[j]
                    tmp = self(minibatch)
                    tmp_dist = tmp.dist
                    tmp_action_mean = tmp.logits[0]
                    tmp_action_std = tmp.logits[1]
                    # Reconstruct per-agent dist using mean and stddev
                    dist = self.dist_fn(
                        tmp_dist.mean[:, i*self.nu:(i+1)*self.nu],
                        tmp_dist.stddev[:, i*self.nu:(i+1)*self.nu]
                    )
                    
                    # Use the same advantage for all agents (from single critic)
                    if self._norm_adv:
                        mean, std = minibatch.adv.mean(), minibatch.adv.std()
                        minibatch.adv = (minibatch.adv - mean) / (std + self._eps)
                    
                    ratio = (dist.log_prob(minibatch.act[:,i*self.nu:(i+1)*self.nu]) -
                            minibatch.logp_old[:, i]).exp().float()
                    ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                    surr1 = ratio * minibatch.adv
                    surr2 = ratio.clamp(
                        1.0 - self._eps_clip, 1.0 + self._eps_clip
                    ) * minibatch.adv
                    
                    if torch.isnan(ratio).any():
                        import pdb; pdb.set_trace()
                    if self._dual_clip:
                        clip1 = torch.min(surr1, surr2)
                        clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                        clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                    else:
                        clip_loss = -torch.min(surr1, surr2).mean()
                    
                    # calculate regularization and overall loss
                    ent_loss = dist.entropy().mean()
                    guidance_loss = 0
                    # Actor loss (no value loss since critic is updated separately)
                    loss = clip_loss - self._weight_ent * ent_loss + guidance_loss
                    
                    getattr(self, f"optim{i}").zero_grad()
                    loss.backward()
                    if self._grad_norm:
                        nn.utils.clip_grad_norm_(
                            getattr(self, f"_actor_critic{i}").parameters(), max_norm=self._grad_norm
                        )
                    getattr(self, f"optim{i}").step()
                    
                    clip_losses[i].append(clip_loss.item())
                    vf_losses[i].append(0.0)  # No individual value loss for actors
                    ent_losses[i].append(ent_loss.item())
                    losses[i].append(loss.item())
        
        self.step = self.step + 1
        for i in range(self.num_players):
            print_dict[f"loss{i}"] = losses[i]
            print_dict[f"loss/clip{i}"] = clip_losses[i]
            print_dict[f"loss/vf{i}"] = vf_losses[i]
            print_dict[f"loss/ent{i}"] = ent_losses[i]
            if self.pure_policy_regulation == False:
                print_dict[f"loss/bc_mean{i}"] = bc_mean_losses[i]
        
        # Add critic loss to print dict
        print_dict["critic_loss"] = critic_vf_loss.item()
        print_dict["bc_weight"] = behavior_loss_weight
        return print_dict
