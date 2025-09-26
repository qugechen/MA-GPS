
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


class IPPOPolicy(MARL_BasePolicy):
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
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
        critic_list,
        optim_list,
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
        super().__init__(#actor, critic, optim, 
                         dist_fn, **kwargs)
        self._eps_clip = eps_clip
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        for i in range(num_players):
            setattr(self, f"actor{i}", actor_list[i])
            setattr(self, f"optim{i}", optim_list[i])
            setattr(self, f"critic{i}", critic_list[i])
            setattr(self, f"_actor_critic{i}", ActorCritic(getattr(self, f"actor{i}"), getattr(self, f"critic{i}")))
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        self._norm_adv = advantage_normalization
        self._recompute_adv = recompute_advantage
        # self._actor_critic: ActorCritic
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
        self.ret_rms = [RunningMeanStd() for i in range(self.num_players)]
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
        bsz = len(batch.obs)
        v_s_tmp, v_s__tmp = [[] for i in range(self.num_players)], [[] for i in range(self.num_players)]
        v_s = np.zeros((bsz, self.num_players))
        v_s_ = np.zeros((bsz, self.num_players))
        batch.v_s = torch.zeros(bsz, self.num_players, device=self.device) 
        batch.v_s_ = torch.zeros(bsz, self.num_players, device=self.device)
        batch.returns = torch.zeros(bsz, self.num_players, device=self.device) 
        for i in range(self.num_players):
            with torch.no_grad():
                for minibatch in batch.split(self._batch, shuffle=False, merge_last=True):
                    model = getattr(self, f"critic{i}")
                    v_s_tmp[i].append(model(minibatch.obs))
                    v_s__tmp[i].append(model(minibatch.obs_next))
            # import pdb; pdb.set_trace()
            batch.v_s[:,i] = torch.cat(v_s_tmp[i], dim=0).flatten()  # old value
            v_s[:,i] = batch.v_s[:,i].cpu().numpy()
            v_s_[:,i] = torch.cat(v_s__tmp[i], dim=0).flatten().cpu().numpy()
            # when normalizing values, we do not minus self.ret_rms.mean to be numerically
            # consistent with OPENAI baselines' value normalization pipeline. Emperical
            # study also shows that "minus mean" will harm performances a tiny little bit
            # due to unknown reasons (on Mujoco envs, not confident, though).
            if self._rew_norm:  # unnormalize v_s & v_s_
                # import pdb; pdb.set_trace()
                v_s[:,i] = v_s[:,i] * np.sqrt(self.ret_rms[i].var + self._eps)
                v_s_[:,i] = v_s_[:,i] * np.sqrt(self.ret_rms[i].var + self._eps)
        
        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_,
            v_s,
            gamma=self._gamma,
            gae_lambda=self._lambda,
            num_players=self.num_players
        )
        if self._rew_norm:
            for i in range(self.num_players):
                # import pdb; pdb.set_trace()
                batch.returns[:,i] = torch.tensor(unnormalized_returns[:,i] / np.sqrt(self.ret_rms[i].var + self._eps), device = self.device)
                self.ret_rms[i].update(unnormalized_returns[:,i])
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
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
            batch.logp_old = torch.zeros(batch.act.shape[0], self.num_players, device=self.device)
            # batch.logp_old = self(batch).dist.log_prob(batch.act) # NOTE!!!!!!
            tmp_dist = self(batch).dist
            for i in range(self.num_players):
                # import pdb; pdb.set_trace()
                dist = self.dist_fn(tmp_dist.mean[:, i*self.nu:(i+1)*self.nu], tmp_dist.stddev[:, i*self.nu:(i+1)*self.nu])
                batch.logp_old[:,i] = dist.log_prob(batch.act[:,i*self.nu:(i+1)*self.nu])
                # check if logp_old has nan
                if torch.isnan(batch.logp_old[:,i]).any():
                    import pdb; pdb.set
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
            # act = logits[0]
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
                np.random.shuffle(self.player_list)
                if self.pure_policy_regulation == False and self.no_gd_regularization == False:
                    # import pdb; pdb.set_trace()
                    with torch.no_grad():
                        expert_action = self.expert.evaluate_expert_action(torch.tensor(minibatch.obs,requires_grad=False), self.wrap_policy).detach()
                        expert_action = torch.clamp(expert_action, min=-1.0, max=1.0)
                        # expert_action = expert_action = torch.zeros(minibatch.act.shape)
                
                for j in range(self.num_players):
                    i = self.player_list[j]
                    tmp = self(minibatch)
                    tmp_dist = tmp.dist # NOTE!!!!!!
                    tmp_action_mean = tmp.logits[0]
                    tmp_action_std = tmp.logits[1]
                    dist = self.dist_fn(tmp_dist.mean[:,i*self.nu:(i+1)*self.nu], tmp_dist.stddev[:,i*self.nu:(i+1)*self.nu])
                    if self._norm_adv:
                        mean, std = minibatch.adv[:,i].mean(), minibatch.adv[:,i].std()
                        minibatch.adv[:,i] = (minibatch.adv[:,i] -
                                        mean) / (std + self._eps)  # per-batch norm    
                    ratio = (dist.log_prob(minibatch.act[:,i*self.nu:(i+1)*self.nu]) -
                            minibatch.logp_old[:, i]).exp().float()
                    ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                    surr1 = ratio * minibatch.adv[:,i]
                    surr2 = ratio.clamp(
                        1.0 - self._eps_clip, 1.0 + self._eps_clip
                    ) * minibatch.adv[:,i]
                    # check if ratio has nan
                    if torch.isnan(ratio).any():
                        import pdb; pdb.set_trace()
                    if self._dual_clip:
                        clip1 = torch.min(surr1, surr2)
                        clip2 = torch.max(clip1, self._dual_clip * minibatch.adv[:,i])
                        clip_loss = -torch.where(minibatch.adv[:,i] < 0, clip2, clip1).mean()
                    else:
                        clip_loss = -torch.min(surr1, surr2).mean()
                    # check if clip_loss = None
                    # if clip_loss == None:
                    #     clip_loss = 0
                    # calculate loss for critic
                    value = getattr(self, f"critic{i}")(minibatch.obs).flatten()
                    if self._value_clip:
                        v_clip = minibatch.v_s[:,i] + \
                            (value - minibatch.v_s[:,i]).clamp(-self._eps_clip, self._eps_clip)
                        vf1 = (minibatch.returns[:,i] - value).pow(2)
                        vf2 = (minibatch.returns[:,i] - v_clip).pow(2)
                        vf_loss = torch.max(vf1, vf2).mean()
                    else:
                        vf_loss = (minibatch.returns[:,i] - value).pow(2).mean()
                    # calculate regularization and overall loss
                    ent_loss = dist.entropy().mean()
                    # if ent_loss == None:
                    #     ent_loss=0
                    guidance_loss = 0
                    if self.pure_policy_regulation == False and self.no_gd_regularization == False:
                        tmp_size = expert_action.shape[0]
                        action_error_mean = expert_action[:, i*self.nu:(i+1)*self.nu] - tmp_action_mean[:tmp_size, i*self.nu:(i+1)*self.nu]
                        # action_error_var = 0.1 - tmp_action_var[:tmp_size, i*self.nu:(i+1)*self.nu]
                        guidance_loss_mean = torch.mean(torch.sum(action_error_mean**2, dim=1))
                        # guidance_loss_var = torch.mean(torch.sum(action_error_var**2, dim=1))
                        guidance_loss = (guidance_loss_mean) * behavior_loss_weight
                        
                    elif self.no_gd_regularization == True:
                        tmp_size = minibatch.act.shape[0]
                        action_error_mean = tmp_action_mean[:tmp_size, i*self.nu:(i+1)*self.nu]

                        guidance_loss_mean = torch.mean(torch.sum(action_error_mean**2, dim=1))

                        guidance_loss = (guidance_loss_mean) * behavior_loss_weight
                        # print("no_expert")
                    loss = clip_loss + self._weight_vf * vf_loss \
                         - self._weight_ent * ent_loss + guidance_loss
                    # if loss > 10000:
                    #     import pdb; pdb.set_trace()
                    getattr(self, f"optim{i}").zero_grad()
                    loss.backward()
                    if self._grad_norm:  # clip large gradient
                        nn.utils.clip_grad_norm_(
                            getattr(self, f"_actor_critic{i}").parameters(), max_norm=self._grad_norm
                            # self._actor_critic.parameters(), max_norm=self._grad_norm
                        )
                    getattr(self, f"optim{i}").step()
                    clip_losses[i].append(clip_loss.item())
                    vf_losses[i].append(vf_loss.item())
                    ent_losses[i].append(ent_loss.item())
                    losses[i].append(loss.item())
                    if self.pure_policy_regulation == False:
                        bc_mean_losses[i].append(guidance_loss_mean.item()) 
                    # bc_var_losses[i].append(guidance_loss_var.item())
        self.step = self.step + 1
        for i in range(self.num_players):
            print_dict[f"loss{i}"] = losses[i]
            print_dict[f"loss/clip{i}"] = clip_losses[i]
            print_dict[f"loss/vf{i}"] = vf_losses[i]
            print_dict[f"loss/ent{i}"] = ent_losses[i]
            if self.pure_policy_regulation == False:
                print_dict[f"loss/bc_mean{i}"] = bc_mean_losses[i]
            # print_dict[f"loss/bc_var{i}"] = bc_var_losses[i]
        print_dict["bc_weight"] = behavior_loss_weight
        return print_dict
