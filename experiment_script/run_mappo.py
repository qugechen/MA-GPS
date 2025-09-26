import argparse
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal

from MAGPS.data import Collector, VectorReplayBuffer
from MAGPS.env import DummyVectorEnv
from MAGPS.trainer import OnpolicyTrainer
from MAGPS.utils import TensorboardLogger
from MAGPS.policy.gym_marl_policy.mappo import MAPPOPolicy
from MAGPS.utils.net.common import ActorCritic, Net
from MAGPS.utils.net.continuous import ActorProb, Critic

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Three_Unicycle_Game-v0")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--total-episodes", type=int, default=30)
    parser.add_argument("--step-per-epoch", type=int, default=40000)
    parser.add_argument("--episode-per-collect", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--critic-net", type=int, nargs="*", default=[512]*3)
    parser.add_argument("--actor-net", type=int, nargs="*", default=[512]*3)
    parser.add_argument("--training-num", type=int, default=64)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument('--continue-training-logdir', type=str, default=None)
    parser.add_argument('--continue-training-epoch', type=int, default=None)
    parser.add_argument('--behavior-loss-weight', type=float, default=0.1)
    parser.add_argument('--behavior-loss-weight-decay', type=float, default=1)
    parser.add_argument('--no-gd-regularization', type=bool, default=False) # if true, then expert_policy = 0
    parser.add_argument('--regularization', type=bool, default= True) # if true, then rho = 0
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    # ppo special
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.005)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.99)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--save-interval", type=int, default=4)
    parser.add_argument('--kwargs', type=str, default='{}')
    return parser.parse_known_args()[0]

args=get_args()
env = gym.make(args.task)
args.max_action = env.action_space.high[0]
args.state_shape = env.observation_space.shape or env.observation_space.n
args.action_shape = env.action_space.shape or env.action_space.n

train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])

test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_envs.seed(args.seed)
test_envs.seed(args.seed)
# model
args.action_shape_per_player = env.nu

# Create actor networks (same as original)
actor_net = Net(args.state_shape, hidden_sizes=args.actor_net, device=args.device)
actor_list = [
    ActorProb(
        actor_net, args.action_shape_per_player, max_action=args.max_action, device=args.device
    ).to(args.device) for i in range(env.num_players)
]

# Create single critic network (different from original)
critic_net = Net(
    args.state_shape,
    hidden_sizes=args.critic_net,
    device=args.device
)
critic = Critic(critic_net, device=args.device).to(args.device)

# Create actor-critic pairs for orthogonal initialization
actor_critic_list = [ActorCritic(actor_list[i], critic) for i in range(env.num_players)]

# orthogonal initialization
for i in range(env.num_players):
    for m in actor_critic_list[i].modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)

# Create optimizers
optim_list = [
    torch.optim.Adam(actor_list[i].parameters(), lr=args.lr) for i in range(env.num_players)
]
critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr)

# replace DiagGuassian with Independent(Normal) which is equivalent
# pass *logits to be consistent with policy.forward
def dist(*logits):
    return Independent(Normal(*logits), 1)

# Create single critic MAPPO policy
policy = MAPPOPolicy(
    actor_list=actor_list,
    critic=critic,
    optim_list=optim_list,
    critic_optim=critic_optim,
    num_players=env.num_players,
    nu=env.nu,
    dist_fn=dist,
    discount_factor=args.gamma,
    max_grad_norm=args.max_grad_norm,
    eps_clip=args.eps_clip,
    vf_coef=args.vf_coef,
    ent_coef=args.ent_coef,
    reward_normalization=args.rew_norm,
    advantage_normalization=args.norm_adv,
    recompute_advantage=args.recompute_adv,
    dual_clip=args.dual_clip,
    value_clip=args.value_clip,
    gae_lambda=args.gae_lambda,
    action_space=env.action_space,
    device=args.device,
    pure_policy_regulation=args.regularization,
    no_gd_regularization=args.no_gd_regularization,
    env=env,
    batch_size=args.batch_size,
)

# collector
train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
    )
test_collector = Collector(policy, test_envs)

# log
print(f"Using Single Critic MAPPO with regularization: {args.regularization}")

log_path = os.path.join(args.logdir, args.task, 
                        'mappo_single_critic_training_num_{}_buffer_size_{}_c_{}_{}_a_{}_{}_gamma_{}_behavior_loss_{}_{}_L2_reg_{}'.format(
    args.training_num, 
    args.buffer_size,
    args.critic_net[0],
    len(args.critic_net),
    args.actor_net[0],
    len(args.actor_net),
    args.gamma,
    args.behavior_loss_weight,
    args.behavior_loss_weight_decay,
    args.regularization
    )
)
log_path = log_path+'/lr_{}_batch_{}_step_per_epoch_{}_kwargs_{}_seed_{}'.format(
    args.lr, 
    args.batch_size,
    args.step_per_epoch,
    args.kwargs,
    args.seed
)

epoch = 0
if args.continue_training_logdir is not None:
    policy.load_state_dict(torch.load(args.continue_training_logdir))
    epoch = args.continue_training_epoch
    print("Loaded the model from the last checkpoint!")
else:
    epoch=0

def save_best_fn(policy, epoch=epoch):
    torch.save(
        policy.state_dict(), 
        os.path.join(
            log_path+"/epoch_id_{}".format(epoch),
            "policy.pth"
        )
    )

def save_checkpoint_fn(epoch, env_step, gradient_step):
    
    ckpt_path = os.path.join(log_path+"/epoch_id_{}".format(epoch), "checkpoint.pth")

    if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
        os.makedirs(log_path+"/epoch_id_{}".format(epoch))
    torch.save(
        {
            "model": policy.state_dict(),
            "optim": optim_list[0].state_dict(),
            "critic_optim": critic_optim.state_dict(),
        }, ckpt_path
    )
    return ckpt_path

def stop_fn(mean_rewards: float) -> bool:
    return False

for iter in range(args.total_episodes):
    print('epoch: ', epoch)
    if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
        print("Just created the log directory!")
        os.makedirs(log_path+"/epoch_id_{}".format(epoch))
    print("log_path: ", log_path+"/epoch_id_{}".format(epoch+args.epoch))
    if not os.path.exists(log_path+"/epoch_id_{}".format(epoch+args.epoch)):
        print("Just created the log directory!")
        os.makedirs(log_path+"/epoch_id_{}".format(epoch+args.epoch))
    writer = SummaryWriter(log_path+"/epoch_id_{}".format(epoch+args.epoch)) 

    
    logger = TensorboardLogger(writer)
    result = OnpolicyTrainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        resume_from_log=args.resume,
        save_checkpoint_fn=save_checkpoint_fn,
        behavior_loss_weight = args.behavior_loss_weight * args.behavior_loss_weight_decay ** iter,
    ).run()
        
    epoch = epoch + args.epoch
    save_best_fn(policy, epoch=epoch)
