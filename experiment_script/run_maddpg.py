import argparse
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from MAGPS.data import Collector, VectorReplayBuffer
from MAGPS.env import DummyVectorEnv
from MAGPS.exploration import GaussianNoise
from MAGPS.policy.gym_marl_policy.maddpg import MADDPGPolicy
from MAGPS.policy.MARL_base import MARL_BasePolicy
from MAGPS.trainer import OffpolicyTrainer
from MAGPS.utils import TensorboardLogger
from MAGPS.utils.net.common import Net
from MAGPS.utils.net.continuous import Actor, Critic
# from tianshou.utils.space_info import SpaceInfo


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Three_Unicycle_Game-v0")
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise", type=float, default=0.5)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--total-episodes", type=int, default=1)
    parser.add_argument("--step-per-epoch", type=int, default=20000)
    parser.add_argument("--step-per-collect", type=int, default=8)
    parser.add_argument("--update-per-step", type=float, default=0.125)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--critic-net", type=int, nargs="*", default=[512, 512, 512, 512,512])
    parser.add_argument("--actor-net", type=int, nargs="*", default=[512, 512, 512])
    parser.add_argument("--training-num", type=int, default=64)
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--rew-norm", action="store_true", default=False)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument('--continue-training-logdir', type=str, default=None)
    parser.add_argument('--continue-training-epoch', type=int, default=None)
    parser.add_argument('--behavior-loss-weight', type=float, default=100.0)
    parser.add_argument('--behavior-loss-weight-decay', type=float, default=0.5)
    parser.add_argument('--regularization', type=bool, default=True) # if true, then expert_policy = 0
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument('--kwargs', type=str, default='{}')
    return parser.parse_known_args()[0]

args=get_args()
env = gym.make(args.task)
args.max_action = env.action_space.high[0]
args.state_shape = env.observation_space.shape or env.observation_space.n
args.action_shape = env.action_space.shape or env.action_space.n
# you can also use tianshou.env.SubprocVectorEnv
# train_envs = gym.make(args.task)
train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
# test_envs = gym.make(args.task)
test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_envs.seed(args.seed)
test_envs.seed(args.seed)
# model
args.action_shape_per_player = env.nu

actor_net = Net(args.state_shape, hidden_sizes=args.actor_net, device=args.device)
actor_list = [
    Actor(
        actor_net, args.action_shape_per_player, max_action=args.max_action, device=args.device
    ).to(args.device) for i in range(env.num_players)
]
actor_optim_list = [
    torch.optim.Adam(actor_list[i].parameters(), lr=args.actor_lr) for i in range(env.num_players)
]
critic_net = Net(
    args.state_shape,
    args.action_shape,
    hidden_sizes=args.critic_net,
    concat=True,
    device=args.device
)
critic_list = [
    Critic(critic_net, device=args.device).to(args.device) for i in range(env.num_players)
]
critic_optim_list = [
    torch.optim.Adam(critic_list[i].parameters(), lr=args.critic_lr) for i in range(env.num_players)
]
policy: MADDPGPolicy = MADDPGPolicy(
    actor_list=actor_list,
    actor_optim_list=actor_optim_list,
    critic_list=critic_list,
    critic_optim_list=critic_optim_list,
    num_players=env.num_players,
    tau=args.tau,
    gamma=args.gamma,
    exploration_noise=GaussianNoise(sigma=args.exploration_noise),
    estimation_step=args.n_step,
    action_space=env.action_space,
    action_dim=env.action_space.shape[0],
    device=args.device,
    env=env,
    pure_policy_regularization = args.regularization,
    batch_size=args.batch_size,
)
# collector
train_collector = Collector(
    policy,
    train_envs,
    VectorReplayBuffer(args.buffer_size, len(train_envs)),
    exploration_noise=True,
)
test_collector = Collector(policy, test_envs)
# log
print(args.regularization)
# import pdb; pdb.set_trace()
log_path = os.path.join(args.logdir, args.task, 
                        'maddpg_tau_{}_training_num_{}_buffer_size_{}_c_{}_{}_a_{}_{}_gamma_{}_behavior_loss_{}_{}_L2_reg_{}'.format(
    args.tau,
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
log_path = log_path+'/noise_{}_actor_lr_{}_critic_lr_{}_batch_{}_step_per_epoch_{}_kwargs_{}_seed_{}'.format(
    args.exploration_noise, 
    args.actor_lr, 
    args.critic_lr, 
    args.batch_size,
    args.step_per_epoch,
    args.kwargs,
    args.seed
)

# writer = SummaryWriter(log_path)
# logger = TensorboardLogger(writer)
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
def stop_fn(mean_rewards: float) -> bool:
    return False



for iter in range(args.total_episodes):
    print('epoch: ', epoch)
    print("log_path: ", log_path+"/epoch_id_{}".format(epoch+args.epoch))
    if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
        print("Just created the log directory!")
        os.makedirs(log_path+"/epoch_id_{}".format(epoch))
    if not os.path.exists(log_path+"/epoch_id_{}".format(epoch+args.epoch)):
        print("Just created the log directory!")
        os.makedirs(log_path+"/epoch_id_{}".format(epoch+args.epoch))
    writer = SummaryWriter(log_path+"/epoch_id_{}".format(epoch+args.epoch)) #filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))

    
    logger = TensorboardLogger(writer)
    result = OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=args.epoch,
    step_per_epoch=args.step_per_epoch,
    step_per_collect=args.step_per_collect,
    episode_per_test=args.test_num,
    batch_size=args.batch_size,
    update_per_step=args.update_per_step,
    stop_fn=stop_fn,
    save_best_fn=save_best_fn,
    logger=logger,
    behavior_loss_weight = args.behavior_loss_weight * args.behavior_loss_weight_decay ** iter
    ).run()
    save_best_fn(policy, epoch=epoch+args.epoch)
        
    epoch = epoch + args.epoch
