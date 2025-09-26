from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

class Three_Unicycle_Game_Env0(gym.Env):
   
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self.dt = 0.1
        self.num_players = 3
        self.players_u_index_list = torch.tensor([[0,1], [2,3], [4,5]])
        self.nx = 4 # each player has 4 states
        self.nu = 2 # each player has 2 control inputs
        self.total_state_dim = self.nx * self.num_players
        self.total_action_dim = self.nu * self.num_players
        
        self.action_low = np.array([
            -1, -1, # player 1 control input 
            -1, -1, # player 2 control input
            -1, -1, # player 3 control input
        ])
        self.action_high = np.array([
            1, 1, # player 1 control input
            1, 1, # player 2 control input
            1, 1, # player 3 control input

        ])
        self.state_low = np.array([
            -4, 0, 0, 1/5*np.pi, # x, y, v, theta of player 1
            -4, 0, 0, 1/5*np.pi, # x, y, v, theta of player 2
            -4, 0, 0, 1/5*np.pi, # x, y, v, theta of player 3
            
        ])
        self.state_high = np.array([
            4, 20, 2, 4/5*np.pi, # x, y, v, theta of player 1
            4, 20, 2, 4/5*np.pi, # x, y, v, theta of player 2
            4, 20, 2, 4/5*np.pi, # x, y, v, theta of player 3

        ])
        # defining state and action spaces:        
        self.action_space = spaces.Box(
            low=self.action_low, high=self.action_high, dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=self.state_low, high=self.state_high, dtype=np.float64
        )
        basis = np.eye(self.total_state_dim)
        basis_u = np.eye(self.total_action_dim)

        self.unicycle_dynamics = lambda x, u: np.array([
            x[0] + self.dt * x[2] * np.cos(x[3]), # x
            x[1] + self.dt * x[2] * np.sin(x[3]), # y
            x[2] + self.dt * 2*u[0], # v
            x[3] + self.dt * u[1], # theta
        ])
        
        # We start with defining the individual dynamics of each player!
        self.f = lambda x, u: np.concatenate((
            self.unicycle_dynamics(x[:4], u[:2]), # player 1
            self.unicycle_dynamics(x[4:8], u[2:4]), # player 2
            self.unicycle_dynamics(x[8:12], u[4:]), # player 3

            )
        ).reshape(-1)

        self.cost_functions = [
            lambda x, u: -(5*(basis[4,:]@x - 0.5)**2 +5*(basis[8,:]@x - 0.5)**2 + (basis[2,:]@x - 1)**2 + (basis[3,:]@x - 1/2 * np.pi)**2 + (basis_u[0,:]@u**2 + basis_u[1,:]@u**2)),

            lambda x, u: -(5*(basis[4,:]@x - basis[0,:]@x)**2 + (basis[6,:]@x - 1)**2 + (basis[7,:]@x - 1/2 * np.pi)**2 + (basis_u[2,:]@u**2 + basis_u[3,:]@u**2)),

            lambda x, u: -(5*(basis[8,:]@x - basis[0,:]@x)**2 + (basis[10,:]@x - 1)**2 + (basis[11,:]@x - 1/2 * np.pi)**2 + (basis_u[4,:]@u**2 + basis_u[5,:]@u**2)),
        ]
        
        
        self.is_nonlinear_game = True
    
    def step(self, u):
        # assert self.action_space.contains(u), f"action {u} is out of action space {self.action_space}"
        # we update state and cost, and return them as the first and second output of step() function
        # update state for each player
        # update cost for each player
        # feasibility_metric_scale = 100
        # state_distance_to_lower_bound = self.state[[0,2,3,4,6,7]] - self.state_low[[0,2,3,4,6,7]]
        # state_distance_to_upper_bound = self.state_high[[0,2,3,4,6,7]] - self.state[[0,2,3,4,6,7]]
        # feasibility_metric = np.min((state_distance_to_lower_bound, state_distance_to_upper_bound))
        # min_feasibility_metric = np.min(feasibility_metric) * feasibility_metric_scale
        
        self.costs  = np.array([
            (cost(self.state, u) + 30)/20 for cost in self.cost_functions
            ])
        sum_costs = np.sum(self.costs)
        self.state = self.f(self.state, u)      

        terminated = False # check whether state is out of bound
        truncated = False # we don't use here
        if np.any(self.state[[0,4,8]] > self.state_high[[0,4,8]]) or np.any(self.state[[0,4,8]] < self.state_low[[0,4,8]]):
            # if min_feasibility_metric < 0:
            terminated = True
        info = {"individual_cost": self.costs }#np.array([sum_costs, sum_costs])}
        # note that we set reward, as shown in the second output, 
        # to the sum of players' costs, but we store individual costs in info!
        return self.state, sum_costs, terminated, truncated, info


    # NOTE! we can assign initial state as env.reset(options = {"initial_state": initial_state}) !!!!!
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        initial_state_low = np.array([
            -1, 0, 0.1, 1/4*np.pi, 
            -1, 0, 0.1, 1/4*np.pi, 
            -1, 0, 0.1, 1/4*np.pi, 

        ])
        initial_state_high = np.array([
            3, 20, 1.9, 3/4*np.pi, 
            3, 20, 1.9, 3/4*np.pi, 
            3, 20, 1.9, 3/4*np.pi, 
        ])
        if options is None:
            self.state = self.np_random.uniform(low=initial_state_low, high=initial_state_high)
        elif "initial_state" in options:
            self.state = options["initial_state"]
        self.costs = np.zeros(self.num_players)
        return self.state, {}

    def render(self):
        pass

    @torch.jit.script
    def costs_jacobian_and_hessian(z):
        """
        Manually compute the Jacobian and Hessian matrices for the two cost functions over a batch of inputs.

        Parameters:
        - z: A tensor of shape (batch_size, input_dim).

        Returns:
        - jacobians: A tensor of shape (2, batch_size, input_dim) containing the Jacobian for each cost function and each input in the batch.
        - hessians: A tensor of shape (2, batch_size, input_dim, input_dim) containing the Hessian for each cost function and each input in the batch.
        """
        batch_size, input_dim = z.shape

        # Initialize tensors to store the Jacobians and Hessians
        jacobians = torch.zeros(3, batch_size, input_dim, device=z.device)
        hessians = torch.zeros(3, batch_size, input_dim, input_dim, device=z.device)

        # Manually compute the Jacobian and Hessian for the first cost function
        jacobians[0, :, 4] = 2 * (z[:, 4] - 0.5) * 5
        jacobians[0, :, 8] = 2 * (z[:, 8] - 0.5) * 5
        jacobians[0, :, 2] = 2 * (z[:, 2] - 1)
        jacobians[0, :, 3] = 2 * (z[:, 3] - 0.5 * torch.pi)
        jacobians[0, :, 12] = 2 * z[:, 12] 
        jacobians[0, :, 13] = 2 * z[:, 13] 

        hessians[0, :, 4, 4] = 2 * 5
        hessians[0, :, 8, 8] = 2 * 5

        hessians[0, :, 2, 2] = 2
        hessians[0, :, 3, 3] = 2
        hessians[0, :, 12, 12] = 2 
        hessians[0, :, 13, 13] = 2 

        # Manually compute the Jacobian and Hessian for the second cost function
        jacobians[1, :, 0] = 2 * (z[:, 0] - z[:,4]) * 5
        jacobians[1, :, 4] = 2 * (z[:, 0] - z[:,4]) * -5

        jacobians[1, :, 6] = 2 * (z[:, 6] - 1)
        jacobians[1, :, 7] = 2 * (z[:, 7] - 0.5 * torch.pi)
        # jacobians[1, :, 14] = 2 * z[:, 10] 
        # jacobians[1, :, 15] = 2 * z[:, 11]
        jacobians[1, :, 14] = 2 * z[:, 14] 
        jacobians[1, :, 15] = 2 * z[:, 15]

        hessians[1, :, 0, 0] = 2 * 5
        hessians[1, :, 0, 4] = -2 * 5


        hessians[1, :, 4, 0] = -2 * 5
        hessians[1, :, 4, 4] = 2 * 5

        hessians[1, :, 6, 6] = 2
        hessians[1, :, 7, 7] = 2
        hessians[1, :, 14, 14] = 2 
        hessians[1, :, 15, 15] = 2 

        # Manually compute the Jacobian and Hessian for the third cost function
        jacobians[2, :, 0] = 2 * (z[:, 0] - z[:,8]) * 5
        jacobians[2, :, 8] = 2 * (z[:, 0] - z[:,8]) * -5

        jacobians[2, :, 10] = 2 * (z[:, 10] - 1)
        jacobians[2, :, 11] = 2 * (z[:, 11] - 0.5 * torch.pi)
        jacobians[2, :, 16] = 2 * z[:, 16] 
        jacobians[2, :, 17] = 2 * z[:, 17] 

        hessians[2, :, 0, 0] = 2 * 5
        hessians[2, :, 0, 8] = -2 * 5

        hessians[2, :, 8, 0] = -2 * 5
        hessians[2, :, 8, 8] = 2 * 5

        hessians[2, :, 10, 10] = 2
        hessians[2, :, 11, 11] = 2
        hessians[2, :, 16, 16] = 2 
        hessians[2, :, 17, 17] = 2 

        return jacobians, hessians
    
    @torch.jit.script
    def dynamics_jacobian(states, controls):
        """
        Computes the Jacobian matrix of the combined unicycle dynamics with respect to the state and control inputs for a batch of data.

        Parameters:
        - states: A tensor of shape (batch_size, 8), where each row is [x1, y1, v1, w1, x2, y2, v2, w2].
        - controls: A tensor of shape (batch_size, 4), where each row is [a1, u1, a2, u2].

        Returns:
        - jacobian: A tensor of shape (batch_size, 8, 12) containing the Jacobian matrix for each input in the batch.
        """
        batch_size = states.shape[0]

        # Extract state variables for both unicycles

        v1 = states[:, 2]
        w1 = states[:, 3]

        v2 = states[:, 6]
        w2 = states[:, 7]

        v3 = states[:, 10]
        w3 = states[:, 11]
        

        # Initialize the Jacobian matrix (batch_size, 8, 12)
        jacobian = torch.zeros(batch_size, 12, 18, device=states.device)

        # Compute the partial derivatives for the first unicycle
        jacobian[:, 0, 0] = 1.0  # ∂x1[t+1]/∂x1[t] = 1
        jacobian[:, 0, 2] = 0.1 * torch.cos(w1)  # ∂x1[t+1]/∂v1[t] = 0.1 * cos(w1[t])
        jacobian[:, 0, 3] = -0.1 * v1 * torch.sin(w1)  # ∂x1[t+1]/∂w1[t] = -0.1 * v1[t] * sin(w1[t])

        jacobian[:, 1, 1] = 1.0  # ∂y1[t+1]/∂y1[t] = 1
        jacobian[:, 1, 2] = 0.1 * torch.sin(w1)  # ∂y1[t+1]/∂v1[t] = 0.1 * sin(w1[t])
        jacobian[:, 1, 3] = 0.1 * v1 * torch.cos(w1)  # ∂y1[t+1]/∂w1[t] = 0.1 * v1[t] * cos(w1[t])

        jacobian[:, 2, 2] = 1.0  # ∂v1[t+1]/∂v1[t] = 1
        jacobian[:, 2, 12] = 0.2  # ∂v1[t+1]/∂a1[t] = 0.1

        jacobian[:, 3, 3] = 1.0  # ∂w1[t+1]/∂w1[t] = 1
        jacobian[:, 3, 13] = 0.1  # ∂w1[t+1]/∂u1[t] = 0.1

        # Compute the partial derivatives for the second unicycle
        jacobian[:, 4, 4] = 1.0  # ∂x2[t+1]/∂x2[t] = 1
        jacobian[:, 4, 6] = 0.1 * torch.cos(w2)  # ∂x2[t+1]/∂v2[t] = 0.1 * cos(w2[t])
        jacobian[:, 4, 7] = -0.1 * v2 * torch.sin(w2)  # ∂x2[t+1]/∂w2[t] = -0.1 * v2[t] * sin(w2[t])

        jacobian[:, 5, 5] = 1.0  # ∂y2[t+1]/∂y2[t] = 1
        jacobian[:, 5, 6] = 0.1 * torch.sin(w2)  # ∂y2[t+1]/∂v2[t] = 0.1 * sin(w2[t])
        jacobian[:, 5, 7] = 0.1 * v2 * torch.cos(w2)  # ∂y2[t+1]/∂w2[t] = 0.1 * v2[t] * cos(w2[t])

        jacobian[:, 6, 6] = 1.0  # ∂v2[t+1]/∂v2[t] = 1
        jacobian[:, 6, 14] = 0.2  # ∂v2[t+1]/∂a2[t] = 0.1

        jacobian[:, 7, 7] = 1.0  # ∂w2[t+1]/∂w2[t] = 1
        jacobian[:, 7, 15] = 0.1  # ∂w2[t+1]/∂u2[t] = 0.1

        jacobian[:, 8, 8] = 1.0  # ∂x2[t+1]/∂x2[t] = 1
        jacobian[:, 8, 10] = 0.1 * torch.cos(w3)  # ∂x2[t+1]/∂v2[t] = 0.1 * cos(w2[t])
        jacobian[:, 8, 11] = -0.1 * v3 * torch.sin(w3)  # ∂x2[t+1]/∂w2[t] = -0.1 * v2[t] * sin(w2[t])

        jacobian[:, 9, 9] = 1.0  # ∂y2[t+1]/∂y2[t] = 1
        jacobian[:, 9, 10] = 0.1 * torch.sin(w3)  # ∂y2[t+1]/∂v2[t] = 0.1 * sin(w2[t])
        jacobian[:, 9, 11] = 0.1 * v3 * torch.cos(w3)  # ∂y2[t+1]/∂w2[t] = 0.1 * v2[t] * cos(w2[t])

        jacobian[:, 10, 10] = 1.0  # ∂v2[t+1]/∂v2[t] = 1
        jacobian[:, 10, 16] = 0.2  # ∂v2[t+1]/∂a2[t] = 0.1

        jacobian[:, 11, 11] = 1.0  # ∂w2[t+1]/∂w2[t] = 1
        jacobian[:, 11, 17] = 0.1  # ∂w2[t+1]/∂u2[t] = 0.1

        return jacobian
    
    @torch.jit.script
    def dynamics(states, controls):
        next_states = torch.zeros_like(states)
        next_states[:,0] = states[:,0] + 0.1 * states[:,2] * torch.cos(states[:,3])
        next_states[:,1] = states[:,1] + 0.1 * states[:,2] * torch.sin(states[:,3])
        next_states[:,2] = states[:,2] + 0.1 * controls[:,0]*2
        next_states[:,3] = states[:,3] + 0.1 * controls[:,1]

        next_states[:,4] = states[:,4] + 0.1 * states[:,6] * torch.cos(states[:,7])
        next_states[:,5] = states[:,5] + 0.1 * states[:,6] * torch.sin(states[:,7])
        next_states[:,6] = states[:,6] + 0.1 * controls[:,2]*2
        next_states[:,7] = states[:,7] + 0.1 * controls[:,3]

        next_states[:,8] = states[:,8] + 0.1 * states[:,10] * torch.cos(states[:,11])
        next_states[:,9] = states[:,9] + 0.1 * states[:,10] * torch.sin(states[:,11])
        next_states[:,10] = states[:,10] + 0.1 * controls[:,4]*2
        next_states[:,11] = states[:,11] + 0.1 * controls[:,5]
        
        return next_states