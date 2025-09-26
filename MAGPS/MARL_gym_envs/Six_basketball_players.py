from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch




class basketball_Env0(gym.Env):

    """
    Clean implementation of linear quadratic game
    
    1. Stage cost of player i:
        c_t^i(x_t, u_t_1, u_t_2)
    where x_t_i and u_t_i are the state and the control input of player i at time t.
    
    2. Dynamics:
        x_{t+1} = f(x_t, u_t_1, u_t_2)
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self.dt = 0.1
        self.num_players = 6
        self.players_u_index_list = torch.tensor([[0,1], [2,3],[4,5], [6,7],[8,9],[10,11]])
        self.nx = 4 
        self.nu = 2 
        self.total_state_dim = self.nx * self.num_players
        self.total_action_dim = self.nu * self.num_players
        
        self.action_low = np.array([
            -1, -1, 
            -1, -1, 
            -1, -1, 
            -1, -1, 
            -1, -1, 
            -1, -1, 

        ])
        self.action_high = np.array([
            1, 1, 
            1, 1, 
            1, 1, 
            1, 1, 
            1, 1, 
            1, 1, 


        ])
        self.state_low = np.array([
            -10, -14, -1, -1, 
            -10, -14, -1, -1, 
            -10, -14, -1, -1, 
            -10, -14, -1, -1, 
            -10, -14, -1, -1, 
            -10, -14, -1, -1, 
        ])
        self.state_high = np.array([
            10, 10, 1, 1, 
            10, 10, 1, 1, 
            10, 10, 1, 1, 
            10, 10, 1, 1, 
            10, 10, 1, 1, 
            10, 10, 1, 1, 

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

        self.basketball_dynamics = lambda x, u: np.array([
            x[0] + self.dt * x[2], # x
            x[1] + self.dt * x[3], # y
            x[2] + self.dt * 5 * u[0], # dx
            x[3] + self.dt * 5 * u[1], # dy
        ])
        
        # We start with defining the individual dynamics of each player!
        self.f = lambda x, u: np.concatenate((
            self.basketball_dynamics(x[:4], u[:2]), 
            self.basketball_dynamics(x[4:8], u[2:4]), 
            self.basketball_dynamics(x[8:12], u[4:6]), 
            self.basketball_dynamics(x[12:16], u[6:8]), 
            self.basketball_dynamics(x[16:20], u[8:10]), 
            self.basketball_dynamics(x[20:24], u[10:12]), 
            )
        ).reshape(-1)
        # There are two ways to define costs, 
        # one is to define the cost function directly in step() function, 
        # and the other is to define the following cost functions! 
        
        self.cost_functions = [
            lambda x, u: -((((basis[0,:]@x)**2 +(basis[1,:]@x)**2) - 1.5**2)**2 + ((basis[2,:]@x)**2+(basis[3,:]@x)**2) +((basis_u[0,:]@u)**2 + (basis_u[1,:]@u)**2) + 100*max(basis[1,:]@x, 0) ), 
            lambda x, u: -(((basis[4,:]@x-0.75*basis[0,:]@x)**2 + (basis[5,:]@x - 0.75*basis[1,:]@x)**2) + ((basis_u[2,:]@u)**2 + (basis_u[3,:]@u**2))),

            lambda x, u: -((((basis[8,:]@x)**2 +(basis[9,:]@x)**2) - 16)**2 + ((basis[10,:]@x)**2+(basis[11,:]@x)**2)+((basis_u[4,:]@u)**2 + (basis_u[5,:]@u**2)) + 100*max(basis[9,:]@x, 0)), 
            lambda x, u: -(((basis[12,:]@x-0.75*basis[8,:]@x)**2 + (basis[13,:]@x - 0.75*basis[9,:]@x)**2) +((basis_u[6,:]@u)**2 + (basis_u[7,:]@u)**2)),

            lambda x, u: -(((basis[16,:]@x-basis[12,:]@x)**2 +(basis[17,:]@x - basis[13,:]@x)**2) + ((basis_u[8,:]@u)**2 + (basis_u[9,:]@u**2))), 
            lambda x, u: -(((basis[20,:]@x-0.75*basis[16,:]@x)**2 + (basis[21,:]@x - 0.75*basis[17,:]@x)**2) + ((basis_u[10,:]@u)**2 + (basis_u[11,:]@u)**2)),

        ]
        

        self.is_nonlinear_game = True
    
    def step(self, u):

        self.costs  = np.array([
            (cost(self.state, u) + 1200)/800 for cost in self.cost_functions
            ])
        # self.costs  = np.array([
        #     (cost(self.state, u))/100 for cost in self.cost_functions
        #     ])
        # self.costs  = np.array([
        #     cost(self.state, u)  for cost in self.cost_functions
        #     ])
        sum_costs = np.sum(self.costs)
        self.state = self.f(self.state, u)      

        terminated = False # check whether state is out of bound
        truncated = False # we don't use here
        if np.any(self.state[[0,1,4,5,8,9,12,13,16,17,20,21]] > self.state_high[[0,1,4,5,8,9,12,13,16,17,20,21]]) or np.any(self.state [[0,1,4,5,8,9,12,13,16,17,20,21]]< self.state_low[[0,1,4,5,8,9,12,13,16,17,20,21]]):
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
            -5, -12, -0.5, -0.5, 
            -5, -12, -0.5, -0.5, 
            -5, -12, -0.5, -0.5, 
            -5, -12, -0.5, -0.5, 
            -5, -12, -0.5, -0.5, 
            -5, -12, -0.5, -0.5, 
        ])
        initial_state_high = np.array([
            5, 4, 0.5, 0.5, 
            5, 4, 0.5, 0.5, 
            5, 4, 0.5, 0.5, 
            5, 4, 0.5, 0.5, 
            5, 4, 0.5, 0.5, 
            5, 4, 0.5, 0.5, 
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
        # import pdb ; pdb.set_trace()
        # Initialize tensors to store the Jacobians and Hessians
        jacobians = torch.zeros(6, batch_size, input_dim, device=z.device)
        hessians = torch.zeros(6, batch_size, input_dim, input_dim, device=z.device)
        I1 = (z[:,1] > 0).float()
        I2 = (z[:,9] > 0).float()
        # print(I1)
        # print(batch_size)
        # Manually compute the Jacobian and Hessian for the first cost function
        jacobians[0, :, 0] = 2 * (((z[:,0])**2 +(z[:,1])**2) - 1.5**2) * 2 * z[:,0]
        jacobians[0, :, 1] = 2 * (((z[:,0])**2 +(z[:,1])**2) - 1.5**2) * 2 * z[:,1] + I1*100
        jacobians[0, :, 2] = 2 * z[:, 2]
        jacobians[0, :, 3] = 2 * z[:, 3] 
        jacobians[0, :, 24] = 2 * z[:, 24]
        jacobians[0, :, 25] = 2 * z[:, 25]

        hessians[0, :, 0, 0] = 2 * (((z[:,0])**2 +(z[:,1])**2) - 1.5**2) * 2 + 8 * z[:,0] * z[:,0]
        hessians[0, :, 0, 1] = 2 * 2 * z[:,1] * 2 * z[:,0]
        hessians[0, :, 1, 0] = 2 * 2 * z[:,1] * 2 * z[:,0]
        hessians[0, :, 1, 1] = 2 * (((z[:,0])**2 +(z[:,1])**2) - 1.5**2) * 2 + 8 * z[:,1] * z[:,1]
        hessians[0, :, 2, 2] = 2
        hessians[0, :, 3, 3] = 2
        hessians[0, :, 24, 24] = 2
        hessians[0, :, 25, 25] = 2

        # Manually compute the Jacobian and Hessian for the second cost function
        jacobians[1, :, 0] = 2 * (z[:,4]-0.75*z[:,0]) * (-0.75)
        jacobians[1, :, 1] = 2 * (z[:,5]-0.75*z[:,1]) * (-0.75)

        jacobians[1, :, 4] = 2 * (z[:,4]-0.75*z[:,0]) 
        jacobians[1, :, 5] = 2 * (z[:,5]-0.75*z[:,1]) 

        # jacobians[1, :, 6] = 2 * (z[:, 6])
        # jacobians[1, :, 7] = 2 * (z[:, 7])

        jacobians[1, :, 26] = 2 * z[:, 26]
        jacobians[1, :, 27] = 2 * z[:, 27]

        hessians[1, :, 0, 0] = 2 * (-0.75) * (-0.75)
        hessians[1, :, 0, 4] = 2 * (-0.75)

        hessians[1, :, 1, 1] = 2 * (-0.75) * (-0.75)
        hessians[1, :, 1, 5] = 2 * (-0.75)

        hessians[1, :, 4, 0] = 2 * (-0.75) 
        hessians[1, :, 4, 4] = 2 

        hessians[1, :, 5, 1] = 2 * (-0.75) 
        hessians[1, :, 5, 5] = 2 

        # hessians[1, :, 6, 6] = 2
        # hessians[1, :, 7, 7] = 2
        hessians[1, :, 26, 26] = 2
        hessians[1, :, 27, 27] = 2



        # Manually compute the Jacobian and Hessian for the 3th cost function
        jacobians[2, :, 8] = 2 * (((z[:,8])**2 +(z[:,9])**2) - 4**2) * 2 * z[:,8]
        jacobians[2, :, 9] = 2 * (((z[:,8])**2 +(z[:,9])**2) - 4**2) * 2 * z[:,9] + I2*100
        jacobians[2, :, 10] = 2 * z[:, 10]
        jacobians[2, :, 11] = 2 * z[:, 11] 
        jacobians[2, :, 28] = 2 * z[:, 28]
        jacobians[2, :, 29] = 2 * z[:, 29]

        hessians[2, :, 8, 8] = 2 * (((z[:,8])**2 +(z[:,9])**2) - 4**2) * 2 + 8 * z[:,8] * z[:,8]
        hessians[2, :, 8, 9] = 2 * 2 * z[:,8] * 2 * z[:,9]
        hessians[2, :, 9, 8] = 2 * 2 * z[:,8] * 2 * z[:,9]
        hessians[2, :, 9, 9] = 2 * (((z[:,8])**2 +(z[:,9])**2) - 4**2) * 2 + 8 * z[:,9] * z[:,9]
        hessians[2, :, 10, 10] = 2
        hessians[2, :, 11, 11] = 2
        hessians[2, :, 28, 28] = 2
        hessians[2, :, 29, 29] = 2


        # Manually compute the Jacobian and Hessian for the 4th cost function
        jacobians[3, :, 8] = 2 * (z[:,12]-0.75*z[:,8]) * (-0.75)
        jacobians[3, :, 9] = 2 * (z[:,13]-0.75*z[:,9]) * (-0.75)

        jacobians[3, :, 12] = 2 * (z[:,12]-0.75*z[:,8]) 
        jacobians[3, :, 13] = 2 * (z[:,13]-0.75*z[:,9]) 

        # jacobians[3, :, 14] = 2 * (z[:, 14])
        # jacobians[3, :, 15] = 2 * (z[:, 15])

        jacobians[3, :, 30] = 2 * z[:, 30]
        jacobians[3, :, 31] = 2 * z[:, 31]

        hessians[3, :, 8, 8] = 2 * (-0.75) * (-0.75)
        hessians[3, :, 8, 12] = 2 * (-0.75)

        hessians[3, :, 9, 9] = 2 * (-0.75) * (-0.75)
        hessians[3, :, 9, 13] = 2 * (-0.75)

        hessians[3, :, 12, 8] = 2 * (-0.75) 
        hessians[3, :, 12, 12] = 2 

        hessians[3, :, 13, 9] = 2 * (-0.75) 
        hessians[3, :, 13, 13] = 2 

        # hessians[3, :, 14, 14] = 2
        # hessians[3, :, 15, 15] = 2
        hessians[3, :, 30, 30] = 2
        hessians[3, :, 31, 31] = 2

        # Manually compute the Jacobian and Hessian for the 5th cost function
        jacobians[4, :, 12] = 2 * (z[:,16]-z[:,12]) * (-1)
        jacobians[4, :, 13] = 2 * (z[:,17]-z[:,13]) * (-1)

        jacobians[4, :, 16] = 2 * (z[:,16]-z[:,12]) 
        jacobians[4, :, 17] = 2 * (z[:,17]-z[:,13]) 

        # jacobians[4, :, 18] = 2 * (z[:, 18])
        # jacobians[4, :, 19] = 2 * (z[:, 19])

        jacobians[4, :, 32] = 2 * z[:, 32]
        jacobians[4, :, 33] = 2 * z[:, 33]

        hessians[4 :, 12, 12] = 2
        hessians[4, :, 12, 16] = -2

        hessians[4, :, 13, 13] = 2 
        hessians[4, :, 13, 17] = -2 

        hessians[4, :, 16, 12] = -2 
        hessians[4, :, 16, 16] = 2 

        hessians[4, :, 17, 13] = -2 
        hessians[4, :, 17, 17] = 2 

        # hessians[4, :, 18, 18] = 2
        # hessians[4, :, 19, 19] = 2
        hessians[4, :, 32, 32] = 2
        hessians[4, :, 33, 33] = 2



# Manually compute the Jacobian and Hessian for the 6th cost function
        jacobians[5, :, 16] = 2 * (z[:,20]-0.75*z[:,16]) * (-0.75)
        jacobians[5, :, 17] = 2 * (z[:,21]-0.75*z[:,17]) * (-0.75)

        jacobians[5, :, 20] = 2 * (z[:,20]-0.75*z[:,16]) 
        jacobians[5, :, 21] = 2 * (z[:,21]-0.75*z[:,17]) 

        # jacobians[5, :, 22] = 2 * (z[:, 22])
        # jacobians[5, :, 23] = 2 * (z[:, 23])

        jacobians[5, :, 34] = 2 * z[:, 34]
        jacobians[5, :, 35] = 2 * z[:, 35]

        hessians[5, :, 16, 16] = 2 * (-0.75) * (-0.75)
        hessians[5, :, 16, 20] = 2 * (-0.75)

        hessians[5, :, 17, 17] = 2 * (-0.75) * (-0.75)
        hessians[5, :, 17, 21] = 2 * (-0.75)

        hessians[5, :, 20, 16] = 2 * (-0.75) 
        hessians[5, :, 20, 20] = 2 

        hessians[5, :, 21, 17] = 2 * (-0.75) 
        hessians[5, :, 21, 21] = 2 

        # hessians[5, :, 22, 22] = 2
        # hessians[5, :, 23, 23] = 2
        hessians[5, :, 34, 34] = 2
        hessians[5, :, 35, 35] = 2


        return jacobians, hessians
    

    @torch.jit.script
    def dynamics_jacobian(states, controls):
        """
        Computes the Jacobian matrix of the combined unicycle dynamics with respect to the state and control inputs
        for a batch of data for n_agents.

        Parameters:
        - states: A tensor of shape (batch_size, 4n), where each row is [x1, y1, v1, w1, ..., xn, yn, vn, wn].
        - controls: A tensor of shape (batch_size, 2n), where each row is [a1, u1, ..., an, un].
        - n_agents: An integer representing the number of agents.

        Returns:
        - jacobian: A tensor of shape (batch_size, 4n, 6n) containing the Jacobian matrix for each input in the batch.
        """
        batch_size = states.shape[0]
        num_players = 6
        total_state = 4 * num_players
        total_control = 2 * num_players
        dt = 0.1  # Time step

        # Initialize the Jacobian matrix with zeros
        jacobian = torch.zeros(batch_size, total_state, total_state + total_control, device=states.device)

        for i in range(num_players):
            # Indices for the current agent's state and control
            s_idx = i * 4
            c_idx = i * 2

            # Extract state variables for the current agent
            # x = states[:, s_idx]
            # y = states[:, s_idx + 1]
            # v = states[:, s_idx + 2]
            # w = states[:, s_idx + 3]

            # # Extract control variables for the current agent
            # ddx = controls[:, c_idx]
            # ddy = controls[:, c_idx + 1]

            # Compute partial derivatives for the current agent's dynamics

            # Partial derivatives for x' = x + dt * v * cos(w)
            jacobian[:, s_idx, s_idx] = 1.0  # ∂x'/∂x = 1
            jacobian[:, s_idx, s_idx + 2] = dt  # ∂x'/∂v = dt * cos(w)

            # Partial derivatives for y' = y + dt * v * sin(w)
            jacobian[:, s_idx + 1, s_idx + 1] = 1.0  # ∂y'/∂y = 1
            jacobian[:, s_idx + 1, s_idx + 3] = dt 

            # Partial derivatives for v' = v + dt * 10 * a
            jacobian[:, s_idx + 2, s_idx + 2] = 1.0  # ∂v'/∂v = 1
            jacobian[:, s_idx + 2, total_state + c_idx] = 5 * dt  # ∂v'/∂a = dt * 10

            # Partial derivatives for w' = w + dt * u
            jacobian[:, s_idx + 3, s_idx + 3] = 1.0  # ∂w'/∂w = 1
            jacobian[:, s_idx + 3, total_state + c_idx + 1] = 5 * dt  # ∂w'/∂u = dt

        return jacobian


    @torch.jit.script
    def dynamics(states, controls):
        
        next_states = torch.zeros_like(states)
        next_states[:,0] = states[:,0] + 0.1 * states[:,2] 
        next_states[:,1] = states[:,1] + 0.1 * states[:,3] 
        next_states[:,2] = states[:,2] + 0.5 * controls[:,0]
        next_states[:,3] = states[:,3] + 0.5 * controls[:,1]

        next_states[:,4] = states[:,4] + 0.1 * states[:,6] 
        next_states[:,5] = states[:,5] + 0.1 * states[:,7] 
        next_states[:,6] = states[:,6] + 0.5 * controls[:,2]
        next_states[:,7] = states[:,7] + 0.5 * controls[:,3]


        next_states[:,8] = states[:,8] + 0.1 * states[:,10] 
        next_states[:,9] = states[:,9] + 0.1 * states[:,11] 
        next_states[:,10] = states[:,10] + 0.5 * controls[:,4]
        next_states[:,11] = states[:,11] + 0.5 * controls[:,5]

        next_states[:,12] = states[:,12] + 0.1 * states[:,14] 
        next_states[:,13] = states[:,13] + 0.1 * states[:,15]
        next_states[:,14] = states[:,14] + 0.5 * controls[:,6]
        next_states[:,15] = states[:,15] + 0.5 * controls[:,7]

        next_states[:,16] = states[:,16] + 0.1 * states[:,18] 
        next_states[:,17] = states[:,17] + 0.1 * states[:,19] 
        next_states[:,18] = states[:,18] + 0.5 * controls[:,8]
        next_states[:,19] = states[:,19] + 0.5 * controls[:,9]

        next_states[:,20] = states[:,20] + 0.1 * states[:,22] 
        next_states[:,21] = states[:,21] + 0.1 * states[:,23]
        next_states[:,22] = states[:,22] + 0.5 * controls[:,10]
        next_states[:,23] = states[:,23] + 0.5 * controls[:,11]

        return next_states