import torch


class LQGame:
    def __init__(
        self, 
        nx: int, 
        nu: int, 
        horizon: int, 
        n_players: int, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        Q: torch.Tensor, 
        R: torch.Tensor, 
        q: torch.Tensor, 
        r: torch.Tensor,
        players_u_index_list: torch.Tensor
        ):
        self.nx = nx
        self.nu = nu
        self.horizon = horizon
        self.n_players = n_players
        self.A_list = A
        self.B_list = B
        self.Q_list = Q
        self.R_list = R
        self.q_list = q
        self.r_list = r
        self.players_u_index_list = players_u_index_list
    def copy(self):
        new_g = LQGame(
            self.nx, 
            self.nu, 
            self.horizon, 
            self.n_players, 
            [self.A_list[t].clone() for t in range(self.horizon)], 
            [self.B_list[t].clone() for t in range(self.horizon)], 
            [[self.Q_list[t][ii].clone() for ii in range(self.n_players)] for t in range(self.horizon+1)], 
            [[self.R_list[t][ii].clone() for ii in range(self.n_players)] for t in range(self.horizon)], 
            [[self.q_list[t][ii].clone() for ii in range(self.n_players)] for t in range(self.horizon+1)], 
            [[self.r_list[t][ii].clone() for ii in range(self.n_players)] for t in range(self.horizon)], 
            self.players_u_index_list
        )
        return new_g

# define a class for nonlinear games, having the following attributes:
# nx: the number of states
# nu: the number of control inputs
# horizon: the number of time steps
# n_players: the number of players
# f_list: a list of the dynamics functions
# l_list: a list of the cost functions
# x0: the initial state
# players_u_index_list: a list of the control indices for each player

class NonlinearGame:
    def __init__(
        self, 
        nx: int, 
        nu: int, 
        horizon: int, 
        n_players: int, 
        f_list, 
        f_list_combined, # combined input [x,u] -> x_next
        l_list: list,
        l_list_combined: list, # combined input [x,u] -> cost
        players_u_index_list: torch.Tensor
        ):
        self.nx = nx
        self.nu = nu
        self.horizon = horizon
        self.n_players = n_players
        self.f_list = f_list
        self.f_list_combined = f_list_combined
        self.l_list = l_list
        self.l_list_combined = l_list_combined
        self.players_u_index_list = players_u_index_list


def lq_approximation(
    ng: NonlinearGame, 
    x_traj: torch.Tensor, 
    u_traj: torch.Tensor, 
    lq_approx: LQGame
    ) -> LQGame:
    for t in range(ng.horizon):
        # evaluate the dynamics
        xt = x_traj[t, :].clone().detach().requires_grad_(True).reshape(ng.nx,1)
        ut = u_traj[t, :].clone().detach().requires_grad_(True).reshape(ng.nu,1)
        # import pdb; pdb.set_trace()
        var = torch.cat((xt, ut), 0).squeeze()
        dyn_jac = torch.autograd.functional.jacobian(ng.f_list_combined, var).reshape(ng.nx, ng.nx+ng.nu)
        # import pdb; pdb.set_trace()
        lq_approx.A_list[t] = dyn_jac[:, :ng.nx].clone().detach().reshape(ng.nx, ng.nx)
        lq_approx.B_list[t] = dyn_jac[:, ng.nx:].clone().detach().reshape(ng.nx, ng.nu)        
        # evaluate the cost function
        for ii in range(ng.n_players):
            # evaluate the gradient
            if t == 0:
                x_next_tensor = x_traj[t+1, :].clone().detach().requires_grad_(True).reshape(ng.nx,1)
                var_cost = torch.cat((x_next_tensor, ut), 0).squeeze()
                cost_jac = torch.autograd.functional.jacobian(ng.l_list_combined[ii], var_cost).reshape(ng.nx+ng.nu,1)
                lq_approx.q_list[t][ii] = cost_jac[:ng.nx].detach().reshape(ng.nx,1) # q_{t}
                cost_hess = torch.autograd.functional.hessian(ng.l_list_combined[ii], var_cost).reshape(ng.nx+ng.nu, ng.nx+ng.nu)
                lq_approx.Q_list[t][ii] = cost_hess[:ng.nx, :ng.nx].detach().reshape(ng.nx, ng.nx) # Q_{t}
            
            x_next_tensor = x_traj[t+1, :].clone().detach().requires_grad_(True).reshape(ng.nx,1)
            var_cost = torch.cat((x_next_tensor, ut), 0).squeeze()
            cost_jac = torch.autograd.functional.jacobian(ng.l_list_combined[ii], var_cost).reshape(ng.nx+ng.nu,1)
            lq_approx.q_list[t+1][ii] = cost_jac[:ng.nx].detach().reshape(ng.nx,1) # q_{t+1}
            lq_approx.r_list[t][ii] = cost_jac[ng.nx:].detach().reshape(ng.nu,1)
            
            # evaluate the hessian
            hess = torch.autograd.functional.hessian(ng.l_list_combined[ii], var_cost).reshape(ng.nx+ng.nu, ng.nx+ng.nu)
            lq_approx.Q_list[t+1][ii]= hess[:ng.nx, :ng.nx].detach().reshape(ng.nx, ng.nx) # Q_{t+1}
            lq_approx.R_list[t][ii] = hess[ng.nx:, ng.nx:].detach().reshape(ng.nu, ng.nu)
    return lq_approx

def lq_approximation_quadratic_cost(
    ng: NonlinearGame, 
    x_traj: torch.Tensor, 
    u_traj: torch.Tensor, 
    lq_approx: LQGame,
    quadratic_cost_Q: torch.Tensor,
    quadratic_cost_R: torch.Tensor,
    quadratic_cost_q: torch.Tensor,
    quadratic_cost_r: torch.Tensor,
    ) -> LQGame:
    for t in range(ng.horizon):
        # evaluate the dynamics
        xt = x_traj[t, :].clone().detach().requires_grad_(True).reshape(ng.nx,1)
        ut = u_traj[t, :].clone().detach().requires_grad_(True).reshape(ng.nu,1)
        var = torch.cat((xt, ut), 0).squeeze()
        dyn_jac = torch.autograd.functional.jacobian(ng.f_list_combined, var).reshape(ng.nx, ng.nx+ng.nu)
        lq_approx.A_list[t] = dyn_jac[:, :ng.nx].clone().detach().reshape(ng.nx, ng.nx)
        lq_approx.B_list[t] = dyn_jac[:, ng.nx:].clone().detach().reshape(ng.nx, ng.nu)        
        # evaluate the cost function
        for ii in range(ng.n_players):
            lq_approx.q_list[t+1][ii] = quadratic_cost_Q[t+1][ii] @ x_traj[t+1,:].reshape(ng.nx,1) + quadratic_cost_q[t+1][ii]
            lq_approx.r_list[t][ii] = quadratic_cost_R[t][ii] @ u_traj[t,:].reshape(ng.nu,1) + quadratic_cost_r[t][ii]
    return lq_approx


def simulate_a_policy_and_return_a_traj_and_diff(
    f_list_combined, 
    P: torch.Tensor, 
    a: torch.Tensor, 
    x_traj: torch.Tensor, 
    u_traj: torch.Tensor, 
    nx: int, 
    nu: int, 
    horizon: int
    ):
    x = x_traj[0, :].clone().detach().requires_grad_(True).reshape(nx,1)
    old_x_traj = x_traj.clone()
    for t in range(horizon):
        # import pdb; pdb.set_trace()
        u = u_traj[t, :].reshape(nu,1) - P[t] @ (x - old_x_traj[t, :].reshape(nx,1)) - a[t].reshape(nu,1)
        # import pdb; pdb.set_trace()
        x_next = f_list_combined(torch.cat((x, u), 0).squeeze()).reshape(nx,1)
        x_traj[t+1, :] = x_next.squeeze()
        u_traj[t, :] = u.squeeze()
        x = x_next
    diff = x_traj - old_x_traj
    max_diff = torch.max(torch.abs(diff))
    return x_traj, u_traj, max_diff


class ilq_games_solver:
    def __init__(
        self, 
        step_size_contraction_rate = 0.5, 
        max_iter = 200,
        line_search_max_iter = 20,
        line_search_tol = torch.inf,
        tol = 1e-1,
        verbose = False, 
        quadratic_cost = False
        # line_search_metric = 'trajectory difference' # 'policy offset' or 'trajectory difference'
        ):
        # assert line_search_metric in ['policy offset', 'trajectory difference'] # only two options for now
        # define boring variables
        self.step_size_contraction_rate = step_size_contraction_rate
        self.max_iter = max_iter
        self.line_search_max_iter = line_search_max_iter
        self.line_search_tol = line_search_tol
        self.tol = tol
        self.verbose = verbose
        self.quadratic_cost = quadratic_cost
        # self.line_search_metric = line_search_metric
    
    def line_search(
        self, 
        P: torch.Tensor, a: torch.Tensor, 
        x_traj: torch.Tensor, u_traj: torch.Tensor, 
        f_list_combined, nx: int, nu: int, horizon: int):
        alpha = 1 # initial step size
        for iter in range(self.line_search_max_iter):
            a = a * alpha
            x_traj, u_traj, error = simulate_a_policy_and_return_a_traj_and_diff(f_list_combined, P, a, x_traj, u_traj, nx,nu, horizon)
            if error < self.line_search_tol:
                print('line search error: ', error) if self.verbose else None
                print('line search iter: ', iter) if self.verbose else None
                return x_traj, u_traj, P, a, error
            else:
                alpha = alpha * self.step_size_contraction_rate
        return x_traj, u_traj, P, a, error
    
    
    def solve(
        self, 
        ng: NonlinearGame, lq_approx: LQGame, 
        P: torch.Tensor, a: torch.Tensor, 
        x_traj: torch.Tensor, u_traj: torch.Tensor, 
        S: torch.Tensor, YP: torch.Tensor, Ya: torch.Tensor, 
        quadratic_cost_Q: torch.Tensor = None,
        quadratic_cost_R: torch.Tensor = None,
        quadratic_cost_q: torch.Tensor = None,
        quadratic_cost_r: torch.Tensor = None
        ):
        assert not (self.quadratic_cost and (quadratic_cost_Q is None or quadratic_cost_R is None or quadratic_cost_q is None or quadratic_cost_r is None))
        convergence = False
        for iter in range(self.max_iter):
            # 1. get the lq approximation
            if self.quadratic_cost:
                lq_approx = lq_approximation_quadratic_cost(
                    ng, 
                    x_traj, 
                    u_traj, 
                    lq_approx, 
                    quadratic_cost_Q, 
                    quadratic_cost_R, 
                    quadratic_cost_q, 
                    quadratic_cost_r
                    )
            else:
                lq_approx = lq_approximation(
                    ng, 
                    x_traj, 
                    u_traj, 
                    lq_approx
                    )
            # 2. solve the lq game
            P, a = lq_solver(
                P, a, 
                lq_approx.A_list, lq_approx.B_list, lq_approx.Q_list, lq_approx.R_list, lq_approx.q_list, lq_approx.r_list, 
                ng.players_u_index_list, ng.nx, ng.nu, ng.horizon, ng.n_players,
                S, YP, Ya)
            # 3. line search for stabilizing the update
            x_traj, u_traj, P, a, error = self.line_search(P, a, x_traj,u_traj, ng.f_list_combined, ng.nx,ng.nu, ng.horizon)
            if error < self.tol:
                print('converged at iter: ', iter) if self.verbose else None
                convergence = True
                break
        return x_traj, u_traj, P, a, convergence, error



@torch.jit.script
def lq_solver(
    P: torch.Tensor, a: torch.Tensor, 
    A_list: torch.Tensor, B_list: torch.Tensor, 
    Q_list: torch.Tensor, R_list: torch.Tensor, q_list: torch.Tensor, r_list: torch.Tensor, 
    players_u_index_list: torch.Tensor, nx: int, nu: int, horizon: int, n_players: int, 
    S: torch.Tensor, YP: torch.Tensor, Ya:torch.Tensor
    ):
    gamma = 0.9
    Z = Q_list[-1]
    kesi = q_list[-1]
    for t in range(horizon-1, -1, -1):
        A = A_list[t]
        B = B_list[t]
        for ii in range(n_players):
            udxi = players_u_index_list[ii]
            BiZi = B[:, udxi].T @ Z[ii]
            S[udxi, :] = R_list[t][ii][udxi, :]*gamma**t + BiZi @ B
            YP[udxi, :] = BiZi @ A
            Ya[udxi, :] = B[:, udxi].T @ kesi[ii] + r_list[t][ii][udxi]*gamma**t
        # print(S)
        Sinv = torch.linalg.inv(S)
        P_mat = Sinv @ YP
        a_vec = Sinv @ Ya
        F = A-B @ P_mat
        beta =  - B @ a_vec
        for ii in range(n_players):
            PRi = P_mat.T @ R_list[t][ii]*gamma**t
            # import pdb; pdb.set_trace()
            kesi[ii] = F.T @ (kesi[ii] + Z[ii] @ beta) + q_list[t][ii]*gamma**t + PRi @ a_vec - P_mat.T @ r_list[t][ii]*gamma**t
            Z[ii] = F.T @ Z[ii] @ F + Q_list[t][ii]*gamma**t + PRi @ P_mat
        P[t] = P_mat
        a[t] = a_vec
    return P, a
# TODO: batch version of the ilq_games_solver



# TODO: batch version of the lq_solver
@torch.jit.script
def lq_solver_batch(
    P: torch.Tensor, a: torch.Tensor, 
    A_list: torch.Tensor, B_list: torch.Tensor, 
    Q_list: torch.Tensor, R_list: torch.Tensor, q_list: torch.Tensor, r_list: torch.Tensor, 
    players_u_index_list: torch.Tensor, nx: int, nu: int, horizon: int, n_players: int, 
    S: torch.Tensor, YP: torch.Tensor, Ya:torch.Tensor
    ):
    # note that all the inputs are batched, batch_size * horizon * nx * nx
    Z = Q_list[:, -1] # batch_size * nx * nx
    kesi = q_list[:, -1] # batch_size * nx
    for t in range(horizon-1, -1, -1):
        A = A_list[:, t] # batch_size * nx * nx
        B = B_list[:, t] # batch_size * nx * nu
        for ii in range(n_players):
            udxi = players_u_index_list[ii]
            BiZi = B[:, :, udxi].transpose(1,2) @ Z[:, ii]
            S[:, udxi, :] = R_list[:, t, ii, udxi, :] + BiZi @ B
            YP[:, udxi, :] = BiZi @ A
            Ya[:, udxi, :] = B[:, :, udxi].transpose(1,2) @ kesi[:, ii] + r_list[:, t, ii, udxi]
        Sinv = torch.linalg.inv(S)
        P_mat = Sinv @ YP
        a_vec = Sinv @ Ya
        F = A-B @ P_mat
        beta =  - B @ a_vec
        for ii in range(n_players):
            PRi = P_mat.transpose(1,2) @ R_list[:, t, ii]
            # import pdb; pdb.set_trace()
            kesi[:, ii] = F.transpose(1,2) @ (kesi[:, ii] + Z[:, ii] @ beta) + q_list[:, t, ii] + PRi @ a_vec - P_mat.transpose(1,2) @ r_list[:, t, ii]
            Z[:, ii] = F.transpose(1,2) @ Z[:, ii] @ F + Q_list[:, t, ii] + PRi @ P_mat
        P[:, t] = P_mat
        a[:, t] = a_vec
    return P, a

