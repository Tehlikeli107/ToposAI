import torch


class TopologicalPlanner:
    """
    Linear-dynamics planning baseline.

    This module contrasts random action search with an analytic least-squares
    action for a one-step linear transition model:

        next_state = T @ start_state + C @ action

    The pseudo-inverse solution is exact only when the requested goal lies in
    the reachable affine subspace. Otherwise it returns the least-squares
    action and a non-zero residual distance. It is a planning toy model, not a
    replacement for reinforcement learning in unknown or nonlinear
    environments.
    """

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.transition_matrix = torch.randn(state_dim, state_dim)
        self.control_matrix = torch.randn(state_dim, action_dim)

    def reinforcement_learning_simulate(self, start_state, goal_state, num_episodes=1000):
        """
        Random-search baseline over one-step actions.

        This intentionally simple baseline samples actions and keeps the best
        one under the known linear dynamics. It is not a full PPO/Q-learning
        implementation.
        """
        best_action = None
        min_dist = float("inf")
        T = self.transition_matrix.to(device=start_state.device, dtype=start_state.dtype)
        C = self.control_matrix.to(device=start_state.device, dtype=start_state.dtype)

        for _ in range(num_episodes):
            action = torch.randn(self.action_dim, device=start_state.device, dtype=start_state.dtype)
            next_state = T @ start_state + C @ action
            dist = torch.norm(next_state - goal_state)

            if dist < min_dist:
                min_dist = dist
                best_action = action

        return best_action, min_dist

    def topos_contravariant_pullback(self, start_state, goal_state):
        """
        Compute a least-squares action with the Moore-Penrose pseudo-inverse.

        Returns:
            (action, residual_distance)

        A residual near zero means the one-step goal is reachable under the
        current linear model. A larger residual means the action is only the
        closest least-squares solution.
        """
        T = self.transition_matrix.to(device=start_state.device, dtype=start_state.dtype)
        C = self.control_matrix.to(device=start_state.device, dtype=start_state.dtype)
        state_diff = goal_state - T @ start_state
        control_pinv = torch.linalg.pinv(C)
        action = control_pinv @ state_diff

        predicted_state = T @ start_state + C @ action
        residual = torch.norm(predicted_state - goal_state)
        return action, residual
