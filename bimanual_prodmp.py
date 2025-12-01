import os
import time
import argparse
import numpy as np
import pandas as pd

# from ur5e_sim import UR5eSim
from bimanual_sim import BimanualSim

import torch
from mp_pytorch.mp import MPFactory
from mp_pytorch.util import tensor_linspace


class ProDMPController:
    """Controller that uses ProDMP for trajectory generation with conditioning."""

    def __init__(self, num_dof=6, num_basis=20, dt=0.01, tau=2.0):
        """Initialize ProDMP controller.

        Args:
            num_dof: Number of degrees of freedom (6 for UR5e)
            num_basis: Number of basis functions
            dt: Time step for trajectory generation
            tau: Default trajectory duration (normalized to [0, 1] for demos)
        """
        self.num_dof = num_dof
        self.num_basis = num_basis
        self.dt = dt
        self.tau = tau

        # ProDMP configuration
        self.config = {
            "mp_type": "prodmp",
            "num_dof": num_dof,
            "tau": tau,
            "learn_tau": False,  # Disabled due to mp_pytorch bug
            "mp_args": {
                "num_basis": num_basis,
                "basis_bandwidth_factor": 2,
                "num_basis_outside": 0,
                "alpha": 25,
                "alpha_phase": 2,
                "dt": dt,
                "relative_goal": True,
                "auto_scale_basis": True
            }
        }

        self.mp = MPFactory.init_mp(**self.config)

        # Learned parameters from demonstrations
        self.learned_params = None
        self.mean_params = None
        self.params_cov = None

        # Current trajectory state
        self.current_time = 0.0
        self.current_pos = None
        self.current_vel = None
        self.trajectory_times = None
        self.trajectory_pos = None
        self.trajectory_vel = None
        self.current_step = 0

    def load_demos_from_folder(self, demo_folder_path, is_left=True):
        """Load demonstration trajectories from a folder with new structure.

        Args:
            demo_folder_path: Path to folder containing demo subfolders (e.g., data/new_data/front_five/)
            is_left: Whether to load left or right robot demos

        Returns:
            demos: numpy array of shape (num_demos, num_timesteps, num_dof)
        """
        robot = "left" if is_left else "right"

        demos = []

        # Get all demo subdirectories (e.g., demo_1, demo_2, ...)
        demo_dirs = sorted([d for d in os.listdir(demo_folder_path)
                           if os.path.isdir(os.path.join(demo_folder_path, d)) and d.startswith("demo_")])

        for demo_dir in demo_dirs:
            # Look for the robot-specific joint trajectory file in this demo folder
            demo_path = os.path.join(demo_folder_path, demo_dir)
            trajectory_files = [f for f in os.listdir(demo_path)
                              if f.startswith(f"{robot}_joint_trajectory_demo") and f.endswith(".csv")]

            if trajectory_files:
                file_path = os.path.join(demo_path, trajectory_files[0])
                demo_data = pd.read_csv(file_path)
                columns = demo_data.columns.str.contains('position')
                demos.append(demo_data.loc[:, columns].values)

        if not demos:
            raise ValueError(f"No demos found for {robot} robot in {demo_folder_path}")

        # Convert to numpy array with same length
        min_len = min(len(d) for d in demos)
        demos = np.deg2rad(np.array([d[:min_len] for d in demos]))

        print(f"Loaded {len(demos)} demos with {min_len} timesteps each")
        return demos

    def learn_from_demos(self, demos):
        """Learn ProDMP parameters from demonstration trajectories.

        Args:
            demos: numpy array of shape (num_demos, num_timesteps, num_dof)
        """
        # Create time vector normalized to [0, tau]
        demo_times = tensor_linspace(0.0, self.tau, demos.shape[1]).unsqueeze(0).repeat(demos.shape[0], 1)
        demo_trajs = torch.tensor(demos, dtype=torch.float32)

        # Learn parameters from all demos together
        params_dict = self.mp.learn_mp_params_from_trajs(demo_times, demo_trajs)

        # Store learned parameters
        self.learned_params = params_dict["params"]
        self.learned_init_time = params_dict["init_time"]
        self.learned_init_pos = params_dict["init_pos"]
        self.learned_init_vel = params_dict["init_vel"]

        # # Compute mean parameters across demos
        # self.mean_params = self.learned_params.mean(dim=0, keepdim=True)

        # # Compute covariance matrix for sampling
        # if self.learned_params.shape[0] > 1:
        #     # Center the parameters
        #     centered_params = self.learned_params - self.mean_params
        #     # Compute covariance: (1/(n-1)) * X^T @ X
        #     self.params_cov = (centered_params.T @ centered_params) / (self.learned_params.shape[0] - 1)
        #     # Compute Cholesky decomposition for sampling
        #     # Add small regularization for numerical stability
        #     reg = 1e-6 * torch.eye(self.params_cov.shape[0])
        #     self.params_L = torch.linalg.cholesky(self.params_cov + reg)
        # else:
        #     # Single demo - use small default covariance
        #     num_params = self.learned_params.shape[1]
        #     self.params_cov = torch.eye(num_params) * 0.01
        #     self.params_L = torch.eye(num_params) * 0.1

        # print(f"Learned parameters shape: {self.learned_params.shape}")
        # print(f"Mean parameters shape: {self.mean_params.shape}")
        # print(f"Parameters covariance shape: {self.params_cov.shape}")

        return params_dict

    def _set_goals_in_params(self, params, goal_relative):
        """Set goal values in parameter tensor.

        Args:
            params: [batch, num_params] parameter tensor
            goal_relative: [num_dof] relative goal positions

        Returns:
            params: modified parameter tensor with new goals
        """
        params = params.clone()
        batch_size = params.shape[0]

        start_idx = 1 if hasattr(self.mp, 'learn_tau') and self.mp.learn_tau else 0

        for dof in range(self.num_dof):
            goal_idx = start_idx + dof * (self.num_basis + 1) + self.num_basis
            params[:, goal_idx] = goal_relative[dof]

        return params

    def create_prodmp_params(self, weights, goals, tau=None):
        """Create correctly structured ProDMP parameters.

        Args:
            weights: [batch, num_dof, num_basis] - trajectory shape parameters
            goals: [batch, num_dof] - target positions (relative to start)
            tau: [batch, 1] - optional trajectory duration

        Returns:
            params: [batch, num_params] - correctly structured parameters
        """
        batch_size = goals.shape[0]

        if weights.ndim == 2:
            weights = weights.reshape(batch_size, self.num_dof, self.num_basis)

        params_list = []
        for b in range(batch_size):
            param_b = []

            if tau is not None:
                param_b.append(tau[b, 0].item())

            for dof in range(self.num_dof):
                param_b.extend(weights[b, dof, :].tolist())
                param_b.append(goals[b, dof].item())

            params_list.append(param_b)

        return torch.tensor(params_list, dtype=torch.float32)

    def condition_trajectory(self, start_pos, goal_pos, start_vel=None, speed_factor=1.0):
        """Generate a trajectory conditioned on start, goal, and speed.

        Args:
            start_pos: Starting joint positions [num_dof]
            goal_pos: Target joint positions [num_dof]
            start_vel: Starting joint velocities [num_dof] (default: zeros)
            speed_factor: Multiplier for trajectory speed (>1 = faster, <1 = slower)

        Returns:
            trajectory_pos: [num_timesteps, num_dof]
            trajectory_vel: [num_timesteps, num_dof]
        """
        if start_vel is None:
            start_vel = np.zeros(self.num_dof)

        # Convert to tensors
        init_pos = torch.tensor(start_pos, dtype=torch.float32).unsqueeze(0)
        init_vel = torch.tensor(start_vel, dtype=torch.float32).unsqueeze(0)
        init_time = torch.zeros(1)

        # Calculate relative goal (goal - start)
        goal_relative = torch.tensor(goal_pos - start_pos, dtype=torch.float32).unsqueeze(0)

        # Adjust tau based on speed factor
        adjusted_tau = self.tau / speed_factor
        print("original tau: ", self.tau)
        print("adjusted tau: ", adjusted_tau)

        # Calculate number of timesteps based on adjusted tau
        num_timesteps = int(adjusted_tau / self.dt)
        print("timesteps: ", num_timesteps)
        times = torch.linspace(0, adjusted_tau, num_timesteps).unsqueeze(0)

        # Use learned weights if available, otherwise use zeros (straight-line)
        if self.mean_params is not None:
            # Extract weights from learned mean parameters
            weights = self._extract_weights_from_params(self.mean_params)
        else:
            weights = torch.zeros(1, self.num_dof, self.num_basis)

        # Create parameters with new goal
        params = self.create_prodmp_params(weights, goal_relative)

        # Generate trajectory
        self.mp.update_inputs(
            times=times,
            params=params,
            init_time=init_time,
            init_pos=init_pos,
            init_vel=init_vel
        )

        traj_dict = self.mp.get_trajs(get_pos=True, get_vel=True)

        # Store trajectory state
        self.trajectory_times = times.squeeze(0).numpy()
        self.trajectory_pos = traj_dict["pos"].squeeze(0).numpy()
        self.trajectory_vel = traj_dict["vel"].squeeze(0).numpy()
        self.current_step = 0
        self.current_time = 0.0
        self.current_pos = init_pos.squeeze(0).numpy().copy()
        self.current_vel = init_vel.squeeze(0).numpy().copy()

        return self.trajectory_pos, self.trajectory_vel

    def _extract_weights_from_params(self, params):
        """Extract weights from ProDMP parameter vector.

        Args:
            params: [batch, num_params] parameter tensor

        Returns:
            weights: [batch, num_dof, num_basis]
        """
        batch_size = params.shape[0]
        weights = torch.zeros(batch_size, self.num_dof, self.num_basis)

        start_idx = 1 if hasattr(self.mp, 'learn_tau') and self.mp.learn_tau else 0

        for dof in range(self.num_dof):
            dof_start = start_idx + dof * (self.num_basis + 1)
            weights[:, dof, :] = params[:, dof_start:dof_start + self.num_basis]

        return weights

    def get_next_step(self, current_robot_pos=None, current_robot_vel=None):
        """Get the next step of the trajectory

        Args:
            current_robot_pos: Current robot joint positions
            current_robot_vel: Current robot joint velocities

        Returns:
            target_pos: Target joint positions for next step
            target_vel: Target joint velocities for next step
            done: Whether trajectory is complete
        """
        if self.trajectory_pos is None:
            raise RuntimeError("No trajectory generated. Call condition_trajectory first.")

        # Check if trajectory is complete
        if self.current_step >= len(self.trajectory_pos):
            return self.trajectory_pos[-1], self.trajectory_vel[-1], True

        # Get target for this step
        target_pos = self.trajectory_pos[self.current_step]
        target_vel = self.trajectory_vel[self.current_step]

        # Update current state (could be used for replanning in advanced scenarios)
        if current_robot_pos is not None:
            self.current_pos = current_robot_pos
        else:
            self.current_pos = target_pos.copy()

        if current_robot_vel is not None:
            self.current_vel = current_robot_vel
        else:
            self.current_vel = target_vel.copy()

        # Advance step counter
        self.current_step += 1
        self.current_time = self.current_step * self.dt

        return target_pos, target_vel, False

    def reset(self):
        """Reset trajectory state."""
        self.current_step = 0
        self.current_time = 0.0
        self.trajectory_pos = None
        self.trajectory_vel = None


# Global controller and simulation state
left_controller = None
right_controller = None
sim = None
last_time = None


def loop_callback():
    """Callback for simulation loop - computes next step of ProDMP policy."""
    global left_controller, right_controller, sim, last_time

    if left_controller is None or left_controller.trajectory_pos is None:
        return
    
    # Rate limiting based on dt
    current_time = time.time()
    if last_time is not None and (current_time - last_time) < left_controller.dt:
        return
    
    last_time = current_time


    for robot_id, controller in [(2, left_controller), (1, right_controller)]:
        # Get current robot state from simulation
        current_robot_pos = sim.robot1_controls if robot_id == 1 else sim.robot2_controls

        controller = left_controller if robot_id == 1 else right_controller

        # Get next step from ProDMP controller
        target_pos, target_vel, done = controller.get_next_step(current_robot_pos)
        target_pos = -1.0 * target_pos  # Negate for MuJoCo coordinate system

        # Send commands to robot (trajectory is already in radians and MuJoCo space)
        sim.set_robot_control(robot_id, target_pos)

    sim.d.ctrl[:] = np.concatenate([sim.robot2_controls, sim.robot1_controls])

    if done:
        # Reset to loop the trajectory (or you could stop here)
        left_controller.current_step = 0
        right_controller.current_step = 0


def main():
    global left_controller, right_controller, sim

    # ========================================
    # PARSE COMMAND-LINE ARGUMENTS
    # ========================================

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'demo_folder_path',
        type=str,
        help='Path to demo folder (e.g., data/new_data/front_five/)'
    )
    parser.add_argument(
        '--num-basis',
        type=int,
        default=20,
        help='Number of basis functions (default: 20)'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.01,
        help='Control timestep in seconds (default: 0.01)'
    )
    parser.add_argument(
        '--speed-factor',
        type=float,
        default=1.0,
        help='Speed multiplier: >1 = faster, <1 = slower (default: 1.0)'
    )

    args = parser.parse_args()

    # ========================================
    # CONFIGURATION
    # ========================================

    sim = BimanualSim(loop_callback=loop_callback)

    # Demo data path from command-line argument
    demo_folder_path = args.demo_folder_path

    # ProDMP parameters
    num_basis = args.num_basis
    dt = args.dt  # Control rate

    # Speed factor: >1 = faster, <1 = slower
    speed_factor = args.speed_factor

    # Starting velocity (Need to change this when operating online)
    start_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # ========================================
    # INITIALIZATION
    # ========================================

    print("Initializing ProDMP Controller...")
    # Left arm controller
    left_controller = ProDMPController(
        num_dof=6,
        num_basis=num_basis,
        dt=dt
    )

    # Load and learn from demonstrations
    left_demos = left_controller.load_demos_from_folder(demo_folder_path)

    # Learning ProDMP parameters from demonstrations
    left_controller.learn_from_demos(left_demos)

    # Right arm controller
    right_controller = ProDMPController(
        num_dof=6,
        num_basis=num_basis,
        dt=dt
    )

    # Load and learn from demonstrations
    right_demos = right_controller.load_demos_from_folder(demo_folder_path, is_left=False)
    
    # Learning ProDMP parameters from demonstrations
    right_controller.learn_from_demos(right_demos)  # Using same demos for simplicity

    # ========================================
    # Condition on start and goal
    # ========================================

    # Test starts
    left_start = left_demos[0, 0, :]
    right_start = right_demos[0, 0, :]

    # Test goals
    left_goal = left_demos[0, -1, :]
    right_goal = right_demos[0, -1, :]

    left_traj_pos, left_traj_vel = left_controller.condition_trajectory(
        start_pos=left_start,
        goal_pos=left_goal,
        start_vel=start_vel,
        speed_factor=speed_factor
    )

    # Verify convergence
    error = np.abs(left_traj_pos[-1] - left_goal).max()
    print(f"  Goal error: {error:.6f} rad")

    right_traj_pos, right_traj_vel = right_controller.condition_trajectory(
        start_pos=right_start,
        goal_pos=right_goal,
        start_vel=start_vel,
        speed_factor=speed_factor
    )

    # Verify convergence
    error = np.abs(right_traj_pos[-1] - right_goal).max()
    print(f"  Goal error: {error:.6f} rad")

    # ========================================
    # RUN SIMULATION
    # ========================================

    print("\nStarting simulation...")
    print("Press SPACE to pause/unpause")

    sim.loop()


if __name__ == "__main__":
    main()
