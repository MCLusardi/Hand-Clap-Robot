import mujoco
from mujoco import viewer
import numpy as np
import pandas as pd
from ur_ikfast import ur_kinematics
from utils.transform_utils import position_quaternion_to_transform, transform_relative
from ur_ikfast import ur_kinematics
import pickle

class BimanualSim:
    def __init__(self, model_path="./mujoco_menagerie/universal_robots_ur5e/bimanual_scene.xml", loop_callback=None):
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        self.loop_callback = loop_callback if loop_callback is not None else self.default_loop_callback
        self.paused = False
        self.ee_trajectory = None
        self.robot1_controls = np.zeros(6)
        self.robot2_controls = np.zeros(6)
        self.timestep = 0

    def loop(self):
        def key_callback(keycode):
            if chr(keycode) == ' ':
                self.paused = not self.paused

        with viewer.launch_passive(self.m, self.d, key_callback=key_callback) as v:
            while v.is_running():
                if not self.paused:
                    self.loop_callback()
                    mujoco.mj_step(self.m, self.d)
                    v.sync()

    def default_loop_callback(self):
        self.d.ctrl[:] = np.concatenate([self.robot1_controls, self.robot2_controls])

    def set_robot_control(self, robot, controls):
        if robot == 1:
            self.robot1_controls = controls
        elif robot == 2:
            self.robot2_controls = controls

    def update_robot_joints(self, robot, joint_angles):
        """Update robot joints immediately (for live/reactive control).

        Args:
            robot: Robot number (1 or 2)
            joint_angles: Array of 6 joint angles in radians
        """
        if robot == 1:
            self.robot1_controls = joint_angles
        elif robot == 2:
            self.robot2_controls = joint_angles
        self.d.ctrl[:] = np.concatenate([self.robot1_controls, self.robot2_controls])

    def set_robot_control_by_ee(self, robot, ee_pos, ee_quat, make_relative=False):
        ur5e_arm = ur_kinematics.URKinematics('ur5e')
        ee_trans = position_quaternion_to_transform(ee_pos, ee_quat)

        if make_relative:
            base_trans = self.get_base_transform(robot)
        ee_trans = transform_relative(base_trans, ee_trans)

        joint_configs = ur5e_arm.inverse(ee_trans[:-1,:], False)

        if robot == 1:
            self.robot1_controls[:-1] = joint_configs
        elif robot == 2:
            self.robot2_controls[:-1] = joint_configs

    def set_gripper(self, robot, ee_angle):
        if robot == 1:
            self.robot1_controls[-1] = ee_angle
        elif robot == 2:
            self.robot2_controls[-1] = ee_angle

    def get_base_transform(self, robot):
        if robot == 1:
            body = self.m.body('arm1_base')
        elif robot == 2:
            body = self.m.body('arm2_base')
        return position_quaternion_to_transform(body.pos, body.quat)

def get_joint_angles(data, idx):
  shoulder = data.iloc[idx]['shoulder_joint_position']
  upper_arm = data.iloc[idx]['upper_arm_joint_position']
  forearm = data.iloc[idx]['forearm_joint_position'] 
  wrist_1 = data.iloc[idx]['wrist_1_joint_position']
  wrist_2 = data.iloc[idx]['wrist_2_joint_position']
  wrist_3 = data.iloc[idx]['wrist_3_joint_position']
  return -1.0 * np.deg2rad(np.array([shoulder, upper_arm, forearm, wrist_1, wrist_2, wrist_3]))
#   return -1.0 * np.array([shoulder, upper_arm, forearm, wrist_1, wrist_2, wrist_3])

import time

def main():
    global sim, left_traj, right_traj, cnt, last_time
    # folder_path = "data/new_front/"
    # folder_path = "data/trajectories/right_five/"
    n = 3
    folder_path =f"data/new_data/right_five/demo_{n}/"
    # folder_path =f"data/new_data/right_five/processed_demos/right_five/demo_{n}/"

    # left_traj = pd.read_csv(folder_path + "left_joint_trajectory.csv")
    # right_traj = pd.read_csv(folder_path + "right_joint_trajectory.csv")
    left_traj = pd.read_csv(folder_path + f"left_joint_trajectory_demo_{n}.csv")
    right_traj = pd.read_csv(folder_path + f"right_joint_trajectory_demo_{n}.csv")
    columns = left_traj.columns.str.contains('position')

    # left_traj = np.deg2rad(left_traj.loc[:, columns].to_numpy())
    # right_traj = np.deg2rad(right_traj.loc[:, columns].to_numpy())
    
    left_traj = left_traj.loc[:, columns]
    right_traj = right_traj.loc[:, columns]

    cnt = 0

    last_time = None


    def loop_callback():
        global sim, left_traj, right_traj, cnt, last_time
        # sim.robot1_controls = -1.0 * left_traj[cnt, :]
        # sim.robot2_controls = -1.0 * right_traj[cnt, :]

        # Rate limiting based on dt
        current_time = time.time()
        if last_time is not None and (current_time - last_time) < 0.1:
            return

        last_time = current_time

        sim.robot1_controls = get_joint_angles(left_traj, cnt)
        sim.robot2_controls = get_joint_angles(right_traj, cnt)
        sim.d.ctrl[:] = np.concatenate([sim.robot2_controls, sim.robot1_controls])
        cnt += 1
        if cnt >= left_traj.shape[0]:
            cnt = 0
        pass

    sim = BimanualSim(loop_callback=loop_callback)

    sim.loop()


if __name__ == "__main__":
    main()