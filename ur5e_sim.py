import mujoco
from mujoco import viewer
import numpy as np
from ur_ikfast import ur_kinematics
from utils.transform_utils import position_quaternion_to_transform, transform_relative
from ur_ikfast import ur_kinematics

class UR5eSim:
    def __init__(self, model_path="./mujoco_menagerie/universal_robots_ur5e/scene.xml", loop_callback=None):
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        self.loop_callback = loop_callback if loop_callback is not None else self.default_loop_callback
        self.paused = False
        self.robot_controls = np.zeros(6)

    def loop(self):
        def key_callback(keycode, paused):
            if chr(keycode) == ' ':
                paused = not paused
                
        with viewer.launch_passive(self.m, self.d, key_callback=lambda x: key_callback(x, self.paused)) as v:
            while v.is_running():
                if not self.paused:
                    self.loop_callback()
                    mujoco.mj_step(self.m, self.d)
                    v.sync()

    def default_loop_callback(self):
        self.d.ctrl[:] = self.robot_controls

    def set_robot_control(self, robot, controls):
        self.robot_controls = controls

    def set_robot_control_by_ee(self, robot, ee_pos, ee_quat, make_relative=False):
        ur5e_arm = ur_kinematics.URKinematics('ur5e')
        ee_trans = position_quaternion_to_transform(ee_pos, ee_quat)

        if make_relative:
            base_trans = self.get_base_transform(robot)
            ee_trans = transform_relative(base_trans, ee_trans)

        joint_configs = ur5e_arm.inverse(ee_trans[:-1,:], False)

        self.robot_controls = joint_configs

    def set_gripper(self, robot, ee_angle):
        self.robot_controls[-1] = ee_angle

    def get_base_transform(self, robot):
        body = self.m.body('base')
        return position_quaternion_to_transform(body.pos, body.quat)

def main():
    sim = UR5eSim()
    robot1_base = sim.get_base_transform(1)
    # Define the desired EE transforms for each robot.
    robot1_pos = np.array([-2.87935048e-01, -1.40407637e-01,  3.06456357e-01])
    robot1_quat = np.array([6.92255947e-01, 7.21356638e-01, -4.29355962e-04, -2.06427328e-02])
    sim.set_robot_control_by_ee(1, robot1_pos, robot1_quat, make_relative=True)

    sim.loop()

if __name__ == "__main__":
    main()
