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
        self.robot1_controls = np.zeros(6)

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
        self.d.ctrl[:] = self.robot1_controls

    def set_robot_control(self, robot, controls):
        if robot == 1:
            self.robot1_controls = controls

    def set_robot_control_by_ee(self, robot, ee_pos, ee_quat, make_relative=False):
        ur5e_arm = ur_kinematics.URKinematics('ur5e')
        ee_trans = position_quaternion_to_transform(ee_pos, ee_quat)

        if make_relative:
            base_trans = self.get_base_transform(robot)
            ee_trans = transform_relative(base_trans, ee_trans)

        joint_configs = ur5e_arm.inverse(ee_trans[:-1,:], False)

        if robot == 1:
            self.robot1_controls[:-1] = joint_configs

    def set_gripper(self, robot, ee_angle):
        if robot == 1:
            self.robot1_controls[-1] = ee_angle

    def get_base_transform(self, robot):
        if robot == 1:
            body = self.m.body('arm1_base')
        return position_quaternion_to_transform(body.pos, body.quat)

def main():
    sim = UR5eSim()
    sim.loop()

if __name__ == "__main__":
    main()
