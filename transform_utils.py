from scipy.spatial.transform import Rotation as R
import numpy as np

def position_quaternion_to_transform(pos, quat):
  # Convert position and quaternion to a 4x4 rigid transformation matrix
  transform = np.eye(4)
  transform[:3, 3] = pos
  transform[:3, :3] = quaternion_to_rotation_matrix(quat)
  return transform

def quaternion_to_rotation_matrix(quat):
  # Convert quaternion to a 3x3 rotation matrix
  rotation_matrix = R.from_quat(quat).as_matrix()
  return rotation_matrix

def inverse_transform(transform):
  # Compute the inverse of a 4x4 rigid transformation matrix
  return np.linalg.inv(transform)

def transform_relative(transform1, transform2):
  # Compute the relative transformation between two 4x4 rigid transformation matrices
  # transform1: 4x4 rigid transformation matrix of the first frame
  # transform2: 4x4 rigid transformation matrix of the second frame
  # Returns: 4x4 rigid transformation matrix of the second frame relative to the first frame
  return np.dot(inverse_transform(transform1), transform2)
