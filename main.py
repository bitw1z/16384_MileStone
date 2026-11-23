import numpy as np 
from xarm7_lib import Robot 
from scipy.spatial.transform import Rotation 

class Robot_Computator():
    def __init__(self): 
        self.x_arm_kinematic_params = [(0, 0, 0.267, 0), 
                                       (0, -np.pi/2, 0, 0), 
                                       (0, np.pi/2, 0.293, 0), 
                                       (0.0525, np.pi/2, 0, 0),
                                       (0.0775, np.pi/2, 0.3425, 0), 
                                       (0, np.pi/2, 0, 0), 
                                       (0.076, -np.pi/2, 0.269, 0)] 
        self.DoF = 7
        self.transform = []
    
    def _rotation_matrix_x(self, theta):
        return np.array([[1, 0, 0, 0], 
                         [0, np.cos(theta), -np.sin(theta), 0], 
                         [0, np.sin(theta), np.cos(theta), 0], 
                         [0, 0, 0, 1]])
        
    def _rotation_matrix_z(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0, 0], 
                         [np.sin(theta), np.cos(theta), 0, 0], 
                         [0, 0, 1, 0], 
                         [0, 0, 0, 1]])
        
    def _translation_matrix_x(self, d): 
        return np.array([[1, 0, 0, d], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0], 
                         [0, 0, 0, 1]])
        
    def _translation_matrix_z(self, d): 
        return np.array([[1, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, d], 
                         [0, 0, 0, 1]])
        
    def _modified_dh_params(self, a, alpha, d, theta): 
        rot_x = self._rotation_matrix_x(alpha)
        trans_x = self._translation_matrix_x(a) 
        rot_z = self._rotation_matrix_z(theta)
        trans_z = self._translation_matrix_z(d) 
        return rot_x @ trans_x @ rot_z @ trans_z 
        
    def forward_kinematics(self, configuration):
        self.transform = []
        a, alpha, d, _ = self.x_arm_kinematic_params[0]
        result = self._modified_dh_params(a, alpha, d, configuration[0]) # T_0_1 
        self.transform.append(result)
        
        for i in range(1, self.DoF): 
            a, alpha, d, _ = self.x_arm_kinematic_params[i]
            next = self._modified_dh_params(a, alpha, d, configuration[i]) # T_i_i+1 
            result = result @ next
            self.transform.append(result)
        
        return result

    def jacobian(self, configuration):
        self.forward_kinematics(configuration) 
        
        ee_pose = self.transform[-1][:3, 3]
        result = np.zeros((6, self.DoF))
        axis_local = np.array([0, 0, 1])
        
        for i in range(self.DoF):
            current_pose = self.transform[i][:3, 3]
            r_vec = ee_pose - current_pose 
            rotation = self.transform[i][:3, :3]
            
            angular_base = rotation @ axis_local
            linear_vec = np.cross(angular_base, r_vec) 
            
            result[:3, i] = linear_vec 
            result[3:, i] = angular_base
            
        return result # 6 by 7 jacobian matrix

    def inverse_kinematics(self, q_current, desired_ee_pose): 
        iterations = 10000
        learning_rate = 0.5
        tolerance = 1e-5
        
        desired_pos = desired_ee_pose[:3, 3] 
        desired_rot = desired_ee_pose[:3, :3]
        
        for _ in range(iterations): 
            current_pose = self.forward_kinematics(q_current) # 4 by 4 matrix 
            curr_pos = current_pose[:3, 3]
            curr_rot = current_pose[:3, :3]
            
            pos_error = curr_pos - desired_pos 
            rot_diff = curr_rot @ desired_rot.T
            rot_error = Rotation.from_matrix(rot_diff).as_rotvec()
            
            error = np.concatenate((pos_error, rot_error))
            print(np.linalg.norm(error))

            if np.linalg.norm(error) < tolerance:
                return q_current
            
            gradient = self.jacobian(q_current).T @ error
            q_current = q_current - learning_rate * gradient
        
        return q_current
            
def main(): 
    from xarm7_lib import Robot

    robot_computator = Robot_Computator() 
    
    """
    with Robot() as robot:
        # STEP 1 OF MILESTONE 1
        
        robot.set_joint_targets([0, 0, 0, 0, 0, 0, 0]) # home pose
        q_current = robot.joint_values
        start_pose = robot_computator.forward_kinematics(q_current)[:3, 3]
        print("initial pose = ", start_pose)
        
        num_steps = 10
        dt = 1.0 / num_steps
        for i in range(3):
            for _ in range(num_steps):
                desired_motion = np.zeros(3)
                desired_motion[i] = 100

                delta_x = desired_motion / num_steps
                x_dot = delta_x / dt
                q_current = np.asarray(robot.joint_values, dtype=float)
                q_dot = np.linalg.pinv(robot_computator.jacobian(q_current)) @ x_dot
                q_next = q_current + q_dot * dt
                robot.set_joint_targets(q_next)

            q_current = robot.joint_values
            end_pose = robot_computator.forward_kinematics(q_current)[:3, 3] 
            print(f"{i}end pose = ", end_pose)
            print(f"{i}th delta = ", end_pose - start_pose)
        
        # STEP 2 OF MILESTONE 1 
        robot.grip(1.0)
        robot.free_drive()
        input("Enter to stop\n")
        robot.end_free_drive()

        saved_pose_q = np.array([0.027495, 0.226465, 0.070983, 0.349805, 0.038457, 0.141063, 0.033092])
        robot.set_joint_targets(saved_pose_q)

        num_steps = 10
        dt = 1.0 / num_steps

        for _ in range(num_steps):
            delta_x = np.array([0, 0, -35]) / num_steps
            x_dot = delta_x / dt
            q_current = np.asarray(robot.joint_values, dtype=float)
            q_dot = np.linalg.pinv(robot_computator.jacobian(q_current)) @ x_dot
            q_next = q_current + q_dot * dt
            robot.set_joint_targets(q_next)

        robot.grip(0.27, speed=0.2)

        for _ in range(num_steps):
            delta_x = np.array([0, 0, 35]) / num_steps
            x_dot = delta_x / dt
            q_current = np.asarray(robot.joint_values, dtype=float)
            q_dot = np.linalg.pinv(robot_computator.jacobian(q_current)) @ x_dot
            q_next = q_current + q_dot * dt
            robot.set_joint_targets(q_next)

        robot.stop()
        """
        
    # STEP 1 of MILESTONE 2 
    desired_ee_pose = np.array([[1, 0, 0, 0.5], 
                                [0, -1, 0, 0], 
                                [0, 0, -1, 0], 
                                [0, 0, 0, 1]])
    q_current = np.array([0, 0, 0, 0, 0, 0, 0])
    
    q = robot_computator.inverse_kinematics(q_current, desired_ee_pose)
    print(q)
    ee_pose = robot_computator.forward_kinematics(q)
    print(ee_pose)
    

           
if __name__ == "__main__":
    main()
    