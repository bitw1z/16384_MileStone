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
        learning_rate = 0.05
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
    from libcapstone import LightsaberDetector 

    robot_computator = Robot_Computator() 
    detector = LightsaberDetector()
    
    with Robot() as robot:
        robot.register_visualizer(detector)



        """
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
        
        # STEP 1 of MILESTONE 2 
        robot.set_joint_targets([0, 0, 0, 0, 0, 0, 0]) # home pose
        robot.grip(1.0) # open the gripper
        desired_ee_pose = np.array([[0, 0, 1, 0.6], 
                                    [0, -1, 0, 0], 
                                    [1, 0, 0, 0.3], 
                                    [0, 0, 0, 1]])
        q_current = robot.joint_values
        q = robot_computator.inverse_kinematics(q_current, desired_ee_pose)
        robot.set_joint_targets(q)

        desired_ee_pose = np.array([[0, 1, 0, 0.4], 
                                    [0, 0, 1, 0.2], 
                                    [1, 0, 0, 0.25], 
                                    [0, 0, 0, 1]])
        q_current = robot.joint_values
        q = robot_computator.inverse_kinematics(q_current, desired_ee_pose)
        robot.set_joint_targets(q)
        
        robot.grip(1.0)
        robot.set_joint_targets([0, 0, 0, 0, 0, 0, 0])

        saved_pose_q = np.array([2.34500e-03, 3.54066e-01, 9.83950e-02, 3.34839e-01, 2.88000e-04, 5.16220e-02, 8.15370e-02])
        robot.set_joint_targets(saved_pose_q)
        robot.grip(0.25, speed=0.2)
        

        robot.grip(1)
        robot.set_joint_targets([0, 0, 0, 0, 0, 0, 0])

        approach_pose_q = np.array([[-0.027909,  0.423891, -0.425588,  0.714793, -0.147502,  0.30308,  -0.292931]])
        robot.set_joint_targets(approach_pose_q)

        saved_pose_q = np.array([-0.249469,  0.57446,  -0.203454,  0.837529,  0.2013, 0.319728, -0.646878])
        robot.set_joint_targets(saved_pose_q)
        robot.grip(0.25, speed=0.2)

        clear_q = np.array([[-0.248467, 0.182317, -0.285644,  0.802882, 0.189809, 0.601524, -0.687375]])
        robot.set_joint_targets(clear_q)

        def select_farthest_point(robot, robot_computator):
            state = detector.query("green")
            while state is None:
                state = detector.query("green")
        
            x1, x2 = state[0], state[1]
            print(x1, x2)

            current_q = robot.joint_values
            robot_computator.forward_kinematics(current_q)
            ee_position = robot_computator.transform[-1][:3, 3]

            x1_dist = np.linalg.norm((x1 - ee_position))
            x2_dist = np.linalg.norm((x2 - ee_position))

            if x1_dist > x2_dist: 
                return x1
            else: 
                return x2 

        tip = select_farthest_point(robot, robot_computator)
        x, y, z = tip
        current_q = robot.joint_values
        robot_computator.forward_kinematics(current_q)
        
        H_base_tip = np.array([[0, 0, 1, x], 
                             [0, -1, 0, y],
                             [1, 0, 0, z],
                             [0, 0, 0, 1]]) 

        ee_rot = robot_computator.transform[-1][:3, :3]
        ee_pos = robot_computator.transform[-1][:3, 3]

        H_base_ee_inv = np.linalg.inv(robot_computator.transform[-1])
        H_ee_tip = H_base_ee_inv @ H_base_tip
        
        
        # horizontal move
        desired_tip = np.array([[0, 1, 0, x],
                                [0, 0, 1, y + 0.3],
                                [1 ,0, 0, z],
                                [0, 0, 0, 1]])
        desired_ee_pose = desired_tip @ np.linalg.inv(H_ee_tip)

        current_q = robot.joint_values
        desired_q = robot_computator.inverse_kinematics(current_q, desired_ee_pose)
        desired_pose = robot_computator.forward_kinematics(desired_q)
        
        robot.set_joint_targets(desired_q)

        # vertical move
        desired_tip = np.array([[1, 0, 0, x - 0.5],
                                [0, 1, 0, y + 0.3],
                                [0 ,0, 1, z + 0.5],
                                [0, 0, 0, 1]])

        desired_ee_pose = desired_tip @ np.linalg.inv(H_ee_tip)

        current_q = robot.joint_values
        desired_q = robot_computator.inverse_kinematics(current_q, desired_ee_pose)
        desired_pose = robot_computator.forward_kinematics(desired_q)
        
        robot.set_joint_targets(desired_q)
        """

        # step 3 of MILESTONE 2 
        import time
        start = time.time()
        points = []
        while True:
            state = detector.query("green")
            if state is not None: 
                print("found point")
                x1, x2 = state[0], state[1]
                points.append(x1)
                points.append(x2)

            if len(points) > 1000: 
                break

            end = time.time()
            if int(end - start) > 20: # sweep 2 seconds 
                break 
        
        print("done", len(points))
        points = np.array(points)

        centroid = np.mean(points, axis=0)
        points = points - centroid 

        __, singular_values, vh = np.linalg.svd(points)
        normal = vh[-1]

        d = -np.dot(normal, centroid)

        a, b, c = normal 
        while True:
            detector.draw_plane(np.array([a, b, c, d]))

if __name__ == "__main__":
    main()

