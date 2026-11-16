import numpy as np 
from xarm7_lib import Robot 
from libcapstone import LightsaberDetector

class Robot_Computator():
    def __init__(self): 
        self.x_arm_kinematic_params = [(0, 0, 267, 0), 
                                       (0, -np.pi/2, 0, 0), 
                                       (0, np.pi/2, 293, 0), 
                                       (52.5, np.pi/2, 0, 0),
                                       (77.5, np.pi/2, 342.5, 0), 
                                       (0, np.pi/2, 0, 0), 
                                       (76, -np.pi/2, 269, 0)] 
        self.DoF = 7  
    
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
        result = np.zeros((3, self.DoF))
        axis_local = np.array([0, 0, 1])
        
        for i in range(self.DoF):
            current_pose = self.transform[i][:3, 3]
            r_vec = ee_pose - current_pose 
            rotation = self.transform[i][:3, :3]
            
            w = rotation @ axis_local
            ang_vec = np.cross(w, r_vec) 
            
            result[:, i] = ang_vec 
            
        return result
    
def main(): 
    from xarm7_lib import Robot
    from libcapstone import LightsaberDetector

    robot_computator = Robot_Computator() 
    
    with Robot() as robot: 
        robot.free_drive()
        input('Enter to stop\n')
        q_current = np.asarray(robot.joint_values(), dtype=float)
        print("q_current =", q_current)
        ee_pose = Robot_Computator.foward_kinematics(q_current)
        print("ee_pose =", ee_pose)
        robot.end_free_drive()
    
    """
    from xarm7_lib import Robot
    from libcapstone import LightsaberDetector
    
    detector = LightsaberDetector()
    robot_computator = Robot_Computator() 
    
    with Robot() as robot:
        robot.register_visualizer(detector)

        num_steps = 100 
        dt = 1.0 / num_steps 
        
        for _ in range(num_steps):
            delta_x = np.array([0.01, 0, 0]) / num_steps
            x_dot = delta_x / dt 
            
            q_current = np.asarray(robot.joint_values(), dtype=float)
            q_dot = np.linalg.pinv(robot_computator.jacobian(q_current)) @ x_dot
            
            q_next = q_current + q_dot * dt 
            
            robot.set_joint_targets(q_next)
        
        robot.stop()
    """
    
if __name__ == "__main__":
    main()
        
        
    

    

