
import cv2
import numpy as np


def calibrate_camera():
    # Define the criteria and object points for calibration
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Read images and find chessboard corners
    for i in range(9):  # Example for 20 images
        img = cv2.imread(f'cal/cal{i+1}.jpeg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Perform the calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Intrinsic matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    
    return mtx, 


def calculate_height(heights_in_pixels, foot_locations, plane_equation, camera_parameters):
    # Unpack camera parameters
    intrinsic_matrix = camera_parameters['intrinsic_matrix']
    rotation_matrix = camera_parameters['rotation_matrix']
    translation_vector = camera_parameters['translation_vector']

    # Unpack the plane equation
    a, b, c, d = plane_equation

    # Initialize lists to store calculated heights
    calculated_heights = []

    # Loop over each image
    for height_in_pixels, foot_location in zip(heights_in_pixels, foot_locations):
        # Convert foot location to homogeneous coordinates
        foot_homogeneous = np.array([foot_location[0], foot_location[1], 1])

        # Backproject the foot location to a 3D ray
        ray_direction = np.linalg.inv(intrinsic_matrix).dot(foot_homogeneous)
        ray_direction = rotation_matrix.T.dot(ray_direction)

        # Compute the intersection of the ray with the plane
        t_part_1 = -(translation_vector.T.dot(np.array([a, b, c])) + d)
        t_part_2 = (ray_direction.T.dot(np.array([a, b, c])))
        t =  t_part_1/t_part_2 
        foot_3d = translation_vector + t * ray_direction

        # Estimate the head location in 3D by adding the height in pixels
        head_location_2d = np.array([foot_location[0], foot_location[1] - height_in_pixels, 1])
        head_direction = np.linalg.pinv(intrinsic_matrix).dot(head_location_2d)
        head_direction = rotation_matrix.T.dot(head_direction)
        head_3d = translation_vector + t * head_direction

        # Calculate the real-world height
        height_3d = np.linalg.norm(head_3d - foot_3d)
        calculated_heights.append(height_3d)

    # Average the calculated heights from all images
    average_height = np.mean(calculated_heights)
    
    return average_height




if __name__ == "__main__":
    camera_parameters = {
        'intrinsic_matrix': np.array([[3200, 0, 1600],
                                        [0, 3200, 1200],
                                        [0,  0,  1]]),
        'rotation_matrix': np.eye(3),  # Assuming no rotation for simplicity
        'translation_vector': np.array([0, 0, 1.5])  # Camera at (0,0,1.5) height, adjust as needed
    }
    
    
    # calibrate_camera()
    height = calculate_height(
        [100, 150, 120], 
        [(100, 100), (200, 200), (300, 300)], 
        [1, 1, 1, 1], 
        camera_parameters)
    
    print(height)