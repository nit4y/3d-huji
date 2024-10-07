import math

from scipy.optimize import least_squares, minimize
import cv2
import numpy as np
import os
import sympy as sp


def distance_between_rays(t, rays, distances):
    t1, t2, t3, t4 = t
    # t1, t2, t3 = t
    points = [rays[0] * t1, rays[1] * t2, rays[2] * t3, rays[3] * t4]
    for i in range(len(points)):
        points[i][2] *= 100
    # points = [rays[0] * t1, rays[1] * t2, rays[2] * t3]

    # Calculate distances between adjacent points
    equations_system = [
        np.linalg.norm(points[0] - points[1]) - distances[0],
        np.linalg.norm(points[1] - points[2]) - distances[1],
        np.linalg.norm(points[2] - points[3]) - distances[2],
        np.linalg.norm(points[3] - points[0]) - distances[3]  # Connect the last point to the first if needed
    ]
    # equatio = [
    #     np.linalg.norm(points[0] - points[1]) - distances[0],
    #     np.linalg.norm(points[1] - points[2]) - distances[1],
    #     np.linalg.norm(points[2] - points[0]) - distances[2]# Connect the last point to the first if needed
    # ]
    return equations_system


def find_plane(p1, p2, p3):
    # Convert points to NumPy arrays for easier calculations
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Create vectors
    v1 = p2 - p1
    v2 = p3 - p1

    # Compute the normal vector using the cross product
    normal = np.cross(v1, v2)

    # Extract A, B, C from the normal vector
    A, B, C = normal

    # Calculate D using point P1
    D = -np.dot(normal, p1)

    return A, B, C, D


def save_image(im, save_path):
    cv2.imwrite(save_path, im)


def real_cube_root(x, exp):
    if x < 0:
        return -pow(-x, exp)  # Handle negative numbers
    else:
        return pow(x, exp)

def calibrate_camera():
    # Define the criteria and object points for calibration
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    BOARD_DIMS = (3,3)

    objp = np.zeros((BOARD_DIMS[0]*BOARD_DIMS[1], BOARD_DIMS[0]), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_DIMS[0], 0:BOARD_DIMS[1]].T.reshape(-1, 2)

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Read images and find chessboard corners

    # Define the directory path
    directory_path = './cal'

    # Loop over all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the filename starts with 'Whatsapp'
        if filename.startswith('WhatsApp'):
            img = cv2.imread(directory_path + '/' + filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, BOARD_DIMS, None)

            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, BOARD_DIMS, corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
    # for i in range(9):  # Example for 20 images
    #     img = cv2.imread(f'cal/cal{i+1}.jpeg')
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    #     ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    #
    #     if ret == True:
    #         objpoints.append(objp)
    #         imgpoints.append(corners)
    #
    #         # Draw and display the corners
    #         img = cv2.drawChessboardCorners(img, (7,6), corners, ret)
    #         cv2.imshow('img', img)
    #         cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Perform the calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Intrinsic matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    
    return mtx


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
        t = t_part_1/ t_part_2
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


def draw_ray_on_image(image, ray_start, ray_direction, save_path):
    """
    Draws a ray on an image based on the given starting point and direction vector.

    :param image: The image (numpy array) on which the ray will be drawn.
    :param ray_start: The starting point of the ray (tuple of two coordinates, x and y).
    :param ray_direction: The direction of the ray (tuple of two values, dx and dy).
    :return: The image with the ray drawn on it.
    """

    # Define image dimensions
    height, width, _ = image.shape

    # Normalize the ray direction to extend it further
    norm_ray_dir = np.array(ray_direction) / np.linalg.norm(ray_direction)

    # Define the end point of the ray
    ray_end = np.array(ray_start) + norm_ray_dir * 1000  # Extend the ray by 1000 units

    # Clip the ray's end point to the image boundaries
    ray_end = (int(np.clip(ray_end[0], 0, width - 1)), int(np.clip(ray_end[1], 0, height - 1)))

    # Draw the ray on the image (from start to clipped end point)
    ray_color = (0, 255, 0)  # Green color for the ray
    ray_thickness = 2
    cv2.line(image, ray_start, ray_end, ray_color, ray_thickness)

    # Save the modified image

    return image


def get_image_center(image):
    """
    Returns the center (x, y) coordinates of the given image.

    :param image: The input image (numpy array).
    :return: A tuple (x_center, y_center) representing the center of the image.
    """
    # Get the dimensions of the image
    height, width, _ = image.shape

    # Calculate the center coordinates
    x_center = width // 2
    y_center = height // 2

    return (x_center, y_center)


def record_height_in_picture(image, foot_locations, plane_equation, camera_params):
    im = cv2.imread(image)
    intrinsic_matrix = camera_params['intrinsic_matrix']
    rotation_matrix = camera_params['rotation_matrix']
    translation_vector = camera_params['translation_vector']
    # Unpack the plane equation
    a, b, c, d = plane_equation
    # Initialize lists to store calculated heights
    for i in range(len(foot_locations)):
        foot_homogeneous = np.array([foot_locations[i][0], foot_locations[i][1], 1])

        # Backproject the foot location to a 3D ray
        ray_direction = np.linalg.inv(intrinsic_matrix).dot(foot_homogeneous)
        ray_direction = rotation_matrix.T.dot(ray_direction)
        ray_direction = np.array([ray_direction[0], ray_direction[1]])
        im_center = intrinsic_matrix[:, 2][:2].astype(int)
        im = draw_ray_on_image(im, im_center, ray_direction, "marked_image.png")
    save_image(im, "marked_image.png")


def solve_rays(rays, distances, initial_guess=[1, 1, 1, 1]):
    # Initial guess for t values
    # t_initial = [1, 1, 1, 1]
    t_initial = initial_guess

    # Solve the system of equations using least squares
    result = least_squares(distance_between_rays, t_initial, args=(rays, distances),
                           bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
                           max_nfev=10000)

    # Return the solution for t1, t2, t3, t4
    return result.x


def find_plane_equation(paper_corners, camera_params):
    """
    This function receives an array of coordinates of paper corners and the camera parameters and returns the equation
    of the plane the paper was in relative to the camera coordinate system
    :param paper_corners: an array of size 4 with each element being a homogeneous coordinate of a corner
    the first and second corner form the bigger length, while the second and third form the other length
    :param camera_params:
    A dictionary holding intrinsic parameters position and rotation
    :return:
    the equation of the plane in the form of tuple of size 4
    """
    # extract camera parameters
    intrinsic_matrix = camera_params['intrinsic_matrix']
    rotation_matrix = camera_params['rotation_matrix']
    translation_vector = camera_params['translation_vector']

    # send rays to each of the corners
    paper_rays = []
    for i in range(len(paper_corners)):
        ray_direction = np.linalg.inv(intrinsic_matrix).dot(paper_corners[i])
        ray_direction = rotation_matrix.T.dot(ray_direction)
        paper_rays.append(ray_direction)
    paper_rays_normalized = paper_rays / np.linalg.norm(paper_rays, axis=1, keepdims=True)
    # distances = [0.297, 0.210, 0.297, 0.210]
    distances = [297, 210, 297, 210]
    # distances.append(math.sqrt(distances[0]**2 + distances[1]**2))
    min_solutions = None
    solutions = None
    min_distance = None
    for i in range(10):
        for j in range(10):
            for k in range(10):
                for l in range(10):
                    solutions = solve_rays(paper_rays, distances, [10 * i, 10 * j, 10 * k, 10 * l])
                    distance = distance_between_rays(solutions, paper_rays, distances)
                    if min_solutions is None:
                        min_solutions = solutions
                        min_distance = distance
                    elif np.linalg.norm(distance) < np.linalg.norm(min_distance):
                        min_distance = distance
                        min_solutions = solutions
    # solutions = solve_rays(paper_rays, distances)
    print(min_solutions)
    dist = distance_between_rays(min_solutions, paper_rays, distances)
    print("final distance:")
    print(dist)
    # for i in range(len(paper_rays)):
    #     print(f'paper corner {i + 1}:')
    #     print(solutions[i] * paper_rays[i])
    for i in range(len(paper_rays)):
        print(f'paper corner {i + 1}:')
        print(min_solutions[i] * paper_rays[i])
    points = []
    for i in range(len(solutions)):
        points.append((min_solutions[i] * paper_rays[i]))
    plane = find_plane((points[0]), (points[1]), (points[2]))
    print("dist from plane:")
    print(np.array(plane[:3]).T.dot(solutions[3] * paper_rays[3]) + plane[3])
    return plane












if __name__ == "__main__":
    camera_parameters = {
        'intrinsic_matrix': np.array([[1920, 5000, 1600],
                                        [1697, 4035, 1200],
                                        [300,  5000,  1]]),
        'rotation_matrix': np.eye(3),  # Assuming no rotation for simplicity
        'translation_vector': np.array([0, 0, 1.7])  # Camera at (0,0,1.5) height, adjust as needed
    }

    camera_parameters['intrinsic_matrix'] = calibrate_camera()

    # convert pixels focal length to meters
    camera_parameters['intrinsic_matrix'][0][0] = 5.91
    camera_parameters['intrinsic_matrix'][1][1] = 5.91

    # calibrate_camera()
    # height = calculate_height(
    #     [450, 436, 244, 204],
    #     [(270, 964), (408, 844), (444, 768), (494, 744)],
    #     [-0.89 * 0.577, 2.43, 0.45 * 0.577, 6.6],
    #     camera_parameters)
    paper_x0_y0_zD = [(164.0, 920.0, 1.0), (216.0, 186.0, 1.0), (730.0, 202.0, 1.0), (715.0, 930.0, 1.0)]
    paper_x0_yD_z0 = [(182.0, 1441.0, 1.0), (383.0, 1048.0, 1.0), (809.0, 1054.0, 1.0), (1115.0, 1453.0, 1.0)]
    paper_xD_y0_z0 = [(612.0, 1086.0, 1.0), (616.0, 462.0, 1.0), (707.0, 233.0, 1.0), (689.0, 1283.0, 1.0)]
    plane_1 = find_plane_equation(paper_x0_y0_zD, camera_parameters)
    # plane_2d = [0, 0, 1, 3]
    record_height_in_picture("./Paper/paper_x0_y0_zD.jpeg", paper_x0_y0_zD, plane_1, camera_parameters)
    plane_2 = find_plane_equation(paper_x0_yD_z0, camera_parameters)
    plane_3 = find_plane_equation(paper_xD_y0_z0, camera_parameters)
    print("plane z:")
    print(plane_1)
    print("plane y:")
    print(plane_2)
    print("plane x:")
    print(plane_3)

    # record_height_in_picture("blah_blah.jpeg", (270, 964), [-0.89 * 0.577, 2.43, 0.45 * 0.577, 6.6], camera_parameters)

    height = calculate_height(
        [807, 539, 424, 361],
        [(270, 964), (408, 844), (444, 768), (494, 744)],
        [-0.89 * 0.577, 2.43, 0.45 * 0.577, 6.6],
        camera_parameters)
    
    print(height)
