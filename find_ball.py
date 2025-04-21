import cv2
import numpy as np
import os
import math

TENNIS_BALL_RADIUS = 0.03175  # in meters (3.175cm radius, 6.35cm diameter)

# my iphone 12 mini camera parameters
CAMERA_MATRIX = np.array([
    [1495.45334, 0, 781.007707], 
    [0.0, 1495.45334, 1005.87303],
    [0.0, 0.0, 1.0]
])

VISUALIZE = True

def find_camera_frame_coords(center, radius_pixels):
    fx = CAMERA_MATRIX[0,0]
    fy = CAMERA_MATRIX[1,1]
    cx = CAMERA_MATRIX[0,2]
    cy = CAMERA_MATRIX[1,2]
    
    # Calculate distance (z) using apparent size
    z = (fy * TENNIS_BALL_RADIUS * 2) / radius_pixels  # diameter = 2*radius
    
    # Calculate x and y positions
    x = (center[0] - cx) * z / fx
    y = (center[1] - cy) * z / fy
    
    return (x, y, z)

def detect_dominant_tennis_ball(image_path):
    # Verify the file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found at path: {image_path}")
        print(f"Current working directory: {os.getcwd()}")
        return None, None
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from path: {image_path}")
        return None, None
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if VISUALIZE:
        cv2.imshow("hsv", hsv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Define range for tennis ball color (green/yellow)
    lower_green = np.array([30, 70, 80])  # Increased minimum saturation and value
    upper_green = np.array([85, 255, 255])
    
    # Create a mask for the tennis ball color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    if VISUALIZE:
        cv2.imshow("after kernel", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Find all white regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return original image
    if not contours:
        return image, []

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Minimum circularity threshold (0.7 is fairly round)
    min_circularity = 0.7
    
    # Variables to track best candidate
    best_circularity = 0
    best_contour = None
    best_hull = None
    
    # Evaluate all contours
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Skip small areas (adjust threshold as needed)
        if area < 200:
            continue
        
        # Get convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)
        
        # Calculate circularity of convex hull
        if hull_perimeter > 0:
            circularity = (4 * math.pi * hull_area) / (hull_perimeter ** 2)
        else:
            circularity = 0
        
        # Track best circularity
        if circularity > best_circularity:
            best_circularity = circularity
            best_contour = contour
            best_hull = hull
    
    # Process the best candidate if found
    if best_contour is not None and best_circularity > 0.9:  # Minimum circularity threshold
        # Get enclosing circle of the best hull
        (x, y), radius = cv2.minEnclosingCircle(best_hull)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Prepare output
        output = image.copy()
        
        # convex hull (blue)
        cv2.drawContours(output, [best_hull], -1, (255, 0, 0), 2)
        # enclosing circle (green)
        cv2.circle(output, center, radius, (0, 255, 0), 2)
        # center (red)
        cv2.circle(output, center, 5, (0, 0, 255), -1)
        
        print(f"Best circularity: {best_circularity:.2f}")
        return output, [(center, radius)]
    
    return image, []

if __name__ == "__main__":
    input_image = "./data/ball/IMG_9350.png"
    
    print(f"Attempting to load image from: {input_image}")
    print(f"Current working directory: {os.getcwd()}")
    
    # find the tennis ball in the image
    result, data = detect_dominant_tennis_ball(input_image)
    
    if result is not None and data:
        # print(f"Detected tennis ball at {data['center_pixels']}")
        # print(f"World coordinates (meters): {data['world_coords']}")
        
        if VISUALIZE:
            cv2.imshow("Tennis Ball Detection", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # cv2.imwrite("detected_balls.jpg", result) # if you want to save an image of the balls
    else:
        print("No tennis ball detected or processing failed.")

    # Find the camera frame coordinates of the ball center