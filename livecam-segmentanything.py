import os
import cv2
import numpy as np
import time
from segment_anything import SamPredictor, sam_model_registry
import shutil

# Initialize global variables
selected_colors = []
clicked_points = []
frame = None
capture_count = 0
total_captures = 5  # Number of captures needed
capture_interval = 6  # Capture interval in seconds
frames = []
num_points_to_select = 8  # Number of points to select before starting capture

# Define the path to the checkpoint and the model type
checkpoint_path1 = "checkpoints/sam_checkpoint.pth"  # Path to the downloaded checkpoint
model_type1 = "default"  # Replace with the actual model type, e.g., "default"

checkpoint_path2 = "checkpoints/sam_vit_l.pth"  # Path to the downloaded checkpoint
model_type2 = "vit_l"  # Replace with the actual model type, e.g., "vit_l"

checkpoint_path3 = "checkpoints/sam_vit_b.pth"  # Path to the downloaded checkpoint
model_type3 = "vit_b"  # Replace with the actual model type, e.g., "vit_b"

# Load the model
sam = sam_model_registry[model_type2](checkpoint=checkpoint_path2)
predictor = SamPredictor(sam)

# Create the output directory
output_dir = "images/results/livecapture"
os.makedirs(output_dir, exist_ok=True)

def clear_output_directory(output_dir):
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def click_event(event, x, y, flags, param):
    global selected_colors, clicked_points, frame, num_points_to_select
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < num_points_to_select:
        selected_color = frame[y, x]
        selected_colors.append(selected_color)
        clicked_points.append((x, y))
        print(f"Selected color: {selected_color}")
        print(f"Points selected: {len(clicked_points)}/{num_points_to_select}")

def capture_frames():
    global capture_count, clicked_points, frame

    # Open the webcam
    cap = cv2.VideoCapture(1)
    cv2.namedWindow('Webcam Feed')
    cv2.setMouseCallback('Webcam Feed', click_event)

    # Wait until the required number of points are selected
    while len(clicked_points) < num_points_to_select:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the current frame
        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Starting capture...")

    while capture_count < total_captures:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the captured frame
        output_path = os.path.join(output_dir, f"capture_{capture_count + 1}.png")
        cv2.imwrite(output_path, frame)
        capture_count += 1
        print(f"Captured frame {capture_count}")
        time.sleep(capture_interval)

        # Display the current frame
        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

def find_best_match_coordinates(frame, selected_color):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    selected_color_hsv = cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)[0][0]

    lower_color = np.array([selected_color_hsv[0] - 5, selected_color_hsv[1] - 10, selected_color_hsv[2] - 10])
    upper_color = np.array([selected_color_hsv[0] + 5, selected_color_hsv[1] + 10, selected_color_hsv[2] + 10])

    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Find contours and select the largest one
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Calculate the centroid of the largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return cY, cX  # (y, x)
    return None

def draw_x_mark(frame, coordinates):
    color = (0, 0, 255)  # Red color for the X mark
    thickness = 2  # Thickness of the lines
    line_length = 10  # Length of the lines

    x, y = coordinates[1], coordinates[0]  # (x, y) format
    cv2.line(frame, (x - line_length, y - line_length), (x + line_length, y + line_length), color, thickness)
    cv2.line(frame, (x + line_length, y - line_length), (x - line_length, y + line_length), color, thickness)

def process_frame_with_sam(frame, coordinates_list):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Set the image in the predictor
    predictor.set_image(frame_rgb)
    
    # Prepare to combine multiple masks
    combined_mask = np.zeros_like(frame_rgb)

    for coordinates in coordinates_list:
        # Define the input points and labels correctly
        input_points = np.array([[coordinates[1], coordinates[0]]])  # (x, y) format
        input_labels = np.array([1])  # 1 for foreground

        # Generate the mask
        masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels)

        # Assuming the desired mask is the first one (mask 0)
        if masks is not None and len(masks) > 0:
            mask_np = masks[0].astype(np.uint8)

            # Create a color overlay where the mask is
            color_mask = np.zeros_like(frame_rgb)
            color_mask[mask_np != 0] = [0, 255, 0]  # Black color

            # Combine the color mask with the combined mask
            combined_mask = cv2.addWeighted(combined_mask, 1, color_mask, 1.0, 0)  # Increase alpha to make it stronger

        # Draw X mark on the original frame
        draw_x_mark(frame_rgb, coordinates)

    # Overlay the combined mask on the original frame
    overlay = cv2.addWeighted(frame_rgb, 1, combined_mask, 0.5, 0)  # Adjust alpha for stronger overlay

    return overlay

def process_saved_frames():
    global selected_colors

    for i in range(total_captures):
        start = time.time()
        # Read the saved frame
        frame_path = os.path.join(output_dir, f"capture_{i + 1}.png")
        frame = cv2.imread(frame_path)

        coordinates_list = []
        for color in selected_colors:
            coordinates = find_best_match_coordinates(frame, color)
            if coordinates:
                coordinates_list.append(coordinates)

        print(coordinates_list)
        if not coordinates_list:
            print(f"No coordinates found for frame {i + 1}")
            continue  # Skip processing if no coordinates are found

        # Process the frame with "Segment Anything"
        processed_frame = process_frame_with_sam(frame, coordinates_list)

        # Save the processed frame to the results folder
        processed_output_path = os.path.join(output_dir, f"processed_capture_{i + 1}.png")
        cv2.imwrite(processed_output_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        print(f"Processed frame saved as {processed_output_path}")
        print("Time taken: ", time.time() - start)

def main():
    # Clear the output directory
    clear_output_directory(output_dir)
    
    # Capture frames from the webcam
    capture_frames()
    print("Processing....")
    # Process the saved frames
    process_saved_frames()

if __name__ == "__main__":
    main()
