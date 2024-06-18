import os
import cv2
import numpy as np
import time
from segment_anything import SamPredictor, sam_model_registry
import shutil

# Initialize global variables
selected_color = None
clicked = False
frame = None
capture_count = 0
total_captures = 3  # Number of captures needed
capture_interval = 6  # Capture interval in seconds
frames = []

# Define the path to the checkpoint and the model type
checkpoint_path = "checkpoints/sam_checkpoint.pth"  # Path to the downloaded checkpoint
model_type = "default"  # Replace with the actual model type, e.g., "default"

# Load the model
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
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
    global selected_color, clicked, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_color = frame[y, x]
        clicked = True
        print(f"Selected color: {selected_color}")

def capture_frames():
    global capture_count, clicked, frame

    # Open the webcam
    cap = cv2.VideoCapture(1)
    cv2.namedWindow('Webcam Feed')
    cv2.setMouseCallback('Webcam Feed', click_event)

    while capture_count < total_captures:
        ret, frame = cap.read()
        if not ret:
            break

        if clicked and capture_count < total_captures:
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

def find_color_coordinates(frame, selected_color):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    selected_color_hsv = cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)[0][0]

    lower_color = np.array([selected_color_hsv[0] - 10, 0, 0])
    upper_color = np.array([selected_color_hsv[0] + 10, 255, 255])

    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    coordinates = cv2.findNonZero(mask)
    if coordinates is not None:
        mean_coordinates = np.mean(coordinates, axis=0)[0]
        return int(mean_coordinates[1]), int(mean_coordinates[0])  # (y, x)
    return None

def process_frame_with_sam(frame, coordinates):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Set the image in the predictor
    predictor.set_image(frame_rgb)

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
        color_mask[mask_np != 0] = [139, 0, 0]  # Green color

        # Overlay the color mask on the original frame
        overlay = cv2.addWeighted(frame_rgb, 1, color_mask, 0.5, 0)

        return overlay
    return frame  # Return the original frame if no mask is found

def process_saved_frames():
    global selected_color

    for i in range(total_captures):
        start = time.time()
        # Read the saved frame
        frame_path = os.path.join(output_dir, f"capture_{i + 1}.png")
        frame = cv2.imread(frame_path)
        coordinates = find_color_coordinates(frame, selected_color)

        if coordinates is None:
            print(f"No coordinates found for frame {i + 1}")
            continue  # Skip processing if no coordinates are found

        # Process the frame with "Segment Anything"
        processed_frame = process_frame_with_sam(frame, coordinates)

        # Save the processed frame to the results folder
        processed_output_path = os.path.join(output_dir, f"processed_capture_{i + 1}.png")
        cv2.imwrite(processed_output_path, cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        print("Coordinates", coordinates)
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
