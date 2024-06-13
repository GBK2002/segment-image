import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

# Define the path to the checkpoint and the model type
checkpoint_path = "checkpoints/sam_checkpoint.pth"  # Path to the downloaded checkpoint
model_type = "default"  # Replace with the actual model type, e.g., "default"

# Load the model
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
predictor = SamPredictor(sam)

def process_frame(frame):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Set the image in the predictor
    predictor.set_image(frame_rgb)

    # Convert the frame to the HSV color space
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Adjust the color range for the pink sticky note
    lower_pink = np.array([176, 130, 220])  # New provided range
    upper_pink = np.array([176, 150, 260])

    # Create a mask for the pink sticky note
    mask = cv2.inRange(hsv_image, lower_pink, upper_pink)

    # Find contours of the object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get a point near the center of the largest contour assuming it's the object
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Sample points from within the largest contour
        sample_points = largest_contour[:, 0, :]

        # Select a point near the center of the contour by averaging
        cX = np.mean(sample_points[:, 0])
        cY = np.mean(sample_points[:, 1])
        foreground_point = (int(cX), int(cY))
    else:
        foreground_point = (0, 0)  # Default to (0, 0) if no contour is found

    # Combine points and labels
    input_points = np.array([foreground_point])
    input_labels = np.array([1])  # 1 for foreground

    # Generate masks
    masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels)
    
    if masks is None or len(masks) == 0:
        return frame

    # Define a function to overlay a colored mask on the frame
    def overlay_mask(image, mask, color, alpha=0.5):
        """Overlay a colored mask on the image."""
        colored_image = image.copy()
        for i in range(3):  # Apply color to each channel
            colored_image[:, :, i] = np.where(mask == 1, color[i], image[:, :, i])
        return cv2.addWeighted(colored_image, alpha, image, 1 - alpha, 0)

    # Choose a color for the mask (e.g., dark blue)
    mask_color = [0, 0, 139]  # Dark blue color

    # Assuming the desired mask is the second one (mask 2)
    desired_mask_index = 2

    if desired_mask_index < len(masks):
        mask_np = masks[desired_mask_index].astype(np.uint8)

        # Overlay the desired mask and return the result
        return overlay_mask(frame, mask_np, mask_color)
    else:
        return frame

def main():
    # Open the webcam
    cap = cv2.VideoCapture(1)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame = process_frame(frame)

        # Display the resulting frame
        cv2.imshow('Webcam Feed', processed_frame)

        # Press 'q' to exit the video feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
