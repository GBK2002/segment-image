import cv2
import numpy as np

# Initialize global variables
selected_color = None
clicked = False

def click_event(event, x, y, flags, param):
    global selected_color, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_color = frame[y, x]
        clicked = True
        print(f"Selected color: {selected_color}")

def process_frame(frame, selected_color):
    # Convert the frame to the HSV color space
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Convert the selected color to HSV
    selected_color_hsv = cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)[0][0]

    # Define the color range for masking based on the selected color
    lower_color = np.array([selected_color_hsv[0] - 20, 50, 50])
    upper_color = np.array([selected_color_hsv[0] + 20, 255, 255])

    # Create a mask for the specified color range
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Create a green overlay where the mask is
    green_mask = np.zeros_like(frame)
    green_mask[mask != 0] = [0, 255, 0]  # Green color

    # Overlay the green mask on the original frame
    overlay = cv2.addWeighted(frame, 1, green_mask, 0.5, 0)

    return overlay

def main():
    global frame, clicked

    # Open the webcam
    cap = cv2.VideoCapture(1)

    # Create a named window and set the mouse callback function
    cv2.namedWindow('Webcam Feed')
    cv2.setMouseCallback('Webcam Feed', click_event)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        if clicked:
            # Process the frame if a color has been selected
            processed_frame = process_frame(frame, selected_color)
        else:
            processed_frame = frame

        # Display the resulting frame
        cv2.imshow('Webcam Feed', processed_frame)

        # Press 'q' to exit the video feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
