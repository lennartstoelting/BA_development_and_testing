import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Path to your downloaded task model
MODEL_PATH = './face_landmarker_v2_with_blendshapes.task'

# Global variable to safely store the latest detection results from the callback
latest_result = None

def result_callback(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """Callback function triggered asynchronously by MediaPipe whenever a frame processed."""
    global latest_result
    latest_result = result

def draw_landmarks(frame, result):
    """Draw simple facial landmark dots on the OpenCV frame."""
    if not result or not result.face_landmarks:
        return

    height, width, _ = frame.shape
    for face_landmarks in result.face_landmarks:
        for landmark in face_landmarks:
            # Convert normalized coordinates (0.0 - 1.0) to pixel coordinates
            cx = int(landmark.x * width)
            cy = int(landmark.y * height)
            cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

def main():
    # 2. Configure MediaPipe Face Landmarker Options
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=1,
        result_callback=result_callback
    )

    # 3. Open webcam capture (0 is usually the default internal/USB webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # 4. Initialize the Landmarker within a context manager
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        print("Starting webcam... Press 'q' to exit.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # OpenCV outputs BGR images; MediaPipe expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Wrap OpenCV NumPy array into a MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # LIVE_STREAM mode requires monotonically increasing millisecond timestamps
            frame_timestamp_ms = int(time.time() * 1000)

            # Send frame for asynchronous processing (triggers result_callback when done)
            landmarker.detect_async(mp_image, frame_timestamp_ms)

            # Draw the most recently received landmarks on the current camera frame
            draw_landmarks(frame, latest_result)

            # Display the video frame
            cv2.imshow('MediaPipe Face Landmarker - Live Stream', frame)

            # Break loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()