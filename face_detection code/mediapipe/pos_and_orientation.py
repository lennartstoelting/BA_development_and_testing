import asyncio
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Tell OpenCV/Qt to use system fontconfig and shut up about font directories
os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts'

MODEL_PATH = './face_landmarker_v2_with_blendshapes.task'

latest_result = None

def callback(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def draw_head_pose_axes(frame, result):
    if not result or not result.facial_transformation_matrixes:
        return

    h, w, _ = frame.shape
    
    for matrix_data in result.facial_transformation_matrixes:
        matrix = np.array(matrix_data).reshape(4, 4)

        # 1. Print position coordinates to console
        tx, ty, tz = matrix[0, 3], matrix[1, 3], matrix[2, 3]
        print(f"\rHead Position -> X: {tx:6.2f} | Y: {ty:6.2f} | Z: {tz:6.2f}", end="")

        if not result.face_landmarks:
            continue
        
        nose_tip = result.face_landmarks[0][1]
        origin = np.array([int(nose_tip.x * w), int(nose_tip.y * h)], dtype=np.int32)

        # 2. Extract 3x3 Rotation matrix
        rot_mat = matrix[:3, :3]

        # 3. Coordinate conversion matrix to map 3D pose -> 2D Image Space (Y-Down, Z-Forward)
        # Flipping Y and Z preserves the right-hand rule without breaking roll/pitch interactions
        cam_transform = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])

        # Combined rotation transformed into screen coordinates
        screen_rot = cam_transform @ rot_mat

        axis_length = 60.0

        # Compute projected 2D offsets directly from transformed rotation
        x_offset = screen_rot @ np.array([axis_length, 0, 0])
        y_offset = screen_rot @ np.array([0, axis_length, 0])
        z_offset = screen_rot @ np.array([0, 0, axis_length])

        # Project 2D endpoints (X and Y components of the projected vectors)
        p_x = (int(origin[0] + x_offset[0]), int(origin[1] + x_offset[1]))
        p_y = (int(origin[0] + y_offset[0]), int(origin[1] + y_offset[1]))
        p_z = (int(origin[0] + z_offset[0]), int(origin[1] + z_offset[1]))

        # Draw RGB coordinate axes
        cv2.line(frame, tuple(origin), p_x, (0, 0, 255), 3)  # Red: X axis (Right)
        cv2.line(frame, tuple(origin), p_y, (0, 255, 0), 3)  # Green: Y axis (Down)
        cv2.line(frame, tuple(origin), p_z, (255, 0, 0), 3)  # Blue: Z axis (Outward)

async def main():
    # Enable matrix output explicitly in options
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=1,
        output_facial_transformation_matrixes=True, # <--- Added flag
        result_callback=callback
    )

    cap = cv2.VideoCapture(0)

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        print("Running Head Pose tracker... Press 'q' to quit.")
        
        while cap.isOpened():
            success, frame = await asyncio.to_thread(cap.read)
            if not success:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            # Draw the 3D RGB Pose Axes instead of the facial dots mesh
            draw_head_pose_axes(frame, latest_result)

            cv2.imshow('MediaPipe Head Pose (RGB Axes)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0.001)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())