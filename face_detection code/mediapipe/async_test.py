import asyncio
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = './face_landmarker_v2_with_blendshapes.task'

latest_result = None

def callback(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def draw_landmarks(frame, result):
    if not result or not result.face_landmarks:
        return
    h, w, _ = frame.shape
    for face_landmarks in result.face_landmarks:
        for lm in face_landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

async def main():
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_faces=1,
        result_callback=callback
    )

    cap = cv2.VideoCapture(0)

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            # 1. Capture frame in background thread (camera hardware I/O can be slow)
            success, frame = await asyncio.to_thread(cap.read)
            if not success:
                continue

            # 2. Process frame with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            # 3. Draw and Render strictly on the MAIN THREAD
            draw_landmarks(frame, latest_result)
            cv2.imshow('MediaPipe Face Landmarker (Async)', frame)

            # 4. Process GUI events synchronously (1ms non-blocking check on main thread)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 5. Yield control back to the asyncio event loop
            await asyncio.sleep(0.001)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())