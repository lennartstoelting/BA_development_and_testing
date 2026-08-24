import asyncio
import cv2
from pupil_labs.realtime_api.device import Device
from pupil_labs.realtime_api.streaming import (
    receive_imu_data,
    receive_video_frames,
)

HOST = "pupil-iot.mtec.tu-berlin.de"
PORT = 8080


async def stream_imu(imu_url: str):
    """Worker task for handling incoming IMU packets."""
    print("Started IMU streaming task...")
    async for imu_datum in receive_imu_data(imu_url):
        accel = imu_datum.accel_data
        gyro = imu_datum.gyro_data
        # Process or buffer IMU data
        print(
            f"[IMU]   t={imu_datum.timestamp_unix_seconds:.3f} | "
            f"accel (g): x: {accel.x:.2f} | y: {accel.y:.2f} | z: {accel.z:.2f}"
            f"gyro (m/s): x: {gyro.x:.2f} | y: {gyro.y:.2f} | z: {gyro.z:.2f}"
        )


async def stream_video(video_url: str):
    """Worker task for rendering incoming Scene Video frames."""
    print("Started Video streaming task with GUI...")
    
    # Create a resizable window since 1600x1200 might exceed screen bounds
    cv2.namedWindow("Pupil Neon Scene Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pupil Neon Scene Camera", 800, 600)

    try:
        async for frame in receive_video_frames(video_url):
            image = frame.bgr_buffer()

            # Render frame
            cv2.imshow("Pupil Neon Scene Camera", image)

            # cv2.waitKey(1) processes GUI events (keypresses, window redraws)
            # Break if user presses 'q' or 'ESC'
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                print("Closing video window...")
                break

            # Crucial: yield execution back to asyncio so IMU streaming isn't blocked
            await asyncio.sleep(0.001)

    finally:
        cv2.destroyAllWindows()


async def main():
    print(f"Connecting to device at {HOST}:{PORT}...")

    async with Device(address=HOST, port=PORT) as device:
        status = await device.get_status()

        imu_sensor = status.direct_imu_sensor()
        world_sensor = status.direct_world_sensor()

        if not imu_sensor or not imu_sensor.connected:
            print("IMU sensor is not connected!")
            return
        if not world_sensor or not world_sensor.connected:
            print("World camera sensor is not connected!")
            return

        print(f"IMU URL:   {imu_sensor.url}")
        print(f"World URL: {world_sensor.url}\n")

        # Run both async streams concurrently
        await asyncio.gather(
            stream_imu(imu_sensor.url),
            stream_video(world_sensor.url),
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping dual stream...")