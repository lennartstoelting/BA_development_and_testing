import asyncio
from pupil_labs.realtime_api.device import Device
from pupil_labs.realtime_api.streaming import receive_imu_data

HOST = "pupil-iot.mtec.tu-berlin.de"
PORT = 8080

async def main():
    print(f"Connecting to device at {HOST}:{PORT}...")
    
    # 1. Connect to the device via HTTP REST API (port 8080)
    async with Device(address=HOST, port=PORT) as device:
        # 2. Get status object containing active sensor URLs
        status = await device.get_status()
        
        print(f"Device IP: {status.phone.ip}")
        print(f"Battery: {status.phone.battery_level}%")
        
        # 3. Retrieve the IMU sensor info/url dynamically
        imu_sensor = status.direct_imu_sensor()
        print(f"IMU sensor connected: {imu_sensor.connected}")
        print(f"IMU Stream URL: {imu_sensor.url}")

        if not imu_sensor.connected:
            print("IMU sensor is not currently connected/active.")
            return

        # 4. Stream IMU data asynchronously using the retrieved URL
        print("\nStreaming IMU data (Accel & Gyro)... Press Ctrl-C to stop.\n")
        async for imu_datum in receive_imu_data(imu_sensor.url):
            accel = imu_datum.accel_data
            gyro = imu_datum.gyro_data
            
            print(
                f"Timestamp: {imu_datum.timestamp_unix_seconds:.3f} | "
                f"Accel (m/s²): ({accel.x:.2f}, {accel.y:.2f}, {accel.z:.2f}) | "
                f"Gyro (deg/s): ({gyro.x:.2f}, {gyro.y:.2f}, {gyro.z:.2f})"
            )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping...")


# little different implementation with the IP adress
# import asyncio
# from pupil_labs.realtime_api.device import Device
# from pupil_labs.realtime_api.streaming import receive_imu_data

# # Use your device's exact IP address
# IP_ADDRESS = "141.23.125.50"  # Replace with your actual IP
# PORT = 8080

# async def main():
#     print(f"Connecting to device at {IP_ADDRESS}:{PORT}...")
    
#     async with Device(address=IP_ADDRESS, port=PORT) as device:
#         status = await device.get_status()
        
#         imu_sensor = status.direct_imu_sensor()
#         if not imu_sensor or not imu_sensor.connected:
#             print("IMU sensor not connected.")
#             return

#         print(f"Streaming from RTSP URL: {imu_sensor.url}")
#         async for imu_datum in receive_imu_data(imu_sensor.url):
#             accel = imu_datum.accel_data
#             gyro = imu_datum.gyro_data
#             print(
#                 f"Accel: ({accel.x:.2f}, {accel.y:.2f}, {accel.z:.2f}) | "
#                 f"Gyro: ({gyro.x:.2f}, {gyro.y:.2f}, {gyro.z:.2f})"
#             )

# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("\nStopping...")