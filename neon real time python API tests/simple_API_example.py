from pupil_labs.realtime_api.simple import Device

# Replace discovery with direct connection to your device's network address
HOST = "pupil-iot.mtec.tu-berlin.de"
PORT = 8080

try:
    print(f"Connecting directly to {HOST}:{PORT}...")
    device = Device(address=HOST, port=PORT)
    
    print(f"Connected to {device.serial_number_glasses}. Press Ctrl-C to stop.")

    # Stream gaze data
    while True:
        # receive_gaze_datum() will return the next available gaze datum
        # or block until one becomes available.
        gaze = device.receive_gaze_datum()
        
        print(
            f"Timestamp: {gaze.timestamp_unix_seconds:.3f} | "
            f"Gaze (x,y): ({gaze.x:.2f}, {gaze.y:.2f}) | "
            f"Worn: {gaze.worn}"
        )

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    # Cleanly close the connection
    if "device" in locals() and device:
        device.close()
    print("Connection closed.")