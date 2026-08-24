import asyncio
import csv
from dataclasses import dataclass
from typing import AsyncGenerator

# Minimal mock classes matching live API accessors
@dataclass
class MockData3D:
    x: float
    y: float
    z: float

@dataclass
class MockIMUData:
    accel_data: MockData3D
    gyro_data: MockData3D
    timestamp_unix_seconds: float
    timestamp_unix_ns: int

async def simulate_imu_stream(
    csv_file_path: str, real_time: bool = False
) -> AsyncGenerator[MockIMUData, None]:
    """Yields MockIMUData objects from recorded CSV matching Pupil Labs API structure."""
    with open(csv_file_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        last_timestamp = None

        for row in reader:
            ts_ns = int(row["timestamp [ns]"])
            ts_sec = ts_ns / 1e9

            accel = MockData3D(
                x=float(row["accel x [g]"]),
                y=float(row["accel y [g]"]),
                z=float(row["accel z [g]"])
            )
            gyro = MockData3D(
                x=float(row["gyro x [deg/s]"]),
                y=float(row["gyro y [deg/s]"]),
                z=float(row["gyro z [deg/s]"])
            )

            # Optional: simulate real-time inter-frame delay (~200Hz sampling)
            if real_time and last_timestamp is not None:
                delay = ts_sec - last_timestamp
                if delay > 0:
                    await asyncio.sleep(delay)
            last_timestamp = ts_sec

            yield MockIMUData(
                accel_data=accel,
                gyro_data=gyro,
                timestamp_unix_seconds=ts_sec,
                timestamp_unix_ns=ts_ns
            )