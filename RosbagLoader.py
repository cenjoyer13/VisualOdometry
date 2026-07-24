import numpy as np
import cv2
import pandas as pd
from scipy.interpolate import interp1d
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R_scipy

from .BaseLoader import BaseLoader
from utils.Cameras import PinholeCamera


class RosbagLoader(BaseLoader):
    def __init__(self, config):
        # 1. Extract paths and topics BEFORE calling BaseLoader's init
        self.bag_path = config.get("bag_path", config.get("datapath"))
        if self.bag_path is None:
            raise ValueError("ERROR: 'bag_path' is missing from your YAML dataset config!")

        self.img_topic = config.get("img_topic", "/camera/image_mono")
        self.gps_topic = config.get("gps_topic", "/fix")
        self.imu_topic = config.get("imu_topic", "/imu/data")
        self.scan_topic = config.get("scan_topic", "/velodyne_points")

        # 2. Call parent init (Handles Time, Rotation, and Camera setup dynamically)
        super().__init__(config)

        # 3. Initialize Pose and ROS Parsing variables
        self.origin_lat = None
        self.origin_lon = None
        self.origin_alt = None
        self.alt = None

        self.actual_alt = None

        self.cur_pose = np.eye(4)
        self.typestore = get_typestore(Stores.ROS1_NOETIC)
        self.body_R = np.eye(3)

        # --- NEW: Initialize PPK before computing altitude ---
        self.ppk_path = config.get("ppk_path", None)
        self.use_ppk = self.ppk_path is not None
        if self.use_ppk:
            print(f"[PPK] Loading high-precision coordinates from: {self.ppk_path}")
            self._init_ppk()

        # --- FIX: PRE-CALCULATE BASELINE AGL ONCE ---
        self.baseline_agl = self.get_minimum_bag_altitude()
        # --------------------------------------------

        # New IMU Integration Variables
        self.imu_velocity = np.zeros((3, 1))
        self.imu_position = np.zeros((3, 1))
        self.last_imu_time = None
        self.gravity = np.array([[0.0], [0.0], [9.81]])

        # NEW: Track the first IMU pose to zero-align its trajectory
        self.first_imu_pose_inv = None
        self.use_imu = config.get('imu', False)

    def _init_ppk(self):
        lines = []
        with open(self.ppk_path, "r") as f:
            for line in f:
                if line.startswith("%"): continue
                parts = line.strip().split()
                if len(parts) >= 14:
                    # Combine Date and Time into one string
                    date_str = parts[0] + " " + parts[1]
                    lat, lon, alt = float(parts[2]), float(parts[3]), float(parts[4])
                    lines.append({"gpst": date_str, "lat": lat, "lon": lon, "alt": alt})

        df = pd.DataFrame(lines)
        df['gpst'] = pd.to_datetime(df['gpst'])

        # FIX TIMEZONE: GPS Time was 18 seconds ahead of UTC in 2022
        df['utc_time'] = df['gpst'] - pd.Timedelta(seconds=18)

        # Convert to UNIX nanoseconds to match rosbag exactly
        df['ts_nanos'] = df['utc_time'].astype('int64')

        times = df['ts_nanos'].values

        # Create continuous interpolators mapped to the time vectors
        self.lat_interp = interp1d(times, df['lat'].values, kind='linear', fill_value='extrapolate')
        self.lon_interp = interp1d(times, df['lon'].values, kind='linear', fill_value='extrapolate')
        self.alt_interp = interp1d(times, df['alt'].values, kind='linear', fill_value='extrapolate')

        self.ppk_alt_values = df['alt'].values

    def get_minimum_bag_altitude(self):
        """Pre-scans the bag (or PPK) to find the absolute lowest altitude (True Ground)."""
        if hasattr(self, 'use_ppk') and self.use_ppk:
            min_alt = float(np.min(self.ppk_alt_values))
            print(f"[Altimeter] SUCCESS: Found true ground level at {min_alt:.2f}m MSL from PPK.")
            return min_alt

        print("[Altimeter] Pre-scanning bag for global minimum GPS altitude...")
        min_alt = float('inf')

        # Open the bag and ONLY read the GPS topic (this takes less than a second)
        with Reader(self.bag_path) as reader:
            connections = [x for x in reader.connections if x.topic == self.gps_topic]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)

                # Ignore NaNs
                if not np.isnan(msg.altitude):
                    if msg.altitude < min_alt:
                        min_alt = msg.altitude

        if min_alt == float('inf'):
            print("[Altimeter] WARNING: No valid GPS altitude found. Defaulting to 0.0")
            return 0.0

        print(f"[Altimeter] SUCCESS: Found true ground level at {min_alt:.2f}m MSL.")
        return min_alt

    def process_imu_pose(self, imu_pose):
        """Universally aligns the first IMU pose to the Origin."""
        if imu_pose is None:
            return np.eye(4)

        # 1. Capture and invert the very first IMU pose we see
        if self.first_imu_pose_inv is None:
            self.first_imu_pose_inv = np.linalg.inv(imu_pose)

        # 2. Shift origin to start at (0,0,0)
        aligned_imu_pose = self.first_imu_pose_inv @ imu_pose

        # 3. Apply the same user-defined rotation so GT and IMU share the same world frame
        return self.align_transform @ aligned_imu_pose

    def _load_default_camera(self):
        """
        Unlike KITTI or TUM, rosbags rarely have a standardized 'calib.txt' fallback.
        If the user did not supply a camera configuration in the YAML, we fall back
        to a generic, un-distorted pinhole camera to prevent a total crash.
        """
        print("WARNING: No camera provided in config. Defaulting to generic Pinhole!")
        return PinholeCamera({
            'image_width': 640, 'image_height': 480,
            'fx': 500.0, 'fy': 500.0,
            'cx': 320.0, 'cy': 240.0
        })

    def _update_gt_translation(self, msg, timestamp):
        # NEW: Interpolate exactly from the PPK curve based on the current ROS nanosecond
        if hasattr(self, 'use_ppk') and self.use_ppk:
            lat = float(self.lat_interp(timestamp))
            lon = float(self.lon_interp(timestamp))
            self.alt = float(self.alt_interp(timestamp))
        else:
            lat, lon = msg.latitude, msg.longitude

            # --- NEW: NaN Safety Check ---
            # If the GPS drops out, return immediately.
            # self.cur_pose will retain its previous valid matrix values.
            if np.isnan(lat) or np.isnan(lon):
                return

            self.alt = msg.altitude if not np.isnan(msg.altitude) else 0.0

        if self.origin_lat is None:
            self.origin_lat, self.origin_lon, self.origin_alt = lat, lon, self.alt

        R = 6378137.0
        # 1. Calculate Distances
        d_lon_east = R * np.radians(lon - self.origin_lon) * np.cos(np.radians(self.origin_lat))
        d_lat_north = R * np.radians(lat - self.origin_lat)
        d_alt_up = self.alt - self.origin_alt

        # 2. Assign to Slots in the 4x4 Pose Matrix
        # Matrix Row 0 (X) = North
        self.cur_pose[0, 3] = d_lon_east
        # Matrix Row 1 (Y) = East
        self.cur_pose[1, 3] = d_lat_north
        # Matrix Row 2 (Z) = Z-up altitude difference
        self.cur_pose[2, 3] = d_alt_up

    def _update_gt_rotation(self, msg):
        q = msg.orientation
        # Use raw quaternion if the sensor is already in NED
        quat = [q.x, q.y, q.z, q.w]

        # Convert directly to matrix and assign to the top-left 3x3 block
        rot_matrix = R_scipy.from_quat(quat).as_matrix()

        # UPDATED: Enforcing Z-up orientation for the body_R frame
        r_cam_to_body = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ], dtype=np.float64)

        self.cur_pose[:3, :3] = rot_matrix
        self.body_R = rot_matrix @ r_cam_to_body

    def _update_imu_path(self, msg, timestamp_nanos):
        current_time_sec = timestamp_nanos / 1e9
        if self.last_imu_time is None:
            self.last_imu_time = current_time_sec
            return

        dt = current_time_sec - self.last_imu_time
        self.last_imu_time = current_time_sec

        # 1. Extract Acceleration (Local Frame)
        accel_local = np.array([
            [msg.linear_acceleration.x],
            [msg.linear_acceleration.y],
            [msg.linear_acceleration.z]
        ])

        # 2. Extract Orientation (Local to World ENU)
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        R_local_to_world = R_scipy.from_quat(quat).as_matrix()

        # 3. Rotate Acceleration into World ENU Frame
        accel_world_enu = R_local_to_world @ accel_local

        # 4. Remove Gravity
        accel_world_enu_true = accel_world_enu - self.gravity

        # 5. CONVERT ENU TO WORLD FRAME TO MATCH GPS
        accel_world_ned_true = np.array([
            [accel_world_enu_true[1, 0]],  # Index 1 is ENU Y
            [accel_world_enu_true[0, 0]],  # Index 0 is ENU X
            [accel_world_enu_true[2, 0]]  # Kept positive to align with Z-up altitude
        ])

        # 6. Integrate to Velocity and Position
        self.imu_velocity += (accel_world_ned_true * dt)
        self.imu_position += (self.imu_velocity * dt)

    def __iter__(self):
        bag_start_time = None
        processing_start_time = None

        # Convert BaseLoader's unified time bounds (seconds) into rosbag nanoseconds
        skip_nanos = self.start_time * 1e9
        end_nanos = self.end_time * 1e9 if self.end_time != float('inf') else None

        with Reader(self.bag_path) as reader:
            connections = [x for x in reader.connections
                           if x.topic in [self.img_topic, self.gps_topic, self.imu_topic, self.scan_topic]]

            for connection, timestamp, rawdata in reader.messages(connections=connections):

                if bag_start_time is None:
                    bag_start_time = timestamp

                # Memory & I/O Optimization: Skip decoding entirely if outside time bounds
                if end_nanos is not None and (timestamp - bag_start_time) > end_nanos:
                    break

                if (timestamp - bag_start_time) < skip_nanos:
                    continue

                if processing_start_time is None:
                    processing_start_time = timestamp

                msg = self.typestore.deserialize_ros1(rawdata, connection.msgtype)

                # 1. Always update IMU (runs in background)
                if connection.topic == self.imu_topic:
                    self._update_gt_rotation(msg)
                    if self.use_imu:
                        self._update_imu_path(msg, timestamp)

                # 2. Update Ground Truth state when GPS arrives (runs in background)
                elif connection.topic == self.gps_topic:
                    if not self.use_ppk:
                        self._update_gt_translation(msg, timestamp)

                # 3. YIELD ON IMAGE ARRIVAL
                elif connection.topic == self.img_topic:

                    # If using PPK, calculate the exact GT position for THIS specific image timestamp
                    if self.use_ppk:
                        self._update_gt_translation(None, timestamp)

                    # Guard: Wait until the first GPS packet establishes a local origin (0,0,0)
                    if self.origin_lat is None:
                        continue

                    # Decode Image
                    if msg.encoding in ['mono8', '8UC1']:
                        img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                    else:
                        img_raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                        img_data = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

                    ts_sec = (timestamp - processing_start_time) / 1e9

                    # Calculate true AGL using the pre-scanned global minimum
                    if self.alt is not None and self.baseline_agl is not None:
                        self.actual_alt = self.alt - self.baseline_agl
                    else:
                        self.actual_alt = 0.0

                    # Universal BaseLoader processing
                    processed_img = self.process_image(img_data)
                    final_pose = self.process_pose(self.cur_pose)

                    # Process and yield the IMU pose if enabled
                    if self.use_imu:
                        # Assemble the local 4x4 matrix using IMU tracking data
                        cur_imu_pose = np.eye(4)
                        cur_imu_pose[:3, :3] = self.cur_pose[:3, :3]
                        cur_imu_pose[:3, 3] = self.imu_position.flatten()

                        final_imu_pose = self.process_imu_pose(cur_imu_pose)
                        yield ts_sec, processed_img.copy(), final_pose.copy(), final_imu_pose.copy()
                    else:
                        yield ts_sec, processed_img.copy(), final_pose.copy(), None