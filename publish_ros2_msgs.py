import sys
import struct
import os
import cv2
import rclpy
from rclpy.node import Node
import csv
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Imu, Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
import numpy as np
import time
from std_msgs.msg import Header
from tqdm import tqdm
import tf2_ros


def adjust_rgb_and_camera_matrix(rgb_image, depth_image, camera_matrix):
    """
    Adjust RGB image size to match Depth image and update camera matrix.

    Args:
        rgb_image (numpy.ndarray): Original RGB image.
        depth_image (numpy.ndarray): Depth image.
        camera_matrix (numpy.ndarray): Original camera intrinsic matrix.

    Returns:
        resized_rgb (numpy.ndarray): Resized RGB image.
        adjusted_camera_matrix (numpy.ndarray): Adjusted camera intrinsic matrix.
    """

    depth_h, depth_w = depth_image.shape[:2]
    rgb_h, rgb_w = rgb_image.shape[:2]

    # Calculate scale factors
    scale_x = depth_w / rgb_w
    scale_y = depth_h / rgb_h

    # Resize RGB image
    resized_rgb = cv2.resize(
        rgb_image, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR
    )

    # Adjust camera matrix
    adjusted_camera_matrix = camera_matrix.copy()
    adjusted_camera_matrix[0, 0] *= scale_x  # fx
    adjusted_camera_matrix[1, 1] *= scale_y  # fy
    adjusted_camera_matrix[0, 2] *= scale_x  # cx
    adjusted_camera_matrix[1, 2] *= scale_y  # cy

    return resized_rgb, adjusted_camera_matrix


def create_pointcloud2(points_with_rgb, frame_id):
    """
    Create a PointCloud2 message from 3D points with RGB.

    Args:
        points_with_rgb (numpy.ndarray): Nx6 array of 3D points with RGB (x, y, z, r, g, b).

    Returns:
        pointcloud_msg (PointCloud2): ROS 2 PointCloud2 message.
    """
    pointcloud_msg = PointCloud2()

    # Header
    pointcloud_msg.header.frame_id = frame_id

    # Define fields for PointCloud2
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
    ]
    pointcloud_msg.fields = fields

    # Calculate point step and row step
    pointcloud_msg.point_step = 16  # 4 bytes (float32) * 3 + 4 bytes (uint32)
    pointcloud_msg.row_step = pointcloud_msg.point_step * len(points_with_rgb)

    # Convert points to byte data
    buffer = []
    for point in points_with_rgb:
        x, y, z, r, g, b = point
        # Ensure RGB values are in 0-255 range and convert to uint32
        r, g, b = int(r), int(g), int(b)
        rgb = struct.unpack("I", struct.pack("BBBB", b, g, r, 0))[0]
        buffer.append(struct.pack("fffI", x, y, z, rgb))

    pointcloud_msg.data = b"".join(buffer)
    pointcloud_msg.is_bigendian = False
    pointcloud_msg.height = 1
    pointcloud_msg.width = len(points_with_rgb)
    pointcloud_msg.is_dense = True

    return pointcloud_msg


def unproject_depth(depth_image, rgb_image, camera_matrix, mask=None):
    """
    Generate a 3D point cloud with color from a depth image and camera matrix.
    Only include points where mask is True.

    Args:
        depth_image (numpy.ndarray): Depth image in mm (uint16).
        rgb_image (numpy.ndarray): RGB image matching the depth image size.
        camera_matrix (numpy.ndarray): Camera intrinsic matrix.
        mask (numpy.ndarray, optional): Boolean mask to filter points.

    Returns:
        point_cloud (numpy.ndarray): Nx6 array of 3D points with RGB colors (x, y, z, r, g, b).
    """
    # Get camera intrinsics
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Create a grid of pixel coordinates
    depth_h, depth_w = depth_image.shape
    u, v = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
    u = u.flatten()
    v = v.flatten()

    # Flatten depth image and filter valid depths
    z = depth_image.flatten().astype(np.float32) / 1000.0  # Convert mm to meters
    valid = z > 0  # Ignore invalid depth values

    if mask is not None:
        mask = mask.flatten().astype(bool)
        valid = valid & mask

    z = z[valid]
    u = u[valid]
    v = v[valid]

    # Backproject to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_3d = np.stack((x, y, z), axis=-1)  # Shape: (N, 3)

    # Add color from RGB image
    rgb_flat = rgb_image.reshape(-1, 3)  # Flatten RGB image
    colors = rgb_flat[valid]  # Get colors for valid depth points

    # Combine points and colors
    return np.concatenate((points_3d, colors), axis=-1)  # Shape: (N, 6)


def read_csv(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return data


def camera_intrinsics_from_odometry(row):
    """Read fx, fy, cx, cy in full RGB pixels; distortion centers may be empty."""
    if len(row) < 15:
        raise ValueError(
            "odometry.csv must contain pose data followed by fx, fy, cx, cy, "
            "distortion_center_x, distortion_center_y"
        )
    intrinsics = tuple(float(value) for value in row[-6:-2])    # select only fx, fy, cx, cy
    if not all(np.isfinite(value) for value in intrinsics):
        raise ValueError("Camera intrinsics must be finite")
    if intrinsics[0] <= 0 or intrinsics[1] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    return intrinsics


def create_camera_info(intrinsics, header, source_size, image_size):
    """Build RGB or aligned-depth CameraInfo from full-resolution RGB intrinsics."""
    fx, fy, cx, cy = intrinsics
    source_width, source_height = source_size
    width, height = image_size
    scale_x, scale_y = width / source_width, height / source_height
    fx, cx = fx * scale_x, cx * scale_x
    fy, cy = fy * scale_y, cy * scale_y

    msg = CameraInfo()
    msg.header = header
    msg.width = width
    msg.height = height
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    # CameraInfo has no distortion-center fields. These coordinates are NOT
    # plumb_bob coefficients (k1, k2, t1, t2, k3). No model/coefficients are
    # supplied by this CSV, so leave distortion unspecified.
    msg.distortion_model = ""
    msg.d = []
    return msg


def read_rgb_source_size(video_path, rgb_data):
    """Get the original RGB size, independent of playback and RGB downsampling."""
    cap = cv2.VideoCapture(video_path)
    try:
        success, frame = cap.read()
        if success and frame is not None:
            return (frame.shape[1], frame.shape[0])
    finally:
        cap.release()
    # Allow playback from the original extracted images if video is unavailable.
    for image_info in rgb_data:
        frame = cv2.imread(image_info["path"])
        if frame is not None:
            return (frame.shape[1], frame.shape[0])
    raise ValueError("Cannot determine full-resolution RGB size for depth intrinsics")


def skew_symmetric(v):
    """
    Creates a skew-symmetric matrix from a vector.
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def combine_and_sort_data(imu_data, odometry_data, rgb_data, depth_data, k=1):
    """
    Combine and sort sensor data, skipping k-1 entries for each type.

    Args:
        imu_data (list): IMU data from CSV.
        odometry_data (list): Odometry data from CSV.
        rgb_data (list): RGB image data.
        depth_data (list): Depth image data.
        k (int): Step size for skipping data. Default is 1 (no skipping).

    Returns:
        list: Combined and sorted data.
    """
    combined_data = []

    # Combine IMU data with skipping
    for i, row in enumerate(imu_data):  # Header already removed by caller
        if i % k == 0:
            combined_data.append(
                {"timestamp": float(row[0]), "type": "imu", "data": row}
            )

    # Combine Odometry data with skipping
    for i, row in enumerate(odometry_data):  # Header already removed by caller
        if i % k == 0:
            combined_data.append(
                {"timestamp": float(row[0]), "type": "odometry", "data": row}
            )

    # Combine RGB data with skipping
    for i, image_info in enumerate(rgb_data):
        if i % k == 0:
            combined_data.append(
                {
                    "timestamp": image_info["timestamp"],
                    "type": "rgb",
                    "data": image_info,
                }
            )

    # Combine Depth data with skipping
    for i, image_info in enumerate(depth_data):
        if i % k == 0:
            combined_data.append(
                {
                    "timestamp": image_info["timestamp"],
                    "type": "depth",
                    "data": image_info,
                }
            )

    # Sort by timestamp
    combined_data.sort(key=lambda x: x["timestamp"])
    return combined_data


class StrayScannerDataPublisher(Node):
    def __init__(self, data_dir, playback_speed=1.0):
        super().__init__("csv_and_video_publisher")

        self.playback_speed = playback_speed  # 배속 계수 저장

        # Publishers
        self.imu_pub = self.create_publisher(Imu, "/imu", 100)
        self.odometry_pub = self.create_publisher(Odometry, "/odometry", 100)
        self.rgb_pub = self.create_publisher(Image, "/camera/rgb", 100)
        self.rgb_camera_info_pub = self.create_publisher(
            CameraInfo, "/camera/rgb/camera_info", 100
        )
        self.depth_pub = self.create_publisher(Image, "/camera/depth", 100)
        self.depth_camera_info_pub = self.create_publisher(
            CameraInfo, "/camera/depth/camera_info", 100
        )

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.publish_static_transforms()

        # Load CSV data
        imu_data = read_csv(f"{data_dir}/imu.csv")[1:]  # Skip header
        odometry_data = read_csv(f"{data_dir}/odometry.csv")[1:]  # Skip header

        # Prepare Depth data with Confidence
        self.depth_dir = os.path.join(data_dir, "depth")
        depth_data = self.prepare_image_paths(self.depth_dir, odometry_data)
        self.num_frames = len(depth_data)

        # Convert the video to images
        video_path = f"{data_dir}/rgb.mp4"
        print(f"video_path: {video_path}")
        self.rgb_dir = os.path.join(data_dir, "images")
        # Check if rgb_dir exists
        if os.path.exists(self.rgb_dir) and len(os.listdir(self.rgb_dir)) > (
            self.num_frames - 3
        ):
            print(f"{self.rgb_dir} already exists. Skipping save_frames.")
        else:
            os.makedirs(self.rgb_dir, exist_ok=True)
            self.save_frames(video_path)

        # Prepare RGB data
        rgb_data = self.prepare_image_paths(self.rgb_dir, odometry_data)
        self.rgb_source_size = read_rgb_source_size(video_path, rgb_data)

        # pub option
        self.skip_frame_k = 1

        self.pose_matrix = None

        # Combine and sort all data
        self.sorted_data = combine_and_sort_data(
            imu_data, odometry_data, rgb_data, depth_data, self.skip_frame_k
        )
        self.current_index = 0
        self.bridge = CvBridge()

        # Initialize progress bar
        self.progress_bar = tqdm(total=len(self.sorted_data), desc="Publishing data")

        # Start publishing
        self.start_time = time.time()
        self.initial_timestamp = self.sorted_data[0]["timestamp"]
        self.timer = self.create_timer(0.001, self.publish_data)  # High frequency timer

    def publish_static_transforms(self):
        transforms = []

        R_baselink_to_cam = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ])
        q = R.from_matrix(R_baselink_to_cam).as_quat()

        tf_rgb = TransformStamped()
        tf_rgb.header.stamp = self.get_clock().now().to_msg()
        tf_rgb.header.frame_id = "base_link"
        tf_rgb.child_frame_id = "camera_rgb_frame"

        tf_rgb.transform.translation.x = 0.0
        tf_rgb.transform.translation.y = 0.0
        tf_rgb.transform.translation.z = 0.0

        tf_rgb.transform.rotation.x = float(q[0])
        tf_rgb.transform.rotation.y = float(q[1])
        tf_rgb.transform.rotation.z = float(q[2])
        tf_rgb.transform.rotation.w = float(q[3])

        transforms.append(tf_rgb)

        tf_depth = TransformStamped()
        tf_depth.header.stamp = self.get_clock().now().to_msg()
        tf_depth.header.frame_id = "camera_rgb_frame"
        tf_depth.child_frame_id = "camera_depth_frame"

        tf_depth.transform.translation.x = 0.0
        tf_depth.transform.translation.y = 0.0
        tf_depth.transform.translation.z = 0.0

        tf_depth.transform.rotation.x = 0.0
        tf_depth.transform.rotation.y = 0.0
        tf_depth.transform.rotation.z = 0.0
        tf_depth.transform.rotation.w = 1.0

        transforms.append(tf_depth)
        self.tf_static_broadcaster.sendTransform(transforms)

    def save_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        rgb_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_path = os.path.join(self.rgb_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(image_path, frame)

            rgb_data.append(image_path)
            frame_idx += 1

        cap.release()
        self.num_frames = len(rgb_data)
        print(f"num_frames: {self.num_frames}")
        return rgb_data

    def prepare_image_paths(self, image_dir, odometry_data):
        image_data = []
        for row in odometry_data:
            frame_id = int(row[1])
            timestamp = float(row[0])
            if "depth" in image_dir:
                image_path = os.path.join(image_dir, f"{frame_id:06d}.png")
                confidence_dir = image_dir.replace("depth", "confidence")
                confidence_path = os.path.join(confidence_dir, f"{frame_id:06d}.png")
                if os.path.exists(image_path) and os.path.exists(confidence_path):
                    image_data.append(
                        {
                            "timestamp": timestamp,
                            "depth_path": image_path,
                            "confidence_path": confidence_path,
                            "intrinsics": camera_intrinsics_from_odometry(row),
                        }
                    )
            elif "confidence" in image_dir:
                # Confidence images are handled alongside depth images
                pass
            else:
                # For RGB images
                image_path = os.path.join(image_dir, f"{frame_id:06d}.jpg")
                if os.path.exists(image_path):
                    image_data.append({
                        "timestamp": timestamp,
                        "path": image_path,
                        "intrinsics": camera_intrinsics_from_odometry(row),
                    })
        return image_data

    def publish_data(self):
        if self.current_index >= len(self.sorted_data):
            self.progress_bar.close()  # Close progress bar
            self.get_logger().info("All data published.")
            self.destroy_timer(self.timer)
            return

        current_time = time.time()
        elapsed_time = (current_time - self.start_time) * self.playback_speed  # 배속 반영

        while (
            self.current_index < len(self.sorted_data)
            and self.sorted_data[self.current_index]["timestamp"]
            - self.initial_timestamp
            <= elapsed_time
        ):
            entry = self.sorted_data[self.current_index]
            timestamp = entry["timestamp"]

            if entry["type"] == "imu":
                # IMU Publishing
                imu_row = entry["data"]
                imu_msg = Imu()
                imu_msg.header.stamp = rclpy.time.Time(seconds=timestamp).to_msg()
                imu_msg.header.frame_id = "imu_frame"
                imu_msg.linear_acceleration.x = float(imu_row[1])
                imu_msg.linear_acceleration.y = float(imu_row[2])
                imu_msg.linear_acceleration.z = float(imu_row[3])
                imu_msg.angular_velocity.x = float(imu_row[4])
                imu_msg.angular_velocity.y = float(imu_row[5])
                imu_msg.angular_velocity.z = float(imu_row[6])
                self.imu_pub.publish(imu_msg)

            elif entry["type"] == "odometry":
                # Odometry Publishing
                odom_row = entry["data"]
                odom_msg = Odometry()
                odom_msg.header.stamp = rclpy.time.Time(seconds=timestamp).to_msg()
                odom_msg.header.frame_id = "odom_frame"
                odom_msg.child_frame_id = "base_link"

                # Extract translation
                x = float(odom_row[2])
                y = float(odom_row[3])
                z = float(odom_row[4])

                # Extract quaternion
                qx = float(odom_row[5])
                qy = float(odom_row[6])
                qz = float(odom_row[7])
                qw = float(odom_row[8])

                # Create a rotation matrix from the quaternion
                rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

                # Construct the SE(3) transformation matrix
                pose_body = np.eye(4)
                pose_body[:3, :3] = rotation  # Top-left 3x3 block is the rotation matrix
                pose_body[:3, 3] = [x, y, z]  # Top-right 3x1 block is the translation vector

                T_cam_to_FLU = np.array(
                    [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
                )

                T_world_to_FLU = np.array([
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ])

                # New: ROS FLU base_link expressed in ARKit camera axes.
                T_arkit_cam_from_base = np.array([
                    [ 0, -1,  0, 0],
                    [ 0,  0,  -1, 0],
                    [1,  0,  0, 0],
                    [ 0,  0,  0, 1]
                ], dtype=float)
                
                # pose_body is actually the raw ARKit camera pose.
                pose_world = (
                    T_world_to_FLU
                    @ pose_body
                    @ T_arkit_cam_from_base
                )

                #pose_world = T_cam_to_FLU @ pose_body

                # Extract position
                position_world = pose_world[:3, 3]

                # Extract rotation matrix
                rotation_matrix_world = pose_world[:3, :3]

                # Convert rotation matrix to quaternion
                quaternion_world = R.from_matrix(rotation_matrix_world).as_quat()  # [qx, qy, qz, qw]

                odom_msg.pose.pose.position.x = position_world[0]
                odom_msg.pose.pose.position.y = position_world[1]
                odom_msg.pose.pose.position.z = position_world[2]
                odom_msg.pose.pose.orientation = Quaternion(
                    x=quaternion_world[0],
                    y=quaternion_world[1],
                    z=quaternion_world[2],
                    w=quaternion_world[3],
                )

                # Construct SE(3) transformation matrix
                pose_matrix = np.eye(4)  # Initialize 4x4 identity matrix
                pose_matrix[:3, :3] = (
                    rotation_matrix_world  # Top-left 3x3 is the rotation matrix
                )
                pose_matrix[:3, 3] = position_world  # Top-right 3x1 is the translation vector
                self.pose_matrix = pose_matrix  # update

                # Publish the updated odometry message
                self.odometry_pub.publish(odom_msg)

                # Create and broadcast transform
                transform = TransformStamped()
                transform.header.stamp = odom_msg.header.stamp
                transform.header.frame_id = "odom_frame"
                transform.child_frame_id = "base_link"

                transform.transform.translation.x = position_world[0]
                transform.transform.translation.y = position_world[1]
                transform.transform.translation.z = position_world[2]

                transform.transform.rotation.x = quaternion_world[0]
                transform.transform.rotation.y = quaternion_world[1]
                transform.transform.rotation.z = quaternion_world[2]
                transform.transform.rotation.w = quaternion_world[3]

                self.tf_broadcaster.sendTransform(transform)

            elif entry["type"] == "rgb":
                # RGB Publishing
                image_info = entry["data"]
                bgr_img = cv2.imread(image_info["path"])
                if bgr_img is not None:
                    # 가로세로 절반으로 다운사이즈
                    height, width = bgr_img.shape[:2]
                    resized_img = cv2.resize(bgr_img, (width // 2, height // 2))

                    # 오른쪽으로 90도 회전
                    #rotated_img = cv2.rotate(resized_img, cv2.ROTATE_90_CLOCKWISE)
                    rotated_img = resized_img

                    img_msg = self.bridge.cv2_to_imgmsg(rotated_img, encoding="bgr8")
                    img_msg.header.stamp = rclpy.time.Time(seconds=timestamp).to_msg()
                    img_msg.header.frame_id = "camera_rgb_frame"
                    camera_info_msg = create_camera_info(
                        image_info["intrinsics"],
                        img_msg.header,
                        (width, height),
                        (rotated_img.shape[1], rotated_img.shape[0]),
                    )
                    self.rgb_pub.publish(img_msg)
                    self.rgb_camera_info_pub.publish(camera_info_msg)

            elif entry["type"] == "depth":
                # Depth and Confidence Handling
                image_info = entry["data"]
                depth_img = cv2.imread(image_info["depth_path"], cv2.IMREAD_UNCHANGED)
                confidence_img = cv2.imread(
                    image_info["confidence_path"], cv2.IMREAD_UNCHANGED
                )  # 0, 1, 2

                if (
                    depth_img is not None
                    and depth_img.dtype == np.uint16
                    and confidence_img is not None
                ):
                    # Publish depth image
                    #rotated_depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)
                    rotated_depth_img = depth_img
                    # Stray Scanner PNGs are uint16 depth in millimeters.
                    depth_msg = self.bridge.cv2_to_imgmsg(rotated_depth_img, encoding="16UC1")
                    depth_msg.header.stamp = rclpy.time.Time(seconds=timestamp).to_msg()
                    depth_msg.header.frame_id = "camera_depth_frame"
                    # ARKit sceneDepth is aligned to the RGB camera. Scale the
                    # original per-frame K directly to depth dimensions, never
                    # using the already downsampled RGB CameraInfo as the source.
                    # https://github.com/strayrobots/scanner/blob/main/scripts/integrate.py
                    depth_info_msg = create_camera_info(
                        image_info["intrinsics"],
                        depth_msg.header,
                        self.rgb_source_size,
                        (rotated_depth_img.shape[1], rotated_depth_img.shape[0]),
                    )
                    self.depth_pub.publish(depth_msg)
                    self.depth_camera_info_pub.publish(depth_info_msg)

                else:
                    self.get_logger().warn(
                        f"Depth or confidence image missing or invalid for frame {self.current_index}."
                    )

            # Update progress bar
            self.progress_bar.update(1)

            self.current_index += 1


def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) < 3:
        print("Usage: python3 publish_ros2_msgs.py <data_directory> <playback_speed>")
        rclpy.shutdown()
        return

    data_dir = sys.argv[1]
    playback_speed = float(sys.argv[2])
    print(f"playback_speed is {playback_speed}")

    node = StrayScannerDataPublisher(data_dir, playback_speed)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
