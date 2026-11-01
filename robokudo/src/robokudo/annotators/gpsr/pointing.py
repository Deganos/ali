from enum import Enum
from timeit import default_timer

from rclpy.node import Node
import geometry_msgs.msg
import numpy
import numpy as np
import open3d as o3d
import py_trees
import py_trees.common
from visualization_msgs.msg import Marker
import rclpy
import visualization_msgs.msg
import robokudo
import robokudo.annotators.core
import robokudo.cas
import robokudo.types.annotation
import robokudo.types.human
import robokudo.types.scene
import robokudo.types.tf
import robokudo.utils.cv_helper
import robokudo.utils.decorators
import robokudo.utils.error_handling
import robokudo.utils.math_helper
import robokudo.utils.o3d_helper
import robokudo.utils.transform
from robokudo.annotators.core import BaseAnnotator

from robokudo.types.annotation import PoseAnnotation, PositionAnnotation
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world, transform_pose_from_world_to_camera
from robokudo.utils.math_helper import compute_direction_vector_angle
from robokudo.utils.transform import get_rotation_matrix_from_q
from robokudo.utils.module_loader import ModuleLoader


class Arm(Enum):
    LEFT = 0
    RIGHT = 1

    def __str__(self):
        return self.name.lower()


def azimuth_angle(direction_vector: np.ndarray) -> float:
    """
    Computes the azimuth angle (the direction) of a direction vector in degrees.

    :param direction_vector: A 3D vector (numpy array) representing the direction.
    :return: Azimuth angle in degrees.
    """
    x, y, z = direction_vector
    rotated_x = y
    rotated_y = -x

    return np.degrees(np.arctan2(rotated_y, rotated_x))


class HumanPointingAnnotator(robokudo.annotators.core.BaseAnnotator):
    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                # A list of spheres defined by triple (name, center, radius). In Cam coordinates.
                # These spheres will be used to check if something is 'pointed at'
                self.pointing_target_structures = []
                self.pointing_target_structures_type = "box" # or "sphere"
                # Starting from the elbow, how long shall the line for pointing checks be? In meters.
                self.pointing_length_meters = 5.0
                self.raise_on_failure = True
                # max elevation of the forearm above horizontal to count as pointing, works with wrist-below-shoulder check
                # TODO: fine-tune value based on real pointing vs waving observations
                self.max_pointing_angle = 45.0
                self.min_arm_extension = 0.0   # minimum wrist-to-torso-center distance (meters)

                # If True, only run on HHs that were already tagged as "Pointing" or "No gesture".
                self.known = False

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="HumanPointingAnnotator", descriptor=Descriptor()):
        super().__init__(name, descriptor)
        self.node = Node('human_pointing_annotator')
        self.pointing_marker_publisher = self.node.create_publisher(
            visualization_msgs.msg.MarkerArray,
            'pointing_marker',  # relativ: <node_name>/pointing_marker
            10
        )
        self.threshold_marker_publisher = self.node.create_publisher(
            visualization_msgs.msg.MarkerArray,
            'pointing_threshold_marker',
            10
        )
        self.possible_structures = {"box": o3d.geometry.TriangleMesh.create_box, "sphere": o3d.geometry.TriangleMesh.create_sphere,}
        if descriptor.parameters.pointing_target_structures_type not in self.possible_structures:
            raise ValueError(f"Unknown pointing target structure type: {descriptor.parameters.pointing_target_structures_type}. "
                             f"Possible values are: {list(self.possible_structures.keys())}")
        self.pointing_target_structures_type = descriptor.parameters.pointing_target_structures_type

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self):
        start_timer = default_timer()

        self.color = self.get_cas().get(robokudo.cas.CASViews.COLOR_IMAGE)
        cloud = self.get_cas().get(robokudo.cas.CASViews.CLOUD)
        all_vis_geometries = [cloud]

        human_hypotheses = self.get_cas().filter_annotations_by_type(robokudo.types.scene.HumanHypothesis)

        if self.descriptor.parameters.known:
            # only process humans the HumanActivityAnnotator already tagged as pointing or no gesture
            human_hypotheses = [
                hh for hh in human_hypotheses
                if any(
                    isinstance(a, robokudo.types.human.ActivityAnnotation)
                    and a.source == "HumanActivityAnnotator"
                    and a.type in ("Pointing", "No gesture")
                    for a in hh.annotations
                )
            ]
            if len(human_hypotheses) == 0:
                self.feedback_message = "known=True but no HH tagged as Pointing, skipping"
                return py_trees.common.Status.SUCCESS
        elif len(human_hypotheses) == 0:
            raise Exception("No humans detected - Can't estimate pointing activity")

        one_human_handled = False
        for human_hypothesis in human_hypotheses:
            keypoint_annotations = robokudo.cas.CAS.filter_by_type(robokudo.types.human.KeypointAnnotation,
                                                                   human_hypothesis.annotations)
            keypoint_3d_annotations = [a for a in keypoint_annotations if
                                       a.kind == robokudo.types.human.KeypointKind.POINT_CLOUD]
            if len(keypoint_3d_annotations) != 1:
                self.rk_logger.warning("Couldn't find 3D Keypoints in the CAS")
                continue

            keypoint_3d_annotation = keypoint_3d_annotations[0]

            # Calculate the centroid for each of the keypoints of interest
            left_elbow_cloud = keypoint_3d_annotation.keypoints[7].points
            left_wrist_cloud = keypoint_3d_annotation.keypoints[9].points

            right_elbow_cloud = keypoint_3d_annotation.keypoints[8].points
            right_wrist_cloud = keypoint_3d_annotation.keypoints[10].points

            if len(left_elbow_cloud.points) == 0 or len(left_wrist_cloud.points) == 0:
                self.rk_logger.warning("One of the relevant left keypoint clouds was empty - Can't estimate pointing")
                continue

            if len(right_elbow_cloud.points) == 0 or len(right_wrist_cloud.points) == 0:
                self.rk_logger.warning("One of the relevant right keypoint clouds was empty - Can't estimate pointing")
                continue

            one_human_handled = True

            le_centroid, _ = left_elbow_cloud.compute_mean_and_covariance()
            lw_centroid, _ = left_wrist_cloud.compute_mean_and_covariance()
            re_centroid, _ = right_elbow_cloud.compute_mean_and_covariance()
            rw_centroid, _ = right_wrist_cloud.compute_mean_and_covariance()

            # if both wrists are close to the torso center the person is not extending an arm
            if self.descriptor.parameters.min_arm_extension > 0.0:
                torso_center = self.torso_center(keypoint_3d_annotation)
                if torso_center is not None:
                    lw_dist = numpy.linalg.norm(lw_centroid - torso_center)
                    rw_dist = numpy.linalg.norm(rw_centroid - torso_center)
                    if lw_dist < self.descriptor.parameters.min_arm_extension and \
                            rw_dist < self.descriptor.parameters.min_arm_extension:
                        self.rk_logger.debug(
                            f"HH {human_hypothesis.id}: both wrists within "
                            f"{self.descriptor.parameters.min_arm_extension}m of torso "
                            f"(lw={lw_dist:.2f}m, rw={rw_dist:.2f}m) skipping"
                        )
                        continue

            # Calculate the line used for visualization and intersection computation
            pointing_direction_left = numpy.subtract(lw_centroid, le_centroid)
            pointing_direction_right = numpy.subtract(rw_centroid, re_centroid)
            unit_pointing_direction_left = pointing_direction_left / numpy.linalg.norm(pointing_direction_left)
            unit_pointing_direction_right = pointing_direction_right / numpy.linalg.norm(pointing_direction_right)

            # Scale direction by defined length
            pointing_line_left = unit_pointing_direction_left * self.descriptor.parameters.pointing_length_meters
            pointing_line_right = unit_pointing_direction_right * self.descriptor.parameters.pointing_length_meters

            pointing_end_left = numpy.add(pointing_line_left, le_centroid)
            pointing_end_right = numpy.add(pointing_line_right, re_centroid)

            # Transform all structures from world to cam coordinates
            updated_structures = {}
            for structure in self.descriptor.parameters.pointing_target_structures:
                pose = PoseAnnotation()
                pose.translation = list(structure[1])  # center of the structure
                if self.pointing_target_structures_type == "sphere":
                    pose.rotation = (1.0, 0.0, 0.0, 0.0)  # No rotation for spheres
                else:
                    pose.rotation = structure[2]

                transformed_pose = transform_pose_from_world_to_camera(self.get_cas(), pose)

                new_structure_pose = (structure[0], (transformed_pose.translation[0], transformed_pose.translation[1], transformed_pose.translation[2]),
                                      (transformed_pose.rotation[0], transformed_pose.rotation[1], transformed_pose.rotation[2], transformed_pose.rotation[3]),
                                      (structure[-1]))


                updated_structures[structure[0]] = new_structure_pose

            # Get all intersections between defined structures and pointing line
            intersections_left = robokudo.utils.math_helper.intersecting_cuboids(lw_centroid, pointing_end_left,
                                                                                 updated_structures.values(),)
            intersections_right = robokudo.utils.math_helper.intersecting_cuboids(rw_centroid, pointing_end_right,
                                                                                  updated_structures.values(),)

            intersections, arm = self.choose_arm(intersections_left, intersections_right,
                                                 pointing_direction_left, pointing_direction_right)

            # wrist above shoulder means raising or waving (camera Y points down so smaller Y = higher)
            left_shoulder, right_shoulder = self.shoulder_centroids(keypoint_3d_annotation)
            if left_shoulder is not None and right_shoulder is not None:
                chosen_shoulder = left_shoulder if arm == Arm.LEFT else right_shoulder
                chosen_wrist = lw_centroid if arm == Arm.LEFT else rw_centroid
                if chosen_wrist[1] < chosen_shoulder[1]:
                    self.rk_logger.debug(
                        f"HH {human_hypothesis.id}: wrist above shoulder on {arm} arm not pointing"
                    )
                    continue

            # set relevant variables based on the chosen arm
            pointing_direction = pointing_direction_left if arm == Arm.LEFT else pointing_direction_right
            pointing_end = pointing_end_left if arm == Arm.LEFT else pointing_end_right
            elbow_centroid = le_centroid if arm == Arm.LEFT else re_centroid
            wrist_centroid = lw_centroid if arm == Arm.LEFT else rw_centroid

            # Create a pose annotation so you can transform the cam pose to world pose
            pose_elbow = robokudo.types.annotation.PoseAnnotation()
            pose_elbow.translation = list(elbow_centroid)

            rotation_matrix = robokudo.utils.transform.get_rotation_matrix_from_direction_vector(pointing_direction)
            pose_elbow.rotation = list(robokudo.utils.transform.get_quaternion_from_rotation_matrix(rotation_matrix))

            transform_elbow = transform_pose_from_camera_to_world(self.get_cas(), pose_elbow)

            pointed_at_objects = []
            object_hypotheses = []
            if intersections:
                self.rk_logger.debug(f"Pointing intersected {intersections}")
                for _, structure_name, _, _, _ in intersections:
                    pointed_at_position = PoseAnnotation()
                    pointed_at_position.source = type(self).__name__
                    pointed_at_position.translation = updated_structures[structure_name][1]
                    pointed_at_position.rotation = updated_structures[structure_name][2]

                    object_hypothesis = robokudo.types.scene.ObjectHypothesis()
                    object_hypothesis.type = structure_name
                    object_hypothesis.source = type(self).__name__
                    object_hypothesis.annotations.append(pointed_at_position)
                    object_hypotheses.append(object_hypothesis)

                    self.rk_logger.debug(f"Pointing hit {structure_name}")
                    pointed_at_objects.append(structure_name)

                    activity = robokudo.types.human.ActivityAnnotation()
                    activity.type = 'pointing'
                    activity.source = type(self).__name__

                    activity.interaction_with = object_hypothesis
                    human_hypothesis.annotations.append(activity)
            else:
                self.rk_logger.debug(f"No intersections found")
                # If no intersections were found, we still want to create an activity annotation if an arm suggests pointing
                self.check_pointing_without_intersections(human_hypothesis, pointing_direction,
                                                          self.descriptor.parameters.max_pointing_angle, transform_elbow)

            self.get_cas().annotations.extend(object_hypotheses)


            all_vis_geometries = self.create_annotator_output_struct(all_vis_geometries, elbow_centroid, wrist_centroid,
                                                                     pointing_end, pointed_at_objects, updated_structures)

            # Publish pointing vector marker
            marker_msg = self.generate_marker(transform_elbow.translation, transform_elbow.rotation, "map")
            marker_array_msg = visualization_msgs.msg.MarkerArray()
            marker_array_msg.markers.append(marker_msg)
            self.pointing_marker_publisher.publish(marker_array_msg)

            # Update semantic map visualization colors if intersections exist
            try:
                if len(pointed_at_objects) > 0:
                    semantic_map = ModuleLoader().load_semantic_map("robokudo_robocup_gmah", "suturo_lab")
                    semantic_map.publish_visualization_markers(highlighted=set(pointed_at_objects))
            except Exception as e:
                self.rk_logger.debug(f"Semantic map highlight update skipped: {e}")

            # Legacy single-human mode (carry-my-luggage): stop after first handled person.
            # Multi-human mode (known=True, GPSR) processes every tagged HH so each gets a direction.
            if one_human_handled and not self.descriptor.parameters.known:
                break

        if not one_human_handled:
            if self.descriptor.parameters.known:
                self.feedback_message = "No pointing detected"
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.SUCCESS

        self.get_annotator_output_struct().set_geometries(all_vis_geometries)

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.common.Status.SUCCESS

    def torso_center(self, keypoint_3d_annotation):
        """Torso center from shoulders + hips. Falls back to shoulders only if hips unavailable."""
        try:
            left_shoulderloud = keypoint_3d_annotation.keypoints[5].points
            right_shoulderloud = keypoint_3d_annotation.keypoints[6].points
            if len(left_shoulderloud.points) == 0 or len(right_shoulderloud.points) == 0:
                return None
            left_shoulder, _ = left_shoulderloud.compute_mean_and_covariance()
            right_shoulder, _ = right_shoulderloud.compute_mean_and_covariance()
            points = [left_shoulder, right_shoulder]
            try:
                left_hiploud = keypoint_3d_annotation.keypoints[11].points
                right_hiploud = keypoint_3d_annotation.keypoints[12].points
                if len(left_hiploud.points) > 0 and len(right_hiploud.points) > 0:
                    left_hip, _ = left_hiploud.compute_mean_and_covariance()
                    right_hip, _ = right_hiploud.compute_mean_and_covariance()
                    points.extend([left_hip, right_hip])
            except (KeyError, IndexError):
                pass
            return sum(points) / len(points)
        except (KeyError, IndexError):
            return None

    def shoulder_centroids(self, keypoint_3d_annotation):
        """Returns (left_shoulder, right_shoulder) centroids, or (None, None) if unavailable."""
        try:
            left_shoulderloud = keypoint_3d_annotation.keypoints[5].points
            right_shoulderloud = keypoint_3d_annotation.keypoints[6].points
            if len(left_shoulderloud.points) == 0 or len(right_shoulderloud.points) == 0:
                return None, None
            left_shoulder, _ = left_shoulderloud.compute_mean_and_covariance()
            right_shoulder, _ = right_shoulderloud.compute_mean_and_covariance()
            return left_shoulder, right_shoulder
        except (KeyError, IndexError):
            return None, None

    def check_pointing_without_intersections(self, human_hypothesis, pointing_direction, maximum_pointing_angle, transform_elbow):
        """
        If no intersections were found, we still want to create an activity annotation if an arm suggests pointing.
        We also add a 45-degree threshold visualization around the elbow pointing direction.

        :param human_hypothesis: The human hypothesis to add the activity annotation to
        :param pointing_direction: The direction vector of the pointing arm
        :param maximum_pointing_angle: The maximum angle from vertical to consider as pointing
        :param transform_elbow: The transformed elbow pose in world coordinates
        :return: True if no intersections are found for both arms, False otherwise
        """
        pointing_angle = compute_direction_vector_angle(direction_vector=pointing_direction)
        if pointing_angle < maximum_pointing_angle:
            activity = robokudo.types.human.ActivityAnnotation()
            azimuth = azimuth_angle(pointing_direction)
            if azimuth < 0:
                activity.type = 'pointing left'
            elif azimuth >= 0:
                activity.type = 'pointing right'
            activity.source = type(self).__name__
            activity.interaction_with = None
            human_hypothesis.annotations.append(activity)

        threshold_markers = self.generate_pointing_threshold_markers(transform_elbow.translation, transform_elbow.rotation,
                                                                     "map", self.descriptor.parameters.max_pointing_angle)
        threshold_marker_array = visualization_msgs.msg.MarkerArray()
        threshold_marker_array.markers.extend(threshold_markers)
        self.threshold_marker_publisher.publish(threshold_marker_array)

    def choose_arm(self, intersections_left, intersections_right, direction_vector_left, direction_vector_right):
        """
        Choose the arm based on the intersections found.
        If both arms have intersections, choose the one which points higher.
        If no intersections, choose the arm with the higher wrist centroid.

        :param intersections_left: List of intersections for the left arm
        :param intersections_right: List of intersections for the right arm
        :param direction_vector_left: The pointing vector of the left arm
        :param direction_vector_right: The pointing vector of the right arm
        :return: Tuple of intersections and the chosen arm
        """

        pointing_angle_left = compute_direction_vector_angle(direction_vector=direction_vector_left)
        pointing_angle_right = compute_direction_vector_angle(direction_vector=direction_vector_right)

        if not intersections_left and not intersections_right:
            self.rk_logger.warning("No pointing intersections found for either arm")
            if pointing_angle_left > pointing_angle_right:
                self.rk_logger.debug("Right arm points higher than left arm (no intersections)")
                return [], Arm.RIGHT
            else:
                self.rk_logger.debug("Left arm points higher than right arm (no intersections)")
                return [], Arm.LEFT

        if intersections_left and not intersections_right:
            self.rk_logger.debug("Only left arm has pointing intersections")
            return intersections_left, Arm.LEFT

        if not intersections_left and intersections_right:
            self.rk_logger.debug("Only right arm has pointing intersections")
            return intersections_right, Arm.RIGHT

        # If both arms have intersections, choose the one which points higher
        if pointing_angle_left > pointing_angle_right:
            # Left arm points higher
            self.rk_logger.debug("Left arm points higher than right arm")
            return intersections_left, Arm.LEFT
        else:
            # Right arm points higher
            self.rk_logger.debug("Right arm points higher than left arm")
            return intersections_right, Arm.RIGHT

    def create_annotator_output_struct(self, all_vis_geometries, elbow_centroid, wrist_centroid, pointing_end, pointed_at_objects,
                                       updated_structures):
        """
        Create the geometries for visualization in Open3D and return them.

        :param all_vis_geometries: List of geometries to visualize
        :param elbow_centroid: Centroid of the elbow
        :param wrist_centroid: Centroid of the wrist
        :param pointing_end: End point of the pointing line
        :param pointed_at_objects: List of objects that were pointed at
        :param updated_structures: List of updated structures with their poses

        :return: List of geometries for visualization
        """

        # Left Elbow
        centroid_structure = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
        centroid_structure.paint_uniform_color([1, 0, 0])
        centroid_structure.translate(elbow_centroid)
        all_vis_geometries.append(centroid_structure)

        # Left Wrist
        centroid_structure = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)  # in meters
        centroid_structure.paint_uniform_color([0, 1, 0])
        centroid_structure.translate(wrist_centroid)
        all_vis_geometries.append(centroid_structure)

        # Pointing line
        all_vis_geometries.append(
            robokudo.utils.o3d_helper.create_line_for_visualization(elbow_centroid,
                                                                    pointing_end,
                                                                    [1, 0, 0]))

        # End of pointing line
        centroid_structure = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)  # in meters
        centroid_structure.paint_uniform_color([1, 0, 0])
        centroid_structure.translate(pointing_end)
        all_vis_geometries.append(centroid_structure)

        # All configured spheres
        for structure in updated_structures.values():
            if self.pointing_target_structures_type == "sphere":
                name, center, _,  radius = structure
                centroid_structure = o3d.geometry.TriangleMesh.create_sphere(radius=radius)  # in meters
            else:
                name, center, orientation, size = structure
                centroid_structure = o3d.geometry.TriangleMesh.create_box(width=size[0], height=size[1], depth=size[2])
                rotation = get_rotation_matrix_from_q(orientation)
                centroid_structure.rotate(rotation)

            if name in pointed_at_objects:
                centroid_structure.paint_uniform_color([0, 1, 1])
            else:
                centroid_structure.paint_uniform_color([0, 0, 1])
            centroid_structure.translate(center)
            all_vis_geometries.append(centroid_structure)

        return all_vis_geometries

    def generate_marker(self, origin, target, frame):
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = frame
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.id = 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose.position.x = origin[0]
        marker.pose.position.y = origin[1]
        marker.pose.position.z = origin[2]
        marker.pose.orientation.x = target[0]
        marker.pose.orientation.y = target[1]
        marker.pose.orientation.z = target[2]
        marker.pose.orientation.w = target[3]

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9

        marker.scale.x = float(self.descriptor.parameters.pointing_length_meters)
        marker.scale.y = float(0.1)
        marker.scale.z = float(0.1)
        marker.ns = f"Pointing Marker-{self.node.get_name()}"
        return marker

    def generate_pointing_threshold_markers(self, origin, orientation_q, frame, max_pointing_angle):
        """
        Visualize the 45-degree threshold cone relative to the floor (z-up), not relative to the arm.
        We draw:
        - A horizontal circle (parallel to floor) at height `length` above the elbow, radius = length * tan(45°).
        - 32 red boundary rays from the elbow to opposite points on that circle, representing 45° from vertical.
        This makes it obvious: the pointing direction must be outside this circle to be accepted as pointing.
        """
        # Parameters
        length = self.descriptor.parameters.pointing_length_meters
        radius = length * numpy.tan(numpy.deg2rad(max_pointing_angle))
        num_segments = 32

        # World-up axis (map frame assumed z-up)
        up = numpy.array([0.0, 0.0, 1.0], dtype=float)

        # Build an arbitrary orthonormal basis (u, v) in the horizontal plane
        # Choose x as primary and y as secondary
        u = numpy.array([1.0, 0.0, 0.0], dtype=float)
        v = numpy.array([0.0, 1.0, 0.0], dtype=float)

        origin_np = numpy.array(origin, dtype=float)
        # origin is elbow position, length * up is the vertical line from elbow. This way, visualization depends on th
        # pointing_length_meters only, not on the actual arm orientation
        circle_center = origin_np - up * length

        # Circle points to draw the boundaries in
        points = []
        for i in range(num_segments + 1):
            # Angle around the circle
            # 2.0 * pi * i equals one full circle (360 degrees)
            # this result / num_segments gives the angle step for each segment
            theta = 2.0 * numpy.pi * i / num_segments

            # Compute the 3D coordinates of the point on the circle.
            # radius * (cos(theta) * u + sin(theta) * v) gives the point on the circle in the horizontal plane around origin
            # u and v are the basis vectors in the horizontal plane
            # Adding circle_center moves the circle to the correct position in space
            point = circle_center + radius * (numpy.cos(theta) * u + numpy.sin(theta) * v)
            p = geometry_msgs.msg.Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
            points.append(p)

        stamp = self.node.get_clock().now().to_msg()
        ns = f"Pointing Threshold-{self.node.get_name()}"

        # Marker for circle boundary
        circle_marker = visualization_msgs.msg.Marker()
        circle_marker.header.frame_id = frame
        circle_marker.header.stamp = stamp
        circle_marker.ns = ns
        circle_marker.id = 2
        circle_marker.type = visualization_msgs.msg.Marker.LINE_STRIP
        circle_marker.action = visualization_msgs.msg.Marker.ADD
        circle_marker.scale.x = 0.02
        circle_marker.color.r = 1.0
        circle_marker.color.g = 0.65
        circle_marker.color.b = 0.0
        circle_marker.color.a = 0.9
        circle_marker.points = points

        # Two boundary rays: opposite points on the circle (0 and 180 degrees)
        rays_marker = visualization_msgs.msg.Marker()
        rays_marker.header.frame_id = frame
        rays_marker.header.stamp = stamp
        rays_marker.ns = ns
        rays_marker.id = 3
        rays_marker.type = visualization_msgs.msg.Marker.LINE_LIST
        rays_marker.action = visualization_msgs.msg.Marker.ADD
        rays_marker.scale.x = 0.02
        rays_marker.color.r = 1.0
        rays_marker.color.g = 0.0
        rays_marker.color.b = 0.0
        rays_marker.color.a = 0.9
        origin_point = geometry_msgs.msg.Point(x=float(origin_np[0]), y=float(origin_np[1]), z=float(origin_np[2]))
        # rays_marker.points = [origin_point, boundary_points[0], origin_point, boundary_points[1]]
        rays_marker.points = []
        for p in points:
            rays_marker.points.append(origin_point)
            rays_marker.points.append(p)

        return [circle_marker, rays_marker]
