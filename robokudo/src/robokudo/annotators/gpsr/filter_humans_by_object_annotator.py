import numpy as np
import py_trees
import open3d as o3d

import robokudo.annotators.core
import robokudo.types.annotation
from robokudo.types.annotation import PoseAnnotation
from robokudo.types.scene import HumanHypothesis, ObjectHypothesis
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world


class FilterHumansByObjectAnnotator(robokudo.annotators.core.BaseAnnotator):
    """
    Removes humans from the CAS that are not near a specific piece of a furniture.

    Without a direction the search region is a symmetric box; a direction restricts it
    to one half (left/right/front/back from the robot's point of view, projected onto
    the world ground plane so a tilted camera head does not distort the directions).
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.box_x = 1.5
                self.box_y = 1.5
                self.box_z = 2.5
                self.direction = None  # None, 'left', 'right', 'front', 'back'
                self.target_type = None

        parameters = Parameters()

    def __init__(self, name="FilterHumansByObjectAnnotator", descriptor=Descriptor()):
        super(FilterHumansByObjectAnnotator, self).__init__(name, descriptor)

    def inside_box(
        self,
        human_pos,
        furniture_pos,
        box_size,
        direction=None,
        right=None,
        forward=None,
    ):
        """Returns True if human is in range of the given furniture. Without a direction
        this is the symmetric world-axis box; a direction keeps only one half-space along
        the robot's ground-plane axes (right > 0 = robot right, forward > 0 = behind)."""
        delta = np.array(human_pos) - np.array(furniture_pos)
        if not direction:
            return np.all(np.abs(delta) <= np.array(box_size) / 2.0)

        box_x, box_y, box_z = box_size
        if abs(delta[2]) > box_z / 2.0:
            return False
        lateral = float(np.dot(delta, right))
        longitudinal = float(np.dot(delta, forward))
        if direction == "right":
            return 0.0 <= lateral <= box_x and abs(longitudinal) <= box_y / 2.0
        if direction == "left":
            return -box_x <= lateral <= 0.0 and abs(longitudinal) <= box_y / 2.0
        if direction == "back":
            return 0.0 <= longitudinal <= box_y and abs(lateral) <= box_x / 2.0
        if direction == "front":
            return -box_y <= longitudinal <= 0.0 and abs(lateral) <= box_x / 2.0
        return np.all(np.abs(delta) <= np.array(box_size) / 2.0)

    def _robot_ground_basis(self, cas):
        """
        Right/forward unit vectors on the world ground plane, from the camera pose.

        Returns (None, None) if the camera looks (nearly) straight up/down.
        """

        def to_world(point):
            pose = PoseAnnotation()
            pose.translation = list(point)
            pose.rotation = [0.0, 0.0, 0.0, 1.0]
            return np.array(transform_pose_from_camera_to_world(cas, pose).translation)

        origin = to_world([0.0, 0.0, 0.0])
        right = to_world([1.0, 0.0, 0.0]) - origin  # camera image-right axis
        forward = to_world([0.0, 0.0, 1.0]) - origin  # camera optical (forward) axis

        def project_ground(v):
            flat = np.array([v[0], v[1], 0.0])
            norm = np.linalg.norm(flat)
            return flat / norm if norm > 1e-6 else None

        return project_ground(right), project_ground(forward)

    def update(self):
        cas = self.get_cas()
        params = self.descriptor.parameters
        target_type = params.target_type
        box_size = [params.box_x, params.box_y, params.box_z]
        direction = params.direction
        furniture_world_positions = []
        furniture_cam_positions = []
        furniture_clouds = []
        human_viz_data = []
        kept = 0

        if not target_type:
            self.rk_logger.warning("No target type set")
            return py_trees.common.Status.SUCCESS

        # getting every human and every furniture of the given type (yolo filtered it already)
        all_hypotheses = cas.filter_annotations_by_type(ObjectHypothesis)
        humans = [h for h in all_hypotheses if isinstance(h, HumanHypothesis)]
        furniture_list = [
            obj for obj in all_hypotheses if not isinstance(obj, HumanHypothesis)
        ]

        # robots ground-plane axes, only needed for directional filtering
        right, forward = None, None
        if direction:
            right, forward = self._robot_ground_basis(cas)
            if right is None or forward is None:
                self.rk_logger.warning(
                    "Degenerate camera orientation, ignoring direction"
                )
                direction = None

        # getting the position of the furniture relativly to the world
        for obj in furniture_list:
            position = cas.filter_by_type(
                robokudo.types.annotation.PositionAnnotation, obj.annotations
            )
            if not position:
                continue
            pose = PoseAnnotation()
            pose.translation = position[0].translation
            pose.rotation = [0.0, 0.0, 0.0, 1.0]
            furniture_cam_positions.append(pose.translation)
            furniture_clouds.append(self._cluster_cloud(obj))
            furniture_world_positions.append(
                transform_pose_from_camera_to_world(cas, pose).translation
            )

        if not furniture_world_positions:
            self.rk_logger.info(
                f"No '{target_type}' detected, removing all {len(humans)} humans"
            )
            for human in humans:
                cas.annotations.remove(human)
            return py_trees.common.Status.SUCCESS

        # getting the position of every human and checking, if it is near (on the requested side)
        for human in humans:
            position = cas.filter_by_type(
                robokudo.types.annotation.PositionAnnotation, human.annotations
            )
            if not position:
                cas.annotations.remove(human)
                continue

            pose = PoseAnnotation()
            pose.translation = position[0].translation
            pose.rotation = [0.0, 0.0, 0.0, 1.0]
            world_pos = transform_pose_from_camera_to_world(cas, pose).translation

            near_furniture = False
            for furniture_pos in furniture_world_positions:
                if self.inside_box(
                    world_pos, furniture_pos, box_size, direction, right, forward
                ):
                    near_furniture = True
                    break

            if near_furniture:
                tag = robokudo.types.annotation.LocationAnnotation()
                tag.name = target_type
                human.annotations.append(tag)
                kept += 1
                human_viz_data.append(
                    (pose.translation, True, self._cluster_cloud(human))
                )
            else:
                cas.annotations.remove(human)
                human_viz_data.append(
                    (pose.translation, False, self._cluster_cloud(human))
                )

        side = direction if direction else "any side"
        self.rk_logger.info(
            f"Kept {kept}/{len(humans)} humans near '{target_type}' ({side})"
        )
        # calling the vizualization step for debug purposes
        self._visualize(
            furniture_cam_positions,
            furniture_clouds,
            human_viz_data,
            box_size,
            direction,
        )
        return py_trees.common.Status.SUCCESS

    def _visualize(
        self,
        furniture_cam_positions,
        furniture_clouds,
        human_viz_data,
        box_size,
        direction,
    ):
        """Green box = search region. Orange cluster cloud/marker = furniture,
        blue = kept human, red = removed human. The coloured cluster clouds show
        what each marker actually is, so the scene is readable without the legend.
        """
        box_x, box_y, box_z = box_size
        geometries = []
        for cam_pos, furniture_cloud in zip(furniture_cam_positions, furniture_clouds):
            center = np.array(cam_pos)
            if direction:
                # camera axes: x=right (box_x), y=down (box_z), z=forward (box_y)
                half = np.array([box_x / 2.0, box_z / 2.0, box_y / 2.0])
                min_c, max_c = center - half, center + half
                if direction == "right":
                    min_c[0], max_c[0] = center[0], center[0] + box_x
                elif direction == "left":
                    min_c[0], max_c[0] = center[0] - box_x, center[0]
                elif direction == "back":
                    min_c[2], max_c[2] = center[2], center[2] + box_y
                elif direction == "front":
                    min_c[2], max_c[2] = center[2] - box_y, center[2]
            else:
                half = np.array(box_size) / 2.0
                min_c, max_c = center - half, center + half
            box = o3d.geometry.AxisAlignedBoundingBox(min_c, max_c)
            box.color = [0.0, 1.0, 0.0]
            geometries.append(box)
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
            marker.translate(center)
            marker.paint_uniform_color([1.0, 0.6, 0.0])
            geometries.append(marker)
            self._append_cloud(geometries, furniture_cloud, [1.0, 0.6, 0.0])
        for cam_pos, was_kept, human_cloud in human_viz_data:
            if cam_pos is None:
                continue
            color = [0.0, 0.5, 1.0] if was_kept else [1.0, 0.2, 0.2]
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
            sphere.translate(np.array(cam_pos))
            sphere.paint_uniform_color(color)
            geometries.append(sphere)
            self._append_cloud(geometries, human_cloud, color)
        self.get_annotator_output_struct().set_geometries(geometries)

    def _cluster_cloud(self, hypothesis):
        """
        The Open3D point cloud of a hypothesis cluster, or None if it has none.

        Mirrors how the ClusterPositionAnnotator reads cluster points.
        """
        if getattr(hypothesis, "points", None) is not None:
            return hypothesis.points
        clouds = self.get_cas().filter_by_type(
            robokudo.types.annotation.CloudAnnotation, hypothesis.annotations
        )
        return clouds[0].points if clouds else None

    def _append_cloud(self, geometries, cloud, color):
        """
        Add a colour-painted copy of a cluster cloud to the geometry list.
        """
        if cloud is None or not cloud.has_points():
            return
        painted = o3d.geometry.PointCloud()
        painted.points = o3d.utility.Vector3dVector(np.asarray(cloud.points))
        painted.paint_uniform_color(color)
        geometries.append(painted)
