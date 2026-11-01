import cv2
import numpy as np
import py_trees
from scipy.spatial.transform import Rotation as R

import robokudo.annotators.core
import robokudo.defs
import robokudo.types.scene
from robokudo.cas import CASViews
from robokudo.types.annotation import PoseAnnotation
from robokudo.utils.annotator_helper import transform_pose_from_cam_to_world
from robokudo.utils.module_loader import ModuleLoader


class FilterObjectsByRegion(robokudo.annotators.core.BaseAnnotator):
    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.world_frame_name = "map"
                self.semantic_map_ros_package = "robokudo"
                self.semantic_map_name = "semantic_map_iai_kitchen"  # should be in descriptors/semantic_maps/
                self.active_region = ""

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="FilterObjectsByRegion", descriptor=Descriptor()):
        super().__init__(name=name, descriptor=descriptor)
        self.semantic_map = None
        self.load_semantic_map()
        self.world_frame_name = self.descriptor.parameters.world_frame_name
        self.active_region = self.descriptor.parameters.active_region

    def load_semantic_map(self) -> None:
        module_loader = ModuleLoader()
        self.semantic_map = module_loader.load_semantic_map(self.descriptor.parameters.semantic_map_ros_package,
                                                            self.descriptor.parameters.semantic_map_name)

    def update(self):
        query = None
        if self.get_cas().contains(CASViews.QUERY):
            query = self.get_cas().get(CASViews.QUERY)

        self.load_semantic_map()
        active_regions = self.semantic_map.entries
        if query is None:
            return py_trees.Status.SUCCESS
        else:
            queried_location = query.obj.location
            if queried_location == "":
                return py_trees.Status.SUCCESS
            elif queried_location != self.active_region:
                self.rk_logger.info(f"Setting filter to check for location '{queried_location}'")
                active_regions = dict()
                try:
                    active_regions[queried_location] = self.semantic_map.entries[queried_location]
                except KeyError as ke:
                    raise Exception(f"Couldn't find requested location {queried_location} in semantic map")

        for key, region in active_regions.items():
            if region.frame_id == self.world_frame_name:
                assert (isinstance(region, robokudo.semantic_map.SemanticMapEntry))
                region_position_and_size = {"position": (region.position_x, region.position_y, region.position_z), "size": (region.x_size, region.y_size,region.z_size)}
                quaternion = [region.orientation_x, region.orientation_y, region.orientation_z, region.orientation_w]
                region_name = key
                region_annotation = robokudo.types.annotation.LocationAnnotation()
                region_annotation.name = region_name

                annotations = self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis)
                for annotation in annotations:
                    for oh_annotation in annotation.annotations:
                        if isinstance(oh_annotation, robokudo.types.annotation.PoseAnnotation):
                            pose_to_transform = oh_annotation
                        elif isinstance(oh_annotation, robokudo.types.annotation.PositionAnnotation):
                            # If PositionAnnotation does not have a rotation, then wrap it
                            pose_to_transform = robokudo.types.annotation.PoseAnnotation()
                            pose_to_transform.rotation = [0.0, 0.0, 0.0, 1.0]
                            pose_to_transform.translation = oh_annotation.translation
                            pose_to_transform.source = oh_annotation.source
                        else:
                            continue
                        pose_map = transform_pose_from_cam_to_world(self.get_cas(), pose_to_transform)
                        position = [pose_map.translation[0], pose_map.translation[1], pose_map.translation[2]]
                        if self.position_in_region(position, region_position_and_size, quaternion):
                            annotation.annotations.append(region_annotation)
                            break

        self._draw_region_overlay(queried_location)
        return py_trees.common.Status.SUCCESS

    def _draw_region_overlay(self, queried_location):
        img = self.get_cas().get(CASViews.COLOR_IMAGE)
        if img is None:
            return
        img = img.copy()
        for hh in self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis):
            try:
                r = hh.roi.roi
                x, y, w, h = int(r.pos.x), int(r.pos.y), int(r.width), int(r.height)
            except AttributeError:
                continue
            in_room = any(
                isinstance(a, robokudo.types.annotation.LocationAnnotation) and a.name == queried_location
                for a in hh.annotations
            )
            color = (0, 200, 0) if in_room else (0, 0, 255)
            if not in_room:
                overlay = img.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, "in room" if in_room else "wrong room",
                        (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        self.get_annotator_output_struct().set_image(img)

    def rotate_point_quaternion(self, point, quaternion, center):
        """
        Rotates a point around a center by a given quaternion.
        """
        cx, cy, cz = center
        px, py, pz = point

        # Translate point to origin
        translated_point = np.array([px - cx, py - cy, pz - cz])

        # Create rotation object from quaternion
        r = R.from_quat(quaternion)

        # Rotate point
        rotated_point = r.apply(translated_point)

        # Translate point back
        qx, qy, qz = rotated_point + np.array([cx, cy, cz])
        return qx, qy, qz

    def position_in_region(self, position_of_object, region_position_and_size, quaternion):
        """
        Check if the position_of_object is within the rotated region defined by region_position_and_size.
        region_position_and_size should be a dictionary with keys 'position' and 'size'.
        The 'position' is the center of the region and 'size' is the width, height, and depth.
        The quaternion represents the rotation.
        """
        region_center = region_position_and_size["position"]
        region_size = region_position_and_size["size"]

        # Translate the object's position relative to the region's center
        translated_position = np.array(position_of_object) - np.array(region_center)

        # Create rotation object from quaternion and get the inverse rotation
        r = R.from_quat(quaternion)
        inverse_r = r.inv()

        # Rotate the object's position by the inverse rotation
        local_position = inverse_r.apply(translated_position)

        # Define the half-sizes of the region
        half_size_x = region_size[0] / 2
        half_size_y = region_size[1] / 2
        half_size_z = region_size[2] / 2

        # Check if the local position is within the region's size boundaries
        if (-half_size_x <= local_position[0] <= half_size_x and
                -half_size_y <= local_position[1] <= half_size_y and
                -half_size_z <= local_position[2] <= half_size_z):
            return True
        return False

























