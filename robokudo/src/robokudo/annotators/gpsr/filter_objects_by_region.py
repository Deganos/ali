import cv2
import py_trees

import robokudo.annotators.core
import robokudo.defs
import robokudo.types.scene
from robokudo.cas import CASViews
from robokudo.types.annotation import PoseAnnotation
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.semantic_annotations.semantic_annotations import Room


class FilterObjectsByRegion(robokudo.annotators.core.BaseAnnotator):
    """
    Marks every hypothesis that stands in the queried room with a LocationAnnotation.
    The rooms come from the shared world; the result annotators keep only the marked hypotheses.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.world_frame_name = "map"
                self.active_region = ""

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="FilterObjectsByRegion", descriptor=Descriptor()):
        super().__init__(name=name, descriptor=descriptor)
        self.world_frame_name = self.descriptor.parameters.world_frame_name
        self.active_region = self.descriptor.parameters.active_region
        self.world = None

    def update(self):
        if not self.get_cas().contains(CASViews.QUERY):
            return py_trees.common.Status.SUCCESS
        queried_location = self.get_cas().get(CASViews.QUERY).obj.location
        if queried_location == "":
            return py_trees.common.Status.SUCCESS

        # The shared world is fetched once, on the first query that asks for a room.
        # Its root frame is assumed to be the robot's map frame.
        if self.world is None:
            node = ROSNodeRegistry().get()
            self.world = fetch_world_from_service(node)
            # keeps the fetched copy up to date, e.g. with rooms added after the fetch
            WorldSynchronizer(_world=self.world, node=node)
        rooms = [room for room in self.world.get_semantic_annotations_by_type(Room)
                 if room.name.name == queried_location]
        if not rooms:
            # No exception here: an unknown room yields an empty answer instead of a query that never returns.
            self.rk_logger.warning(f"The shared world has no room '{queried_location}', no hypothesis is kept")
            self._draw_region_overlay(queried_location)
            return py_trees.common.Status.SUCCESS
        floor_boxes = list(rooms[0].floor.root.collision.as_bounding_box_collection_in_frame(self.world.root))

        region_annotation = robokudo.types.annotation.LocationAnnotation()
        region_annotation.name = queried_location
        for annotation in self.get_cas().filter_annotations_by_type(robokudo.types.scene.ObjectHypothesis):
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
                pose_map = transform_pose_from_camera_to_world(self.get_cas(), pose_to_transform)
                x, y = pose_map.translation[0], pose_map.translation[1]
                # the floor is flat, so only x and y decide whether the hypothesis stands in the room
                if any(box.min_x <= x <= box.max_x and box.min_y <= y <= box.max_y for box in floor_boxes):
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

























