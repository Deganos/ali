import json

import py_trees
import robokudo.analysis_engine
import robokudo.annotators.core
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.query import GenerateQueryResult
from robokudo.cas import CASViews
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.yolo_annotator import YoloAnnotator

from robokudo.annotators.gpsr.filter_humans_by_object_annotator import \
    FilterHumansByObjectAnnotator
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator

# TODO: find a way, to make it cleaner/ better and mapping every item, that could come (maybe like semmanticannotations)
FURNITURE = {
    'chair':        56,
    'couch':        57,
    'bed':          59,
    'table':        60,
    'dining_table': 60,
    'toilet':       61,
    'tv':           62,
    'microwave':    68,
    'oven':         69,
    'sink':         71,
    'refrigerator': 72,
}

# Optional directional
DIRECTIONS = {'left', 'right', 'front', 'back'}


class YoloFurnitureSetup(robokudo.annotators.core.BaseAnnotator):

    def __init__(self, yolo_params, filter_params):
        super().__init__(name="YoloFurnitureSetup")
        self.yolo_params = yolo_params
        self.filter_params = filter_params
        # fallback when the query carries no direction
        self.default_direction = filter_params.direction

    # TODO: find a way, to make it cleaner/ better
    # Maybe map the Yolo stuff via an annotator and the location in used annotator not here!
    def update(self):
        cas = self.get_cas()
        if cas.contains(CASViews.QUERY):
            query = cas.get(CASViews.QUERY)
            location = query.obj.location
            coco_id = FURNITURE.get(location)
            if coco_id is not None:
                # CRAM's YoloAnnotator has no positive class filter, so every other class is ignored instead
                class_names = json.loads(
                    self.yolo_params.file_in_ros_package(self.yolo_params.class_names_path).read_text())
                self.yolo_params.ignored_class_names = [
                    name for class_id, name in class_names.items() if int(class_id) != coco_id]
            if location:
                self.filter_params.target_type = location
            # optional direction from the query extracted under Attribute
            direction = next((a for a in (query.obj.attribute or []) if a in DIRECTIONS), None)
            self.filter_params.direction = direction if direction else self.default_direction
        return py_trees.common.Status.SUCCESS


class Subtree(robokudo.analysis_engine.SubtreeInterface):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def implementation(self, name=None, direction=None, box_x=None, box_y=None,
                       box_z=None, *args, **kwargs):
        yolo_descriptor = YoloAnnotator.Descriptor()
        yolo_descriptor.parameters.precision_mode = True
        yolo_descriptor.parameters.ros_package_name = 'robokudo_suturo'
        yolo_descriptor.parameters.weights_path = 'weights/yolov8n.pt'
        yolo_descriptor.parameters.class_names_path = 'data/json/id2name_yolov8Model.json'
        yolo_descriptor.parameters.threshold = 0.6

        filter_descriptor = FilterHumansByObjectAnnotator.Descriptor()
        if direction is not None:
            filter_descriptor.parameters.direction = direction
        if box_x is not None:
            filter_descriptor.parameters.box_x = box_x
        if box_y is not None:
            filter_descriptor.parameters.box_y = box_y
        if box_z is not None:
            filter_descriptor.parameters.box_z = box_z

        seq = py_trees.composites.Sequence("subtree_human_near_object", memory=True)
        seq.add_children([
            YoloFurnitureSetup(yolo_descriptor.parameters, filter_descriptor.parameters),
            YoloAnnotator(descriptor=yolo_descriptor),
            HumanAndPoseAnnotator(),
            ClusterPositionAnnotator(),
            FilterHumansByObjectAnnotator(descriptor=filter_descriptor),
            BeliefStateAnnotator(),
            GenerateQueryResult(),
        ])
        return seq
