import py_trees.composites

import robokudo.analysis_engine
import robokudo.pipeline
from robokudo.annotators.cluster_pose_bb import ClusterPoseBBAnnotator
from robokudo.annotators.outlier_removal_objecthypothesis import OutlierRemovalOnObjectHypothesisAnnotator
from robokudo_suturo.annotators.cutlery_handle_annotator import CutleryHandleAnnotator
from robokudo_suturo.annotators.filter_object_by_semantic_annotation import FilterObjectBySemanticAnnotation
from robokudo_suturo.annotators.type_mapping_annotator import TypeMappingAnnotator
from robokudo_yolo.annotators.YoloAnnotator import YoloAnnotator


class Subtree(robokudo.analysis_engine.SubtreeInterface):
    def __init__(self, ros_pkg="robokudo_suturo", model="ultimate_model_v11_1.26.4", skip_outlier_removal_classes=None, with_semantic_filter=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ros_pkg = ros_pkg
        self.model = model
        self.skip_outlier_removal_classes = skip_outlier_removal_classes
        self.with_semantic_filter = with_semantic_filter

        """
        The RoboCup always includes cutlery and cutlery is small/thin and has few depth points,
        making it hard for outlier removal to yield a good result for the below objects.
        """
        if skip_outlier_removal_classes is None:
            self.skip_outlier_removal_classes = [
                "spoon_dinner_blackgrip",
                "spoon_dinner_redgrip",

                "fork_dinner_redgrip",
                "fork_dinner_blackgrip",

                "knife_butter_blackgrip",
                "knife_butter_redgrip",
                "knife_dinner_blackgrip",
                "knife_dinner_redgrip",

                #"marker_black",
                #"spatula_black",
                #"wineglass_shortstem"
            ]

    def implementation(self):
        """
        Returns a common object detection subtree which is used across all challenges
        """
        yolo_descriptor = YoloAnnotator.Descriptor()
        yolo_descriptor.parameters.precision_mode = True
        yolo_descriptor.parameters.ros_pkg_path = self.ros_pkg
        yolo_descriptor.parameters.weights_path = f"weights/{self.model}.pt"
        yolo_descriptor.parameters.id2name_json_path = f"data/json/id2name_{self.model}.json"
        yolo_descriptor.parameters.threshold = 0.6

        outlier_removal_oh_descriptor = OutlierRemovalOnObjectHypothesisAnnotator.Descriptor()
        outlier_removal_oh_descriptor.parameters.dbscan_neighbors = 90
        outlier_removal_oh_descriptor.parameters.dbscan_epsilon = 0.02
        outlier_removal_oh_descriptor.parameters.stat_neighbors = 300
        outlier_removal_oh_descriptor.parameters.stat_std = 0.45
        outlier_removal_oh_descriptor.parameters.skip_removal_on_classes = self.skip_outlier_removal_classes

        seq = py_trees.composites.Sequence("object_detection", memory=True)
        seq.add_children([
            YoloAnnotator(descriptor=yolo_descriptor),
            OutlierRemovalOnObjectHypothesisAnnotator(descriptor=outlier_removal_oh_descriptor),
            ClusterPoseBBAnnotator(),
            TypeMappingAnnotator(),
            #CutleryHandleAnnotator(),
        ])
        # optionally filter results to only objects matching the semantic query
        if self.with_semantic_filter:
            seq.add_children([FilterObjectBySemanticAnnotation()])
        return seq
