import py_trees
import robokudo.analysis_engine
from robokudo_suturo.annotators.belief_state import BeliefStateAnnotator
from robokudo_suturo.annotators.filter_objects_by_region import FilterObjectsByRegion
from robokudo_suturo.annotators.generate_all_annotations import GenerateAllQueryResult
from robokudo_suturo.descriptors.analysis_engines.subtree_object_detection import Subtree as ObjectDetectionSubtree


class Subtree(robokudo.analysis_engine.SubtreeInterface):

    def implementation(self, name=None, *args, **kwargs):
        filter_region_descriptor = FilterObjectsByRegion.Descriptor()
        filter_region_descriptor.parameters.semantic_map_ros_package = "robokudo_suturo"
        filter_region_descriptor.parameters.semantic_map_name = "suturo_robocup_stick_to_the_rules"

        seq = py_trees.composites.Sequence("subtree_obj_in_room", memory=True)
        seq.add_children([
            ObjectDetectionSubtree(model="ultimate_model_v11_1.26.4").implementation(),
            FilterObjectsByRegion(descriptor=filter_region_descriptor),
            BeliefStateAnnotator(),
            GenerateAllQueryResult(),
        ])
        return seq
