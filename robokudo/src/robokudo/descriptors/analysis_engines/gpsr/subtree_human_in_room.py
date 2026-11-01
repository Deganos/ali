import py_trees
import robokudo.analysis_engine
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo_human_activity.annotators.human_activity_annotator import HumanActivityAnnotator
from robokudo_human_activity.annotators.pointing import HumanPointingAnnotator
from robokudo_human_activity.annotators.waving import HumanWavingAnnotator
from robokudo_human_detection.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo_robocup_gpsr.annotators.gpsr_results import GenerateHumanActivityResult
from robokudo_suturo.annotators.belief_state import BeliefStateAnnotator
from robokudo_suturo.annotators.check_for_human_hypothesis import CheckForHumanHypothesis
from robokudo_suturo.annotators.filter_human_by_activity_annotator import FilterHumanByActivityAnnotator
from robokudo_suturo.annotators.filter_objects_by_region import FilterObjectsByRegion
from robokudo_suturo.annotators.generate_all_annotations import GenerateAllQueryResult


class Subtree(robokudo.analysis_engine.SubtreeInterface):

    def __init__(self, with_activity=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_activity = with_activity

    def implementation(self, name=None, *args, **kwargs):
        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True
        if self.with_activity:
            human_descriptor.parameters.generate_clouds_for_keypoints = True

        filter_region_descriptor = FilterObjectsByRegion.Descriptor()
        filter_region_descriptor.parameters.semantic_map_ros_package = "robokudo_suturo"
        filter_region_descriptor.parameters.semantic_map_name = "suturo_robocup_stick_to_the_rules"

        seq = py_trees.composites.Sequence("subtree_human_in_room", memory=True)
        seq.add_child(HumanAndPoseAnnotator(descriptor=human_descriptor))
        seq.add_child(CheckForHumanHypothesis())

        if self.with_activity:
            waving_descriptor = HumanWavingAnnotator.Descriptor()
            waving_descriptor.parameters.check_activity = True

            pointing_descriptor = HumanPointingAnnotator.Descriptor()
            pointing_descriptor.parameters.pointing_length_meters = 5
            pointing_descriptor.parameters.known = True
            pointing_descriptor.parameters.min_arm_extension = 0.25

            seq.add_children([
                HumanActivityAnnotator(),
                ClusterPositionAnnotator(),
                HumanWavingAnnotator(descriptor=waving_descriptor),
                HumanPointingAnnotator(descriptor=pointing_descriptor),
                FilterObjectsByRegion(descriptor=filter_region_descriptor),
                BeliefStateAnnotator(),
                FilterHumanByActivityAnnotator(),
                GenerateHumanActivityResult(),
            ])
        else:
            seq.add_children([
                ClusterPositionAnnotator(),
                FilterObjectsByRegion(descriptor=filter_region_descriptor),
                BeliefStateAnnotator(),
                GenerateAllQueryResult(),
            ])

        return seq
