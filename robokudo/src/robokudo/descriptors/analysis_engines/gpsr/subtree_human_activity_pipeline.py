import os

import py_trees
import robokudo.analysis_engine
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo_human_activity.annotators.pointing import HumanPointingAnnotator
from robokudo_human_activity.annotators.waving import HumanWavingAnnotator
from robokudo_human_detection.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo_robocup_gpsr.annotators.gpsr_results import GenerateHumanActivityResult
from robokudo_human_activity.annotators.human_activity_annotator import HumanActivityAnnotator
from robokudo_suturo.annotators.belief_state import BeliefStateAnnotator
from robokudo_suturo.annotators.check_for_human_hypothesis import CheckForHumanHypothesis
from robokudo_suturo.annotators.filter_human_by_activity_annotator import FilterHumanByActivityAnnotator


class Subtree(robokudo.analysis_engine.SubtreeInterface):
    def __init__(self, with_filter=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_filter = with_filter

    def implementation(self, name=None, *args, **kwargs):
        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True
        human_descriptor.parameters.generate_clouds_for_keypoints = True

        waving_descriptor = HumanWavingAnnotator.Descriptor()
        waving_descriptor.parameters.check_activity = True

        pointing_descriptor = HumanPointingAnnotator.Descriptor()
        pointing_descriptor.parameters.pointing_length_meters = 5
        pointing_descriptor.parameters.known = True
        pointing_descriptor.parameters.min_arm_extension = 0.25

        seq = py_trees.composites.Sequence("subtree_human_activity_pipeline", memory=True)
        seq.add_children([
            HumanAndPoseAnnotator(descriptor=human_descriptor),
            CheckForHumanHypothesis(),
            HumanActivityAnnotator(),
            ClusterPositionAnnotator(),
            HumanWavingAnnotator(descriptor=waving_descriptor),
            HumanPointingAnnotator(descriptor=pointing_descriptor),
            BeliefStateAnnotator(),
        ])
        if self.with_filter:
            seq.add_child(FilterHumanByActivityAnnotator())
        seq.add_child(GenerateHumanActivityResult())
        return seq