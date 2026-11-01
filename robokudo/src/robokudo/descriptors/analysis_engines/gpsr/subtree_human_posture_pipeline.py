import py_trees
import robokudo.analysis_engine
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.types.scene import ObjectHypothesis
from robokudo_human_detection.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo_robocup_gpsr.annotators.gpsr_results import GenerateHumanPostureResults
from robokudo_suturo.annotators.belief_state import BeliefStateAnnotator
from robokudo_zero_shot_classification.annotators.zero_shot_classification import ZeroShotClfAnnotator


class Subtree(robokudo.analysis_engine.SubtreeInterface):
    def __init__(self, postures, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.postures = postures

    def implementation(self, name=None, *args, **kwargs):
        """
        Returns the list of annotators to be used in this subtree.
        """
        zero_shot_descriptor = ZeroShotClfAnnotator.Descriptor()
        zero_shot_descriptor.parameters.analysis_scope = ObjectHypothesis
        zero_shot_descriptor.parameters.save_top_k = 1
        zero_shot_descriptor.parameters.classes = self.postures

        seq = py_trees.composites.Sequence("subtree_human_position_pipeline", memory=True)
        seq.add_children([
            HumanAndPoseAnnotator(),
            ZeroShotClfAnnotator(descriptor=zero_shot_descriptor),
            ClusterPositionAnnotator(),
            BeliefStateAnnotator(),
            GenerateHumanPostureResults(),
        ])
        return seq