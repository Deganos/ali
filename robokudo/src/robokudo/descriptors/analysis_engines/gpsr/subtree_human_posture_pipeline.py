import py_trees
import robokudo.analysis_engine
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.types.scene import ObjectHypothesis
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.gpsr.gpsr_results import GenerateHumanPostureResults
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator
from robokudo.annotators.clip_annotator import CLIPAnnotator


class Subtree(robokudo.analysis_engine.SubtreeInterface):
    def __init__(self, postures, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.postures = postures

    def implementation(self, name=None, *args, **kwargs):
        """
        Returns the list of annotators to be used in this subtree.
        """
        zero_shot_descriptor = CLIPAnnotator.Descriptor()
        zero_shot_descriptor.parameters.analysis_scope = ObjectHypothesis
        zero_shot_descriptor.parameters.save_top_k = 1
        zero_shot_descriptor.parameters.classes = self.postures

        seq = py_trees.composites.Sequence("subtree_human_position_pipeline", memory=True)
        seq.add_children([
            HumanAndPoseAnnotator(),
            CLIPAnnotator(descriptor=zero_shot_descriptor),
            ClusterPositionAnnotator(),
            BeliefStateAnnotator(),
            GenerateHumanPostureResults(),
        ])
        return seq