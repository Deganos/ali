import py_trees
import robokudo.analysis_engine
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo_human_activity.annotators.waving import HumanWavingAnnotator
from robokudo_human_detection.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo_robocup_gpsr.annotators.gpsr_results import GenerateWavingRaisingResult
from robokudo_suturo.annotators.belief_state import BeliefStateAnnotator


class Subtree(robokudo.analysis_engine.SubtreeInterface):
    def __init__(self, attribute=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attribute = attribute

    def implementation(self, name=None, *args, **kwargs):
        seq = py_trees.composites.Sequence("subtree_waving_raising_pipeline", memory=True)
        seq.add_children([
            HumanAndPoseAnnotator(),
            ClusterPositionAnnotator(),
            HumanWavingAnnotator(),
            BeliefStateAnnotator(),
            GenerateWavingRaisingResult(),
        ])
        return seq