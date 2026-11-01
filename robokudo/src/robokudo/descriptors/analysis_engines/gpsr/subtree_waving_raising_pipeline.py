import py_trees
import robokudo.analysis_engine
from robokudo.annotators.cluster_position import ClusterPositionAnnotator
from robokudo.annotators.gpsr.waving import HumanWavingAnnotator
from robokudo.annotators.human_and_pose import HumanAndPoseAnnotator
from robokudo.annotators.gpsr.gpsr_results import GenerateWavingRaisingResult
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator


class Subtree(robokudo.analysis_engine.SubtreeInterface):
    def __init__(self, attribute=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attribute = attribute

    def implementation(self, name=None, *args, **kwargs):
        # CRAM's HumanAndPoseAnnotator works without depth by default, the person's position needs it
        human_descriptor = HumanAndPoseAnnotator.Descriptor()
        human_descriptor.parameters.global_with_depth = True

        seq = py_trees.composites.Sequence("subtree_waving_raising_pipeline", memory=True)
        seq.add_children([
            HumanAndPoseAnnotator(descriptor=human_descriptor),
            ClusterPositionAnnotator(),
            HumanWavingAnnotator(),
            BeliefStateAnnotator(),
            GenerateWavingRaisingResult(),
        ])
        return seq