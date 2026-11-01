import py_trees
import robokudo.analysis_engine
from robokudo.annotators.gpsr.belief_state_lookup_annotator import (
    BeliefStateLookupAnnotator,
)


class Subtree(robokudo.analysis_engine.SubtreeInterface):
    def implementation(self, name=None, *args, **kwargs):
        seq = py_trees.composites.Sequence("subtree_belief_lookup", memory=True)
        seq.add_child(BeliefStateLookupAnnotator())
        return seq
