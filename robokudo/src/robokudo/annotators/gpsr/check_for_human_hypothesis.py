import robokudo
import py_trees
import robokudo.annotators
import robokudo.types.scene


class CheckForHumanHypothesis(robokudo.annotators.core.BaseAnnotator):
    """
    This Annotator checks the cas, if it already contains a HumanHypothesis.
    """

    def __init__(self, name="CheckForHumanHypothesis"):
        super().__init__(name=name)

    def update(self):
        annotations = self.get_cas().annotations
        for annotation in annotations:
            if isinstance(annotation, robokudo.types.scene.HumanHypothesis):
                return py_trees.common.Status.SUCCESS
        raise AttributeError("No human detected within the area!")
