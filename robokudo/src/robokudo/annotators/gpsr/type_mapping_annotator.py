import json
import py_trees

from ament_index_python.packages import get_package_share_directory

import robokudo.types.scene
import robokudo.types.annotation
from robokudo.annotators.core import BaseAnnotator


class TypeMappingAnnotator(BaseAnnotator):
    """
    Adds a mapped Classification to each ObjectHypothesis based on a mapping file.

    Ensures the mapped Classification is the first Classification on the OH.
    """

    def __init__(
        self,
        name="TypeMappingAnnotator",
        mapping_filename: str = "mapping_perception_semantic-annotations.json",
    ):
        super().__init__(name=name)
        self._type_map = {}
        try:
            pkg_share = get_package_share_directory("robokudo_suturo")
            mapping_path = f"{pkg_share}/data/json/{mapping_filename}"
            with open(mapping_path, "r") as f:
                self._type_map = json.load(f)
        except Exception as e:
            self.rk_logger.warning(
                f"Could not load mapping file '{mapping_filename}': {e}"
            )

    def update(self):
        cas = self.get_cas()
        for oh in cas.annotations:
            if not isinstance(oh, robokudo.types.scene.ObjectHypothesis):
                continue

            classifications = []
            non_classifications = []
            # Split annotations into classifications and all other types
            for a in oh.annotations:
                if isinstance(a, robokudo.types.annotation.Classification):
                    classifications.append(a)
                else:
                    non_classifications.append(a)
            # Skip objects without any classification
            if not classifications:
                continue

            raw = classifications[0].classname
            mapped = self._type_map.get(raw)
            # Skip if no mapping exists for this type
            if not mapped:
                continue

            # Check if the mapped classification already exists
            for c in classifications:
                if c.classname == mapped:
                    break
            else:
                # Create and prepend the mapped classification if it is missing
                mapped_classification = robokudo.types.annotation.Classification()
                mapped_classification.classname = mapped
                mapped_classification.source = self.name
                oh.annotations = (
                    [mapped_classification] + classifications + non_classifications
                )

        return py_trees.common.Status.SUCCESS
