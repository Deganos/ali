import json

import cv2
import py_trees

from ament_index_python.packages import get_package_share_directory

import robokudo.types.scene
import robokudo.types.annotation
from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.utils.error_handling import catch_and_raise_to_blackboard


class FilterObjectBySemanticAnnotation(BaseAnnotator):
    """
    Filters ObjectHypotheses in the CAS by semantic annotation type.
    Reads the requested type from the active query and retains only objects
    whose primary Classification matches it.
    """

    def __init__(self, name="FilterObjectBySemanticAnnotation",
                 mapping_filename: str = "mapping_perception_semantic-annotations.json"):
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

    @catch_and_raise_to_blackboard
    def update(self):
        query = self.get_cas().get(CASViews.QUERY)
        # the queried category is normally in the type field (e.g. Receptionist);
        # the GPSR dispatcher puts 'object' there for routing and carries the
        # category in the description instead, so fall back to it.
        query_type = query.obj.type
        if query_type not in self._type_map.values() and query.obj.description:
            query_type = query.obj.description[0]

        if query_type not in self._type_map.values():
            self.feedback_message = f"No semantic category to filter by ('{query_type}'), skipping"
            return py_trees.common.Status.SUCCESS

        cas = self.get_cas()
        objects = [oh for oh in cas.annotations
                   if isinstance(oh, robokudo.types.scene.ObjectHypothesis)]
        matching_objects = [oh for oh in objects if self._matches(oh, query_type)]

        # Debug visualization: green box = kept, red box = filtered out.
        try:
            self._visualize(cas, objects, query_type)
        except Exception as e:
            self.rk_logger.warning(f"Could not render filter visualization: {e}")

        if not matching_objects:
            raise Exception(f"No object found with semantic annotation '{query_type}'")

        cas.annotations = matching_objects
        self.feedback_message = f"Kept {len(matching_objects)} object(s) with annotation '{query_type}'"
        return py_trees.common.Status.SUCCESS

    def _matches(self, oh, query_type):
        """True if the object carries a Classification for the queried category."""
        for a in oh.annotations:
            if isinstance(a, robokudo.types.annotation.Classification) and a.classname == query_type:
                return True
        return False

    def _label(self, oh):
        """`fine_label -> Category` for the object, mirroring the type mapping."""
        classnames = [a.classname for a in oh.annotations
                      if isinstance(a, robokudo.types.annotation.Classification)]
        fine = next((c for c in classnames if c in self._type_map), None)
        if fine is not None:
            return f"{fine} -> {self._type_map[fine]}"
        return classnames[0] if classnames else "object"

    def _visualize(self, cas, objects, query_type):
        """Draw every detection box on the color image: green if its semantic
        category matches the query (kept), red if not (filtered out). Published
        to the annotator visualizer so the filter result can be screenshotted."""
        image = cas.get_copy(CASViews.COLOR_IMAGE)
        font = cv2.FONT_HERSHEY_COMPLEX
        for oh in objects:
            color = (0, 255, 0) if self._matches(oh, query_type) else (0, 0, 255)
            x1 = int(oh.roi.roi.pos.x)
            y1 = int(oh.roi.roi.pos.y)
            x2 = x1 + int(oh.roi.roi.width)
            y2 = y1 + int(oh.roi.roi.height)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, self._label(oh), (x1, max(y1 - 5, 12)),
                        font, 0.5, color, 1, 2)
        self.get_annotator_output_struct().set_image(image)
