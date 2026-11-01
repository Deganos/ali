import cv2
import py_trees

import robokudo.types.scene
import robokudo.types.human
from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.utils.error_handling import catch_and_raise_to_blackboard


class FilterHumanByActivityAnnotator(BaseAnnotator):
    """
    Filters HumanHypotheses in the CAS by posture and/or gesture.
    attribute[0] = posture filter and attribute[1] = gesture filter.
    Empty string means no filter for that slot. Both are compared in lowercase.
    """

    def __init__(self, name="FilterHumanByActivityAnnotator"):
        super().__init__(name=name)

    @catch_and_raise_to_blackboard
    def update(self):
        query = self.get_cas().get(CASViews.QUERY)
        attribute = query.obj.attribute
        cas = self.get_cas()
        matching = []

        if len(attribute) > 0:
            posture_filter = attribute[0].strip().lower()
        else:
            posture_filter = ""

        if len(attribute) > 1:
            gesture_filter = attribute[1].strip().lower()
        else:
            gesture_filter = ""

        if not posture_filter and not gesture_filter:
            self.feedback_message = "No activity filter requested, skipping"
            return py_trees.common.Status.SUCCESS

        for hh in cas.annotations:
            if not isinstance(hh, robokudo.types.scene.HumanHypothesis):
                continue

            activity_types = {
                a.type.lower()
                for a in hh.annotations
                if isinstance(a, robokudo.types.human.ActivityAnnotation)
            }

            if posture_filter == "":
                posture_ok = True
            else:
                posture_ok = posture_filter in activity_types

            if gesture_filter == "":
                gesture_ok = True
            else:
                gesture_ok = gesture_filter in activity_types

            if posture_ok and gesture_ok:
                matching.append(hh)

        self._draw_filter_overlay(cas, matching)
        cas.annotations = matching
        return py_trees.common.Status.SUCCESS

    def _draw_filter_overlay(self, cas, matching):
        img = cas.get(CASViews.COLOR_IMAGE)
        if img is None:
            return
        img = img.copy()
        for hh in cas.annotations:
            if not isinstance(hh, robokudo.types.scene.HumanHypothesis):
                continue
            r = hh.roi.roi
            x, y, w, h = int(r.pos.x), int(r.pos.y), int(r.width), int(r.height)
            kept = hh in matching
            color = (0, 200, 0) if kept else (0, 0, 255)
            if not kept:
                overlay = img.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                img,
                "match" if kept else "filtered",
                (x, max(y - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
        self.get_annotator_output_struct().set_image(img)
