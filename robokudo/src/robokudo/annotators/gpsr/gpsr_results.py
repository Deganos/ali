from copy import deepcopy
from py_trees.common import Status
from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification, LocationAnnotation, PositionAnnotation
from robokudo.types.human import ActivityAnnotation
from robokudo.types.scene import HumanHypothesis, ObjectHypothesis
from robokudo.utils.query import QueryHandler
from robokudo.utils.type_conversion import get_geometry_msgs_pose_stamped_from_position_annotation
from robokudo_msgs.action import Query
from robokudo_msgs.msg import ObjectDesignator
from robokudo.annotators.gpsr.human_activity_annotator import HumanActivityAnnotator
from robokudo.annotators.gpsr.belief_state import BeliefEntry


class GenerateHumanPostureResults(BaseAnnotator):
    """
    Annotator to generate human posture results.
    """

    def __init__(self, name="GenerateHumanPostureResults", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def update(self):
        query = self.get_cas().get(CASViews.QUERY)

        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        query_result = []
        result = Query.Result()

        for hypothesis in human_hypotheses:
            object_designator = ObjectDesignator()
            object_designator.type = query.obj.type
            classifications = self.get_cas().filter_by_type(Classification, hypothesis.annotations)
            positions = self.get_cas().filter_by_type(PositionAnnotation, hypothesis.annotations)

            for classification in classifications:
                if classification.source == "ZeroShotClfAnnotator":
                    if classification.classname == query.obj.attribute[0]:
                        object_designator.attribute.append(classification.classname)

            if object_designator.attribute:
                belief_ids = self.get_cas().filter_by_type(BeliefEntry, hypothesis.annotations)
                if belief_ids:
                    object_designator.uid = str(belief_ids[0].id)
                for position in positions:
                    header = deepcopy(self.get_cas().get(CASViews.CAM_INFO).header)
                    pose_stamped = get_geometry_msgs_pose_stamped_from_position_annotation(position, header)
                    object_designator.pose.append(pose_stamped)
                    object_designator.pose_source.append(position.source)

                query_result.append(object_designator)

        result.res = query_result
        QueryHandler.send_answer(result)
        self.feedback_message = "Query result generated."
        return Status.SUCCESS


class GenerateObjectDetectionResult(BaseAnnotator):
    """
    Annotator to generate object detection results.
    """

    def __init__(self, name="ObjectDetectionResult", *args, **kwargs):
        super().__init__(*args, **kwargs)


    def update(self):
        object_hypotheses = self.get_cas().filter_annotations_by_type(ObjectHypothesis)
        query_result = []
        result = Query.Result()

        for hypothesis in object_hypotheses:
            object_designator = ObjectDesignator()
            object_designator.type = 'object'
            classifications = self.get_cas().filter_by_type(Classification, hypothesis.annotations)
            positions = self.get_cas().filter_by_type(PositionAnnotation, hypothesis.annotations)

            for classification in classifications:
                if classification.source == "YoloAnnotator":
                    object_designator.description.append(classification.classname)

            if object_designator.description:
                belief_ids = self.get_cas().filter_by_type(BeliefEntry, hypothesis.annotations)
                if belief_ids:
                    object_designator.uid = str(belief_ids[0].id)
                for position in positions:
                    header = deepcopy(self.get_cas().get(CASViews.CAM_INFO).header)
                    pose_stamped = get_geometry_msgs_pose_stamped_from_position_annotation(position, header)
                    object_designator.pose.append(pose_stamped)
                    object_designator.pose_source.append(position.source)

                query_result.append(object_designator)

        result.res = query_result
        QueryHandler.send_answer(result)

        self.feedback_message = "Query result generated."
        return Status.SUCCESS


class GenerateWavingRaisingResult(BaseAnnotator):
    """
    Annotator to generate waving and arm raising results.
    Arm raising can be either left or right.
    """

    def __init__(self, name="ObjectDetectionResult", *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self):
        query = self.get_cas().get(CASViews.QUERY)

        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        query_result = []
        result = Query.Result()

        for hypothesis in human_hypotheses:
            object_designator = ObjectDesignator()
            object_designator.type = query.obj.type
            positions = self.get_cas().filter_by_type(PositionAnnotation, hypothesis.annotations)
            activities = self.get_cas().filter_by_type(ActivityAnnotation, hypothesis.annotations)

            for activity in activities:
                if activity.source == "HumanWavingAnnotator":
                    if activity.type == query.obj.attribute[0]:
                        object_designator.description.append(activity.type)

            if object_designator.description:
                belief_ids = self.get_cas().filter_by_type(BeliefEntry, hypothesis.annotations)
                if belief_ids:
                    object_designator.uid = str(belief_ids[0].id)
                for position in positions:
                    header = deepcopy(self.get_cas().get(CASViews.CAM_INFO).header)
                    pose_stamped = get_geometry_msgs_pose_stamped_from_position_annotation(position, header)
                    object_designator.pose.append(pose_stamped)
                    object_designator.pose_source.append(position.source)

                query_result.append(object_designator)

        result.res = query_result
        QueryHandler.send_answer(result)

        self.feedback_message = "Query result generated."
        return Status.SUCCESS


class GeneratePointingResult(BaseAnnotator):
    """
    Annotator to generate pointing results.
    """

    def __init__(self, name="PointingResult", *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self):
        query = self.get_cas().get(CASViews.QUERY)

        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        query_result = []
        result = Query.Result()

        for hypothesis in human_hypotheses:
            object_designator = ObjectDesignator()
            object_designator.type = query.obj.type
            activities = self.get_cas().filter_by_type(ActivityAnnotation, hypothesis.annotations)
            positions = self.get_cas().filter_by_type(PositionAnnotation, hypothesis.annotations)

            for activity in activities:
                if activity.source == "HumanPointingAnnotator":
                    object_designator.description.append(f"Behavior suggests {activity.type}.")

            if object_designator.description:
                if activity.type == query.obj.attribute[0]:
                    belief_ids = self.get_cas().filter_by_type(BeliefEntry, hypothesis.annotations)
                    if belief_ids:
                        object_designator.uid = str(belief_ids[0].id)
                    for position in positions:
                        header = deepcopy(self.get_cas().get(CASViews.CAM_INFO).header)
                        pose_stamped = get_geometry_msgs_pose_stamped_from_position_annotation(position, header)
                        object_designator.pose.append(pose_stamped)
                        object_designator.pose_source.append(position.source)
                else:
                    continue

                query_result.append(object_designator)

        result.res = query_result
        QueryHandler.send_answer(result)

        self.feedback_message = "Query result generated."
        return Status.SUCCESS



class GenerateHumanActivityResult(BaseAnnotator):
    """
    Annotator to generate human activity results (posture + gestures).
    Reads ActivityAnnotations written by HumanActivityAnnotator.
    """

    def __init__(self, name="GenerateHumanActivityResult", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def update(self):
        query = self.get_cas().get(CASViews.QUERY)

        human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
        query_result = []
        result = Query.Result()

        for hypothesis in human_hypotheses:
            object_designator = ObjectDesignator()
            object_designator.type = query.obj.type
            positions = self.get_cas().filter_by_type(PositionAnnotation, hypothesis.annotations)
            activities = self.get_cas().filter_by_type(ActivityAnnotation, hypothesis.annotations)

            pointing_refinements = [a.type for a in activities if a.source == "HumanPointingAnnotator"]

            for activity in activities:
                if activity.source == "HumanActivityAnnotator":
                    if activity.type in HumanActivityAnnotator.Descriptor.Parameters().posture_labels:
                        object_designator.attribute.append(activity.type)
                    elif activity.type == "Pointing" and pointing_refinements:
                        # skeleton already has a concrete direction so we skip the ambiguous CLIP pointing label
                        continue
                    elif activity.type == "No gesture" and (
                        pointing_refinements
                        or any(a.source == "HumanWavingAnnotator" for a in activities)
                    ):
                        # skeleton found a real gesture so the CLIP no gesture label is dropped
                        continue
                    else:
                        object_designator.description.append(activity.type)
                elif activity.source == "HumanPointingAnnotator":
                    object_designator.description.append(activity.type)
                elif activity.source == "HumanWavingAnnotator":
                    object_designator.description.append(activity.type)


            if object_designator.attribute or object_designator.description:
                belief_ids = self.get_cas().filter_by_type(BeliefEntry, hypothesis.annotations)
                if belief_ids:
                    object_designator.uid = str(belief_ids[0].id)
                for position in positions:
                    header = deepcopy(self.get_cas().get(CASViews.CAM_INFO).header)
                    pose_stamped = get_geometry_msgs_pose_stamped_from_position_annotation(position, header)
                    object_designator.pose.append(pose_stamped)
                    object_designator.pose_source.append(position.source)

                query_result.append(object_designator)

        result.res = query_result
        QueryHandler.send_answer(result)
        self.feedback_message = "Query result generated."
        return Status.SUCCESS


class GenerateGSamResult(BaseAnnotator):
    """
    Annotator to generate waving and arm raising results.
    Arm raising can be either left or right.
    """

    def __init__(self, name="GSamResult", with_region_filter=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_region_filter = with_region_filter

    def containment_ratio(self, inner, outer):
        """
        Calculate the containment ratio of two hypotheses.
        :param inner: The inner hypothesis (the one that is contained).
        :param outer: The outer hypothesis (the one that contains).
        """
        inner_x_min = inner.roi.roi.pos.x
        inner_x_max = inner.roi.roi.pos.x + inner.roi.roi.width
        inner_y_min = inner.roi.roi.pos.y
        inner_y_max = inner.roi.roi.pos.y + inner.roi.roi.height
        outer_x_min = outer.roi.roi.pos.x
        outer_x_max = outer.roi.roi.pos.x + outer.roi.roi.width
        outer_y_min = outer.roi.roi.pos.y
        outer_y_max = outer.roi.roi.pos.y + outer.roi.roi.height

        xA = max(inner_x_min, outer_x_min)
        yA = max(inner_y_min, outer_y_min)
        xB = min(inner_x_max, outer_x_max)
        yB = min(inner_y_max, outer_y_max)

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        inner_area = (inner_x_max - inner_x_min) * (inner_y_max - inner_y_min)

        self.rk_logger.info(f"The outer hypothesis contains the inner hypothesis to {min(interArea / inner_area, 100)} of its area")
        return interArea / inner_area


    def intersection_over_union(self, gsam, human):
        """
        Calculate the intersection over union of two hypotheses.
        :param gsam: The first hypothesis (the one that is contained).
        :param human: The second hypothesis (the one that contains).
        """

        box1_x_min = gsam.roi.roi.pos.x
        box1_x_max = gsam.roi.roi.pos.x + gsam.roi.roi.width
        box1_y_min = gsam.roi.roi.pos.y
        box1_y_max = gsam.roi.roi.pos.y + gsam.roi.roi.height
        box2_x_min = human.roi.roi.pos.x
        box2_x_max = human.roi.roi.pos.x + human.roi.roi.width
        box2_y_min = human.roi.roi.pos.y
        box2_y_max = human.roi.roi.pos.y + human.roi.roi.height

        xA = max(box1_x_min, box2_x_min)
        yA = max(box1_y_min, box2_y_min)
        xB = min(box1_x_max, box2_x_max)
        yB = min(box1_y_max, box2_y_max)

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        box1_area = (box1_x_max - box1_x_min) * (box1_y_max - box1_y_min)
        box2_area = (box2_x_max - box2_x_min) * (box2_y_max - box2_y_min)

        iou = interArea / float(box1_area + box2_area - interArea)

        self.rk_logger.info(f"Intersection over union between {gsam} and {human}: {iou}")
        return iou

    def update(self):
        query = self.get_cas().get(CASViews.QUERY)

        # HumanPipeline
        if query.obj.type == "human" or query.obj.type == "humans":
            human_hypotheses = self.get_cas().filter_annotations_by_type(HumanHypothesis)
            object_hypotheses = self.get_cas().filter_annotations_by_type(ObjectHypothesis)

            if self.with_region_filter and query.obj.location:
                human_hypotheses = [
                    h for h in human_hypotheses
                    if any(isinstance(a, LocationAnnotation) and a.name == query.obj.location
                           for a in h.annotations)
                ]

            query_result = []
            result = Query.Result()

            for human_hypothesis in human_hypotheses:
                for object_hypothesis in object_hypotheses:
                    # We dont want to compare human with human or object with object
                    if not isinstance(object_hypothesis, HumanHypothesis):
                        gsam_hypothesis = self.get_cas().filter_by_type(Classification, object_hypothesis.annotations)[0]
                        if gsam_hypothesis.source == "GroundedSAM":
                            containment_ratio = self.containment_ratio(object_hypothesis, human_hypothesis)
                            if containment_ratio > 0.8:
                                object_designator = ObjectDesignator()
                                object_designator.type = query.obj.type
                                object_designator.description.append(gsam_hypothesis.classname)
                                belief_ids = self.get_cas().filter_by_type(BeliefEntry, human_hypothesis.annotations)
                                if belief_ids:
                                    object_designator.uid = str(belief_ids[0].id)

                                positions = self.get_cas().filter_by_type(PositionAnnotation, human_hypothesis.annotations)
                                for position in positions:
                                    header = deepcopy(self.get_cas().get(CASViews.CAM_INFO).header)
                                    pose_stamped = get_geometry_msgs_pose_stamped_from_position_annotation(position, header)
                                    object_designator.pose.append(pose_stamped)
                                    object_designator.pose_source.append(position.source)

                                query_result.append(object_designator)

        # ObjectPipeline
        elif query.obj.type == "object":
            object_hypotheses = self.get_cas().filter_annotations_by_type(ObjectHypothesis)
            query_result = []
            result = Query.Result()

            for object_hypothesis in object_hypotheses:
                assert len(self.get_cas().filter_by_type(Classification, object_hypothesis.annotations)) > 0, (
                        len(self.get_cas().filter_by_type(Classification, object_hypothesis.annotations)) < 2)
                gsam_hypothesis = self.get_cas().filter_by_type(Classification, object_hypothesis.annotations)[0]
                if gsam_hypothesis.source == "GroundedSAM":
                    object_designator = ObjectDesignator()
                    object_designator.type = query.obj.type
                    object_designator.description.append(gsam_hypothesis.classname)

                    positions = self.get_cas().filter_by_type(PositionAnnotation, object_hypothesis.annotations)
                    for position in positions:
                        header = deepcopy(self.get_cas().get(CASViews.CAM_INFO).header)
                        pose_stamped = get_geometry_msgs_pose_stamped_from_position_annotation(position, header)
                        object_designator.pose.append(pose_stamped)
                        object_designator.pose_source.append(position.source)

                    query_result.append(object_designator)
        else:
            return Status.FAILURE

        result.res = query_result
        QueryHandler.send_answer(result)

        self.feedback_message = "Query result generated."
        return Status.SUCCESS

