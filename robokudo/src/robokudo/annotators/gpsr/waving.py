import copy
from timeit import default_timer

import geometry_msgs.msg
import numpy
import open3d as o3d
import py_trees
import rclpy
import visualization_msgs.msg

import robokudo
import robokudo.annotators.core
import robokudo.cas
import robokudo.types.annotation
import robokudo.types.human
import robokudo.types.scene
import robokudo.types.tf
import robokudo.utils.cv_helper
import robokudo.utils.decorators
import robokudo.utils.error_handling
import robokudo.utils.math_helper
import robokudo.utils.o3d_helper
import robokudo.utils.transform
from robokudo.annotators.human_and_pose import PoseKeypoint


class HumanWavingAnnotator(robokudo.annotators.core.BaseAnnotator):
    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.raise_on_failure = False
                self.keypoint_confidence_threshold = 0.7
                # If True, only run on HHs that have "No gesture" from HumanActivityAnnotator.
                self.check_activity = False

                # This factor is multiplied by the shoulder<->eye-level distance in pixel
                # coordinates to be used as an offset from the eye-level
                # to decide if a human is waving or not
                self.waving_decision_factor = 0.2

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="HumanWavingAnnotator", descriptor=Descriptor()):
        super().__init__(name, descriptor)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self):
        start_timer = default_timer()

        self.color = self.get_cas().get(robokudo.cas.CASViews.COLOR_IMAGE)
        self.cloud = self.get_cas().get(robokudo.cas.CASViews.CLOUD)

        human_hypotheses = self.get_cas().filter_annotations_by_type(robokudo.types.scene.HumanHypothesis)

        if self.descriptor.parameters.check_activity:
            # only process humans the HumanActivityAnnotator already tagged as no gesture
            human_hypotheses = [
                hh for hh in human_hypotheses
                if any(
                    isinstance(a, robokudo.types.human.ActivityAnnotation)
                    and a.source == "HumanActivityAnnotator"
                    and a.type == "No gesture"
                    for a in hh.annotations
                )
            ]
            if not human_hypotheses:
                self.feedback_message = "No 'No gesture' HH found, skipping"
                return py_trees.common.Status.SUCCESS

        all_vis_geometries = [self.cloud]

        for human_hypothesis in human_hypotheses:
            keypoint_annotations = robokudo.cas.CAS.filter_by_type(robokudo.types.human.KeypointAnnotation,
                                                                   human_hypothesis.annotations)
            keypoint_2d_annotations = [a for a in keypoint_annotations if
                                       a.kind == robokudo.types.human.KeypointKind.IMAGE_POSITION]

            # Human detected, but no keypoint annotations. Can't estimate waving.
            if len(keypoint_2d_annotations) == 0:
                continue

            keypoint_2d_annotation = keypoint_2d_annotations[0]

            raising = self.keypoints_suggest_arm_raising(keypoint_2d_annotation)
            if raising == "left":
                activity = robokudo.types.human.ActivityAnnotation()
                activity.type = 'raising right arm'
                activity.source = type(self).__name__
                human_hypothesis.annotations.append(activity)
            elif raising == "right":
                activity = robokudo.types.human.ActivityAnnotation()
                activity.type = 'raising left arm'
                activity.source = type(self).__name__
                human_hypothesis.annotations.append(activity)
            waving = self.keypoints_suggest_waving(keypoint_2d_annotation)
            if waving:
                activity = robokudo.types.human.ActivityAnnotation()
                activity.type = 'waving'
                activity.source = type(self).__name__
                human_hypothesis.annotations.append(activity)

        self.get_annotator_output_struct().set_geometries(all_vis_geometries)
        self.get_annotator_output_struct().set_image(copy.deepcopy(self.color))

        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.common.Status.SUCCESS

    def keypoints_suggest_arm_raising(self, keypoint_2d_annotation: robokudo.types.human.KeypointAnnotation):
        left_wrist = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_WRIST]
        right_wrist = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_WRIST]
        left_shoulder = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_SHOULDER]
        right_shoulder = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_SHOULDER]
        left_eye = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_EYE]
        right_eye = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_EYE]

        left_eye_shoulder_distance_pixel = left_shoulder.y - left_eye.y
        right_eye_shoulder_distance_pixel = right_shoulder.y - right_eye.y

        left_above_eye_level_offset_pixel = (left_eye_shoulder_distance_pixel
                                             * self.descriptor.parameters.waving_decision_factor)
        right_above_eye_level_offset_pixel = (right_eye_shoulder_distance_pixel
                                              * self.descriptor.parameters.waving_decision_factor)

        if left_wrist.y < (left_shoulder.y - left_above_eye_level_offset_pixel) and \
                left_wrist.confidence > self.descriptor.parameters.keypoint_confidence_threshold:
            return "right"
        if right_wrist.y < (right_shoulder.y - right_above_eye_level_offset_pixel) and \
                right_wrist.confidence > self.descriptor.parameters.keypoint_confidence_threshold:
            return "left"
        return None

    def keypoints_suggest_waving(self, keypoint_2d_annotation: robokudo.types.human.KeypointAnnotation):
        left_wrist = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_WRIST]
        right_wrist = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_WRIST]
        left_shoulder = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_SHOULDER]
        right_shoulder = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_SHOULDER]
        left_eye = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_EYE]
        right_eye = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_EYE]

        # Because this approach is using 2d coordinates, we need some kind of size estimate
        # to define proper thresholds in the pixel-level
        # We'll take the distance between shoulder and the eyes on each side
        # as a rough distance clue
        left_eye_shoulder_distance_pixel = left_shoulder.y - left_eye.y
        right_eye_shoulder_distance_pixel = right_shoulder.y - right_eye.y

        left_above_eye_level_offset_pixel = (left_eye_shoulder_distance_pixel
                                             * self.descriptor.parameters.waving_decision_factor)
        right_above_eye_level_offset_pixel = (right_eye_shoulder_distance_pixel
                                              * self.descriptor.parameters.waving_decision_factor)

        # y coordinates start with 0 from the top of the image
        if left_wrist.y < (left_eye.y - left_above_eye_level_offset_pixel) and \
                left_wrist.confidence > self.descriptor.parameters.keypoint_confidence_threshold:
            return True
        if right_wrist.y < (right_eye.y - right_above_eye_level_offset_pixel) and \
                right_wrist.confidence > self.descriptor.parameters.keypoint_confidence_threshold:
            return True

        return False
