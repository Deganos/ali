import copy
from timeit import default_timer
import py_trees
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


class HumanStandingAnnotator(robokudo.annotators.core.BaseAnnotator):
    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.raise_on_failure = False

                # Minimum confidence required for a keypoint to be considered valid.
                self.keypoint_confidence_threshold = 0.5

                # Thresholds used for standing detection.
                self.min_rel_vertical = (
                    0.6  # Person must be significantly taller than wide.
                )
                self.max_rel_horizontal = 0.45  # Horizontal spread must be limited.
                self.min_shoulder_over_hip = (
                    0.25  # Shoulders must be clearly above the hips.
                )
                self.min_hip_over_knee = 0.30  # Hips must be clearly above the knees.
                self.min_head_over_shoulder = (
                    0.10  # Head must be above the shoulder line.
                )

        parameters = Parameters()

    def __init__(self, name="HumanStandingAnnotator", descriptor=Descriptor()):
        super().__init__(name, descriptor)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self):
        start_timer = default_timer()

        # Retrieve current sensor data from CAS.
        self.color = self.get_cas().get(robokudo.cas.CASViews.COLOR_IMAGE)
        self.cloud = self.get_cas().get(robokudo.cas.CASViews.CLOUD)

        human_hypotheses = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.HumanHypothesis
        )

        all_vis_geometries = [self.cloud]

        # Process each detected human individually.
        for human_hypothesis in human_hypotheses:
            keypoint_annotations = robokudo.cas.CAS.filter_by_type(
                robokudo.types.human.KeypointAnnotation, human_hypothesis.annotations
            )
            # Only use 2D keypoints for pose analysis.
            keypoint_2d_annotations = [
                a
                for a in keypoint_annotations
                if a.kind == robokudo.types.human.KeypointKind.IMAGE_POSITION
            ]

            if len(keypoint_2d_annotations) == 0:
                continue

            keypoints = keypoint_2d_annotations[0]

            # Evaluate whether the person is standing.
            if self.keypoints_standing(keypoints):
                activity = robokudo.types.human.ActivityAnnotation()
                activity.type = "standing"
                activity.source = type(self).__name__
                human_hypothesis.annotations.append(activity)

        self.get_annotator_output_struct().set_geometries(all_vis_geometries)
        self.get_annotator_output_struct().set_image(copy.deepcopy(self.color))

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS

        # Determines whether the given keypoints indicate a standing posture.

    def keypoints_standing(self, keypoint_2d_annotation):
        p = self.descriptor.parameters
        th = p.keypoint_confidence_threshold

        # Load all relevant keypoints (hips, knees, shoulders, head).
        left_hip = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_HIP]
        right_hip = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_HIP]
        left_knee = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_KNEE]
        right_knee = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_KNEE]
        left_shoulder = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_SHOULDER]
        right_shoulder = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_SHOULDER]
        nose = keypoint_2d_annotation.keypoints[PoseKeypoint.NOSE]

        # Check upper body visibility
        upper_ok = (
            min(left_shoulder.confidence, right_shoulder.confidence, nose.confidence)
            >= th
        )

        # Check left leg visibility
        left_leg_ok = min(left_hip.confidence, left_knee.confidence) >= th

        # Check right leg visibility
        right_leg_ok = min(right_hip.confidence, right_knee.confidence) >= th

        # Require upper body + at least one leg
        if not upper_ok or not (left_leg_ok or right_leg_ok):
            return False

            # Collect all x and y coordinates to compute bounding box dimensions.
        xs = [
            left_hip.x,
            right_hip.x,
            left_knee.x,
            right_knee.x,
            left_shoulder.x,
            right_shoulder.x,
            nose.x,
        ]

        ys = [
            left_hip.y,
            right_hip.y,
            left_knee.y,
            right_knee.y,
            left_shoulder.y,
            right_shoulder.y,
            nose.y,
        ]
        # Compute vertical and horizontal extent of the detected person.
        vertical_h = max(ys) - min(ys)
        horizontal_w = max(xs) - min(xs)

        if vertical_h < 20:
            return False

        # Compute normalized height/width ratios.
        rel_vertical = vertical_h / (vertical_h + horizontal_w)
        rel_horizontal = horizontal_w / (vertical_h + horizontal_w)

        # Compute average vertical positions of shoulders, hips, and knees.
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        # Use the more confident leg for posture analysis
        if left_leg_ok:
            hip_y = left_hip.y
            knee_y = left_knee.y
        elif right_leg_ok:
            hip_y = right_hip.y
            knee_y = right_knee.y
        else:
            return False  # Shouldn't happen due to earlier check

        shoulder_hip_diff = (shoulder_y - hip_y) / vertical_h
        hip_knee_diff = (hip_y - knee_y) / vertical_h
        head_shoulder_diff = (nose.y - shoulder_y) / vertical_h

        # Standing heuristic:
        standing = (
            rel_vertical
            >= p.min_rel_vertical  # Body is significantly taller than it is wide
            and rel_horizontal
            <= p.max_rel_horizontal  # Horizontal spread is small (not lying)
            and shoulder_hip_diff
            <= -p.min_shoulder_over_hip  # Shoulders are clearly above the hips
            and hip_knee_diff
            <= -p.min_hip_over_knee  # Hips are clearly above the knees
            and head_shoulder_diff
            <= -p.min_head_over_shoulder  # Head is positioned above the shoulders
        )

        return standing
