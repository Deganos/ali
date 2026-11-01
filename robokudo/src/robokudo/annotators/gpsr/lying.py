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


class HumanLyingAnnotator(robokudo.annotators.core.BaseAnnotator):
    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.raise_on_failure = False

                # Minimum confidence required for a keypoint to be considered valid.
                self.keypoint_confidence_threshold = 0.5

                # Lying threshold TODO Anpassen
                self.max_vertical_rel_height = 0.35  # Person appears vertically short.
                self.min_horizontal_rel_width = 0.4  # Person appears horizontally wide.
                self.max_knee_hip_vertical_diff = (
                    0.2  # Knees and hips are nearly at the same
                )

        parameters = (
            Parameters()
        )  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="HumanLyingAnnotator", descriptor=Descriptor()):
        super().__init__(name, descriptor)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self):
        start_timer = default_timer()

        # Retrieve sensor data from CAS (color image, point cloud, camera info).
        self.color = self.get_cas().get(robokudo.cas.CASViews.COLOR_IMAGE)
        self.cloud = self.get_cas().get(robokudo.cas.CASViews.CLOUD)

        human_hypotheses = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.HumanHypothesis
        )
        all_vis_geometries = [self.cloud]

        # Process each detected human individually.
        for human_hypothesis in human_hypotheses:
            # Extract all keypoint annotations for this human.
            keypoint_annotations = robokudo.cas.CAS.filter_by_type(
                robokudo.types.human.KeypointAnnotation, human_hypothesis.annotations
            )

            keypoint_2d_annotations = [
                a
                for a in keypoint_annotations
                if a.kind == robokudo.types.human.KeypointKind.IMAGE_POSITION
            ]

            if len(keypoint_2d_annotations) == 0:
                continue

            keypoints = keypoint_2d_annotations[0]

            # Evaluate whether the person is lying.
            if self.keypoints_lying(keypoints):
                activity = robokudo.types.human.ActivityAnnotation()
                activity.type = "lying"
                activity.source = type(self).__name__
                human_hypothesis.annotations.append(activity)

        self.get_annotator_output_struct().set_geometries(all_vis_geometries)
        self.get_annotator_output_struct().set_image(copy.deepcopy(self.color))

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS

    def keypoints_lying(self, keypoint_2d_annotation):
        p = self.descriptor.parameters
        th = p.keypoint_confidence_threshold

        # Load lower-body keypoints.
        left_hip = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_HIP]
        left_knee = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_KNEE]
        left_ankle = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_ANKLE]

        right_hip = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_HIP]
        right_knee = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_KNEE]
        right_ankle = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_ANKLE]

        # Load upper-body keypoints.
        left_shoulder = keypoint_2d_annotation.keypoints[PoseKeypoint.LEFT_SHOULDER]
        right_shoulder = keypoint_2d_annotation.keypoints[PoseKeypoint.RIGHT_SHOULDER]
        nose = keypoint_2d_annotation.keypoints[PoseKeypoint.NOSE]

        # Check whether lower and upper body keypoints are sufficiently reliable.
        lower_ok = (
            min(
                left_hip.confidence,
                left_knee.confidence,
                left_ankle.confidence,
                right_hip.confidence,
                right_knee.confidence,
                right_ankle.confidence,
            )
            >= th
        )

        upper_ok = (
            min(left_shoulder.confidence, right_shoulder.confidence, nose.confidence)
            >= th
        )

        if not lower_ok and not upper_ok:
            return False

        xs = [
            left_hip.x,
            left_knee.x,
            left_ankle.x,
            right_hip.x,
            right_knee.x,
            right_ankle.x,
            left_shoulder.x,
            right_shoulder.x,
            nose.x,
        ]

        ys = [
            left_hip.y,
            left_knee.y,
            left_ankle.y,
            right_hip.y,
            right_knee.y,
            right_ankle.y,
            left_shoulder.y,
            right_shoulder.y,
            nose.y,
        ]

        vertical_h = max(ys) - min(ys)
        horizontal_w = max(xs) - min(xs)

        if vertical_h < 20:
            return False

        # Compute normalized height/width ratios.
        rel_vertical = vertical_h / (vertical_h + horizontal_w)
        rel_horizontal = horizontal_w / (vertical_h + horizontal_w)

        knee_hip_diff_left = abs(left_knee.y - left_hip.y) / vertical_h
        knee_hip_diff_right = abs(right_knee.y - right_hip.y) / vertical_h

        shoulder_diff = abs(left_shoulder.y - right_shoulder.y) / vertical_h

        head_shoulder_diff = (
            abs(nose.y - (left_shoulder.y + right_shoulder.y) / 2) / vertical_h
        )

        # print(f"rel_vertical={rel_vertical}, rel_horizontal={rel_horizontal}")
        # print(f"knee_hip_diff_left={knee_hip_diff_left}, shoulder_diff={shoulder_diff}")
        # print(f"head shoulder diff_left={head_shoulder_diff}")

        # TODO: Thresholds für Liegend Erkennung anpassen
        # Lying posture heuristic.
        lying = (
            rel_vertical <= p.max_vertical_rel_height
            and rel_horizontal >= p.min_horizontal_rel_width
            and (
                knee_hip_diff_left <= p.max_knee_hip_vertical_diff
                or knee_hip_diff_right <= p.max_knee_hip_vertical_diff
            )
            and shoulder_diff <= 0.35
            and head_shoulder_diff <= 0.4
        )

        return lying
