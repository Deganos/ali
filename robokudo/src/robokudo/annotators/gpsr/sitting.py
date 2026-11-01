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
from robokudo_human_detection.annotators.human_and_pose import HumanAndPoseAnnotator

# Structure to hold keypoint data and confidence to ease handling of multiple keypoints
class KeypointData:
    def __init__(self, data: list):
        assert (len(data) == 3)
        self.x = data[0]
        self.y = data[1]
        self.confidence = data[2]

class HumanSittingAnnotator(robokudo.annotators.core.BaseAnnotator):
    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.raise_on_failure = False

                # Minimum confidence required for a keypoint to be considered valid.
                self.keypoint_confidence_threshold = 0.5

                #Sitting threshold TODO Anpassen
                self.max_hip_knee_rel_bbox = 0.5  # Hip–knee distance relative to body height.
                self.max_knee_knee_rel_bbox = 0.5 # Vertical difference between left/right knees.
                self.min_ankle_knee_rel_bbox = 0.2 # Knee–ankle distance to ensure the person is not lying.

        parameters = Parameters()  # overwrite the parameters explicitly to enable auto-completion

    def __init__(self, name="HumanSittingAnnotator", descriptor=Descriptor()):
        super().__init__(name, descriptor)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self):
        start_timer = default_timer()

        # Retrieve current sensor data from CAS.
        self.color = self.get_cas().get(robokudo.cas.CASViews.COLOR_IMAGE)
        self.cloud = self.get_cas().get(robokudo.cas.CASViews.CLOUD)
        self.cam_info = self.get_cas().get(robokudo.cas.CASViews.CAM_INFO)

        # Retrieve all detected human hypotheses.
        human_hypotheses = self.get_cas().filter_annotations_by_type(robokudo.types.scene.HumanHypothesis)
        all_vis_geometries = [self.cloud]

        # Process each detected human individually.
        for human_hypothesis in human_hypotheses:
            keypoint_annotations = robokudo.cas.CAS.filter_by_type(robokudo.types.human.KeypointAnnotation,
                                                                   human_hypothesis.annotations)
            keypoint_2d_annotations = [a for a in keypoint_annotations if
                                       a.type == robokudo.types.human.KeypointAnnotation.KP_TYPE_2D]

            # If no keypoints are available, skip this human.
            if len(keypoint_2d_annotations) == 0:
                continue

            keypoint_2d_annotations = keypoint_2d_annotations[0]

            # Evaluate whether the person is sitting.
            if self.keypoints_sitting(keypoint_2d_annotations):
                activity = robokudo.types.human.ActivityAnnotation()
                activity.type = "sitting"
                activity.source = type(self).__name__
                human_hypothesis.annotations.append(activity)

        self.get_annotator_output_struct().set_geometries(all_vis_geometries)
        self.get_annotator_output_struct().set_image(copy.deepcopy(self.color))

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return py_trees.common.Status.SUCCESS

    def keypoints_sitting(self, keypoint_2d_annotation: robokudo.types.human.KeypointAnnotation.KP_TYPE_2D):
        p = self.descriptor.parameters
        th = p.keypoint_confidence_threshold

        # Load keypoints for left and right body sides.
        left_hip = KeypointData(keypoint_2d_annotation.keypoints[HumanAndPoseAnnotator.GetKeypoint.LEFT_HIP])
        left_knee = KeypointData(keypoint_2d_annotation.keypoints[HumanAndPoseAnnotator.GetKeypoint.LEFT_KNEE])
        left_ankle = KeypointData(keypoint_2d_annotation.keypoints[HumanAndPoseAnnotator.GetKeypoint.LEFT_ANKLE])

        right_hip = KeypointData(keypoint_2d_annotation.keypoints[HumanAndPoseAnnotator.GetKeypoint.RIGHT_HIP])
        right_knee = KeypointData(keypoint_2d_annotation.keypoints[HumanAndPoseAnnotator.GetKeypoint.RIGHT_KNEE])
        right_ankle = KeypointData(keypoint_2d_annotation.keypoints[HumanAndPoseAnnotator.GetKeypoint.RIGHT_ANKLE])

        # Check whether keypoints on each side are reliable.
        left_ok = min(left_hip.confidence, left_knee.confidence, left_ankle.confidence) >= th
        right_ok = min(right_hip.confidence, right_knee.confidence, right_ankle.confidence) >= th
        #print("Linke Knie confidence: ",left_knee.confidence,"Linke ankle confidence: ", left_ankle.confidence, "Linke hüfte confidence: ",left_hip.confidence)

        # If neither side has reliable keypoints, sitting cannot be evaluated.
        if not left_ok and not right_ok:
            return False

        # Helper function to evaluate sitting posture for one body side.
        def side_is_sitting(hip, knee, ankle):
            ys = [hip.y, knee.y, ankle.y]
            body_h = max(ys) - min(ys)

            if body_h < 20:
                return False

            # Compute normalized vertical distances.
            hip_knee_rel = abs(hip.y - knee.y) / body_h
            knee_ankle_rel = abs(knee.y - ankle.y) / body_h

            knee_below_hip = knee.y > hip.y
            ankle_below_knee = ankle.y > knee.y

            # Sitting heuristic for one side:
            return (
                    knee_below_hip and
                    ankle_below_knee and
                    hip_knee_rel <= p.max_hip_knee_rel_bbox and
                    knee_ankle_rel >= p.min_ankle_knee_rel_bbox
            )

        # If either side satisfies the sitting heuristic, classify as sitting.
        if left_ok and side_is_sitting(left_hip, left_knee, left_ankle):
            return True
        if right_ok and side_is_sitting(right_hip, right_knee, right_ankle):
            return True

        # Optional: if both sides are visible, check whether knees are at similar height.
        if left_ok and right_ok:
            body_h = max(left_hip.y, left_knee.y, left_ankle.y, right_hip.y, right_knee.y, right_ankle.y) - min(left_hip.y, left_knee.y, left_ankle.y, right_hip.y, right_knee.y, right_ankle.y)
            if body_h > 20:
                knee_knee_rel = abs(left_knee.y - right_knee.y) / body_h
                if knee_knee_rel <= p.max_knee_knee_rel_bbox:
                    return side_is_sitting(left_hip, left_knee, left_ankle) or side_is_sitting(right_hip, right_knee, right_ankle)

        return False






























