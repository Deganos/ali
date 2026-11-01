from timeit import default_timer

import cv2
import numpy as np
import open3d as o3d
import py_trees

import robokudo.annotators.core
import robokudo.types.annotation
import robokudo.utils.error_handling
import robokudo.utils.transform
from robokudo.cas import CAS, CASViews
from robokudo.io.camera_model_adapters import Open3DCameraModelAdapter
from robokudo.types.scene import ObjectHypothesis

DOOR_CLASSES = ('door', 'cabinet door', 'refrigerator door')
HANDLE_CLASS = 'handle'
GRASP_CLASS = 'grasp_point'


class DoorHandlePoseAnnotator(robokudo.annotators.core.BaseAnnotator):
    """Estimate a grasp pose for door handles.

    The door is a plane, the handle is what sticks out of it. A RANSAC plane fit on
    the door region gives the approach direction, the points in front of that plane
    give the handle geometry, and a PCA on those points gives its axis.
    """

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                # Depth values are stored in millimetres by the camera interfaces.
                self.depth_scale = 0.001
                self.plane_distance_threshold = 0.02
                self.ransac_iterations = 200
                # A handle sticks out of the door by roughly this much.
                self.min_protrusion = 0.02
                self.max_protrusion = 0.12
                self.min_handle_points = 40
                self.valid_depth_range = (0.3, 4.0)
                # Extra room per jaw beyond the measured handle thickness.
                self.jaw_clearance = 0.015
                # Contact patch of one finger: width along the handle, depth towards the door.
                self.jaw_pad_width = 0.04
                self.jaw_pad_depth = 0.035
                # Air between the outer edge of the pad and the free end of the handle.
                self.tip_clearance = 0.01
                # Below this difference in protrusion both ends of the handle look alike.
                self.min_protrusion_gradient = 0.005
                # Draw the door plane and the rejected detections as well.
                self.debug_view = False
                # Drop handle detections without a grasp pose from the query result.
                self.drop_handles_without_pose = True

        parameters = Parameters()

    def __init__(self, name="DoorHandlePoseAnnotator", descriptor=Descriptor()):
        super(DoorHandlePoseAnnotator, self).__init__(name, descriptor)

    def get_hypotheses_by_classname(self, classnames):
        result = []
        for hypothesis in self.get_cas().filter_annotations_by_type(ObjectHypothesis):
            classifications = CAS.filter_by_type(robokudo.types.annotation.Classification,
                                                 hypothesis.annotations)
            for classification in classifications:
                if classification.classname in classnames:
                    result.append((hypothesis, classification))
                    break
        return result

    def roi_box(self, hypothesis, width, height):
        roi = hypothesis.roi.roi
        x = int(max(roi.pos.x, 0))
        y = int(max(roi.pos.y, 0))
        w = int(min(roi.width, width - x))
        h = int(min(roi.height, height - y))
        return x, y, w, h

    def points_in_box(self, box):
        """Backproject the valid depth pixels of one image region into camera coordinates."""
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return np.empty((0, 3)), np.empty((0, 2), dtype=int)

        depth = self.depth[y:y + h, x:x + w]
        lo, hi = self.descriptor.parameters.valid_depth_range
        valid = np.isfinite(depth) & (depth > lo) & (depth < hi)
        if not valid.any():
            return np.empty((0, 3)), np.empty((0, 2), dtype=int)

        vs, us = np.nonzero(valid)
        us = us + x
        vs = vs + y
        z = depth[valid]
        points = np.stack([(us - self.cx) * z / self.fx,
                           (vs - self.cy) * z / self.fy,
                           z], axis=1)
        return points, np.stack([us, vs], axis=1)

    def fit_plane(self, points):
        if len(points) < 100:
            return None
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        model, inliers = cloud.segment_plane(
            distance_threshold=self.descriptor.parameters.plane_distance_threshold,
            ransac_n=3,
            num_iterations=self.descriptor.parameters.ransac_iterations)
        a, b, c, d = model
        normal = np.array([a, b, c], dtype=float)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            return None
        normal, d = normal / length, d / length
        if normal[2] > 0:  # let the normal point back towards the camera
            normal, d = -normal, -d
        return normal, d, len(inliers)

    def free_end_sign(self, along, protrusion, axis, centroid, door_center):
        """Which way along the axis the handle ends free.

        Towards the rosette the handle curves back into the door, so its protrusion drops.
        When both ends look alike, fall back on the door: a lever points to the door centre.
        """
        edge = max(int(0.1 * len(along)), 5)
        order = np.argsort(along)
        low = float(protrusion[order[:edge]].mean())
        high = float(protrusion[order[-edge:]].mean())
        if abs(high - low) >= self.descriptor.parameters.min_protrusion_gradient:
            return (1.0 if high > low else -1.0), "protrusion"
        if door_center is None:
            return 1.0, "no hint"
        return (1.0 if np.dot(door_center - centroid, axis) > 0 else -1.0), "door centre"

    def grasp_pose_from_points(self, points, protrusion, normal, door_center):
        """Build a right handed frame: x approaches the door, y follows the handle axis."""
        centroid = points.mean(axis=0)
        centered = points - centroid
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
        axis = eigenvectors[:, eigenvalues.argmax()]

        approach = -normal
        axis = axis - np.dot(axis, approach) * approach
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            return None
        axis = axis / norm

        along = centered @ axis
        sign, reason = self.free_end_sign(along, protrusion, axis, centroid, door_center)
        axis = sign * axis  # from here on the axis points at the free end

        third = np.cross(approach, axis)
        rotation = np.column_stack([approach, axis, third])

        # Sit one pad half plus some air inside the tip, never past the centroid.
        margin = 0.5 * self.descriptor.parameters.jaw_pad_width + \
            self.descriptor.parameters.tip_clearance
        tip = float(np.percentile(sign * along, 98))  # robust against single stray points
        offset = max(tip - margin, 0.0)
        translation = centroid + offset * axis
        length = float(along.max() - along.min())

        # How thick the handle is across the closing direction. The jaws have to clear that.
        spread = float(np.abs(centered @ third).max())
        return centroid, translation, rotation, axis, spread, length, offset, reason

    def frame_from_normal(self, normal):
        """Frame of a flat surface: x approaches it, y points up in the image."""
        approach = -normal
        up = np.array([0.0, -1.0, 0.0])
        axis = up - np.dot(up, approach) * approach
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            return None
        axis = axis / norm
        return np.column_stack([approach, axis, np.cross(approach, axis)])

    def pose_on_plane(self, points, normal, plane_d):
        """Centre of the plane region, snapped exactly onto the plane."""
        signed = points @ normal + plane_d
        inliers = points[np.abs(signed) < self.descriptor.parameters.plane_distance_threshold]
        if len(inliers) == 0:
            return None
        rotation = self.frame_from_normal(normal)
        if rotation is None:
            return None
        centroid = inliers.mean(axis=0)
        return centroid - (np.dot(centroid, normal) + plane_d) * normal, rotation

    def grasp_hypothesis(self, translation, rotation):
        """Own hypothesis, so the grasp point becomes a separate entry in the query result."""
        hypothesis = ObjectHypothesis()
        hypothesis.type = GRASP_CLASS
        classification = robokudo.types.annotation.Classification()
        classification.classname = GRASP_CLASS
        classification.source = type(self).__name__
        hypothesis.annotations.append(classification)
        self.add_pose(hypothesis, translation, rotation, GRASP_CLASS)
        return hypothesis

    def add_pose(self, hypothesis, translation, rotation, source):
        """Attach one pose to a hypothesis. The source label survives into pose_source."""
        pose_annotation = robokudo.types.annotation.PoseAnnotation()
        pose_annotation.translation = list(translation)
        pose_annotation.rotation = list(
            robokudo.utils.transform.get_quaternion_from_rotation_matrix(rotation))
        pose_annotation.source = source
        hypothesis.annotations.append(pose_annotation)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def update(self):
        start_timer = default_timer()

        cas = self.get_cas()
        color = cas.get_copy(CASViews.COLOR_IMAGE)
        depth = cas.get(CASViews.DEPTH_IMAGE)
        intrinsics = Open3DCameraModelAdapter.to_intrinsic(cas.require_camera_observation().effective_camera_model)

        self.depth = np.asarray(depth, dtype=np.float32) * self.descriptor.parameters.depth_scale
        k = intrinsics.intrinsic_matrix
        self.fx, self.fy, self.cx, self.cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]

        height, width = self.depth.shape[:2]
        doors = self.get_hypotheses_by_classname(DOOR_CLASSES)
        handles = self.get_hypotheses_by_classname([HANDLE_CLASS])

        geometries = []
        messages = []

        plane = None
        plane_box = None
        door_hypothesis = None
        door_points = np.empty((0, 3))
        if doors:
            door_hypothesis = max(doors, key=lambda dh: dh[0].roi.roi.width * dh[0].roi.roi.height)[0]
            plane_box = self.roi_box(door_hypothesis, width, height)
            door_points, _ = self.points_in_box(plane_box)
            plane = self.fit_plane(door_points)
        if plane is None:
            plane_box = (0, 0, width, height)
            door_points, _ = self.points_in_box(plane_box)
            plane = self.fit_plane(door_points)
            if doors:
                messages.append("door plane fit fell back to the full image")

        if plane is None:
            messages.append("no plane found")
            self.draw(color, plane_box, [], messages)
            self.finish(start_timer, geometries, messages)
            return py_trees.common.Status.SUCCESS

        normal, plane_d, inlier_count = plane
        tilt = np.degrees(np.arccos(min(abs(normal[2]), 1.0)))
        messages.append(f"plane: {inlier_count} inliers, tilt {tilt:.1f} deg")

        door_center = None
        if door_hypothesis is not None:
            door_pose = self.pose_on_plane(door_points, normal, plane_d)
            if door_pose is None:
                messages.append("no door pose")
            else:
                door_center = door_pose[0]
                self.add_pose(door_hypothesis, door_center, door_pose[1], "door")

        results = []
        grasped = []
        grasp_hypotheses = []
        for hypothesis, classification in handles:
            box = self.roi_box(hypothesis, width, height)
            points, pixels = self.points_in_box(box)
            if len(points) == 0:
                results.append((box, None, "no depth data on the handle"))
                continue

            signed = points @ normal + plane_d
            selected = ((signed > self.descriptor.parameters.min_protrusion) &
                        (signed < self.descriptor.parameters.max_protrusion))
            count = int(selected.sum())
            if count < self.descriptor.parameters.min_handle_points:
                results.append((box, None,
                                f"only {count} points in front of the plane "
                                f"({100.0 * count / max(len(points), 1):.0f} % of the box)"))
                continue

            pose = self.grasp_pose_from_points(points[selected], signed[selected],
                                               normal, door_center)
            if pose is None:
                results.append((box, None, "handle axis is degenerate"))
                continue

            centroid, translation, rotation, axis, spread, length, offset, reason = pose
            half_gap = spread + self.descriptor.parameters.jaw_clearance
            self.add_pose(hypothesis, centroid, rotation, "handle")
            grasp_hypotheses.append(self.grasp_hypothesis(translation, rotation))

            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            frame.transform(transform)
            geometries.append(frame)

            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            marker.translate(translation)
            marker.paint_uniform_color([0.0, 0.35, 1.0])
            marker.compute_vertex_normals()
            geometries.append(marker)

            protrusion = float(np.median(signed[selected]))
            horizontal = np.degrees(np.arccos(min(abs(axis[0]), 1.0)))
            orientation = "horizontal" if horizontal < 45 else "vertical"

            reach = self.descriptor.parameters.jaw_pad_depth + self.descriptor.parameters.tip_clearance
            if protrusion < reach:
                messages.append(f"fingers reach {reach * 100:.1f} cm but the handle only "
                                f"stands {protrusion * 100:.1f} cm out")

            grasped.append(hypothesis)
            results.append((box, (pixels[selected], translation),
                            f"{orientation}, {protrusion * 100:.1f} cm out, "
                            f"{translation[2]:.2f} m away, "
                            f"jaws {2 * half_gap * 100:.1f} cm apart, "
                            f"handle {length * 100:.1f} cm, grasp {offset * 100:.1f} cm "
                            f"off centre towards the free end ({reason})"))

        if not handles:
            messages.append("no handle detected")
        messages.append(f"{len(grasped)} of {len(handles)} handles with grasp pose")

        # The image only carries the dot, the numbers go to the terminal.
        for _, payload, note in results:
            self.rk_logger.info(f"{'grasp' if payload else 'rejected'}: {note}")

        if self.descriptor.parameters.drop_handles_without_pose:
            rejected = {id(h) for h, _ in handles} - {id(h) for h in grasped}
            cas.annotations = [a for a in cas.annotations if id(a) not in rejected]
        cas.annotations.extend(grasp_hypotheses)

        self.draw(color, plane_box, results, messages)
        self.finish(start_timer, geometries, messages)
        return py_trees.common.Status.SUCCESS

    def project(self, point):
        if point[2] <= 1e-6:
            return None
        return (int(round(point[0] * self.fx / point[2] + self.cx)),
                int(round(point[1] * self.fy / point[2] + self.cy)))

    def draw_grasp_point(self, image, translation):
        """One dot on the spot the gripper should close on. Nothing else."""
        origin = self.project(translation)
        if origin is None:
            return
        cv2.circle(image, origin, 10, (255, 255, 255), -1)
        cv2.circle(image, origin, 9, (255, 90, 0), -1)

    def draw(self, image, plane_box, results, messages):
        debug = self.descriptor.parameters.debug_view
        if debug and plane_box is not None:
            x, y, w, h = plane_box
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 160, 0), 2)

        for box, payload, note in results:
            x, y, w, h = box
            if payload is None:
                if debug:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                continue
            pixels, translation = payload
            image[pixels[:, 1], pixels[:, 0]] = (0, 255, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 0), 2)
            self.draw_grasp_point(image, translation)

        self.get_annotator_output_struct().set_image(image)

    def finish(self, start_timer, geometries, messages):
        if geometries and self.get_cas().contains(CASViews.CLOUD):
            cloud = self.get_cas().get(CASViews.CLOUD)
            self.get_annotator_output_struct().set_geometries([cloud] + geometries)
        self.feedback_message = (f'{"; ".join(messages)} | '
                                 f'Processing took {(default_timer() - start_timer):.4f}s')
