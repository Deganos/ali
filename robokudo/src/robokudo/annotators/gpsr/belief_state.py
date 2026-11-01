import json
import math
import time
from enum import Enum
from pathlib import Path

import py_trees

import robokudo.annotators.core
import robokudo.types.annotation
import robokudo.types.human
import robokudo.types.scene
import robokudo.types.cv
from robokudo.utils.annotator_helper import transform_pose_from_cam_to_world


class EntityStatus(str, Enum):
    VISIBLE = "visible"
    PROBABLY_THERE = "probably_there"


class EntityType(str, Enum):
    PERSON = "person"
    OBJECT = "object"


class BeliefEntry:

    def __init__(self, entry_id, entity_type, classname):
        self.id = entry_id
        self.entity_type = entity_type
        self.classname = classname
        self.yolo_label = ""
        self.semantic_labels = []
        self.activities = []
        self.position = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, 0.0, 1.0]
        self.dimensions = [0.0, 0.0, 0.0]
        self.last_seen = 0.0
        self.confidence = 0.0
        self.status = EntityStatus.VISIBLE
        self.name = "anonymous"

    def to_dict(self):
        base = {}
        base["id"] = self.id
        base["entity_type"] = self.entity_type.value
        base["position"] = {
            "x": self.position[0],
            "y": self.position[1],
            "z": self.position[2]
        }
        base["orientation"] = {
            "x": self.orientation[0],
            "y": self.orientation[1],
            "z": self.orientation[2],
            "w": self.orientation[3]
        }
        base["dimensions"] = {
            "x": self.dimensions[0],
            "y": self.dimensions[1],
            "z": self.dimensions[2]
        }
        base["last_seen"] = time.strftime("%H:%M:%S", time.localtime(self.last_seen))
        base["confidence"] = self.confidence
        base["status"] = self.status.value
        base["activities"] = list(self.activities)

        if self.entity_type == EntityType.PERSON:
            base["name"] = self.name
        else:
            base["classname"] = self.classname
            base["yolo_label"] = self.yolo_label
            base["semantic_labels"] = list(self.semantic_labels)

        return base


class BeliefState:

    def __init__(self):
        self.entries = {}

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries.values())

    def find_by_id(self, entry_id):
        return self.entries.get(entry_id)

    def find_by_classname(self, classname):
        result = []
        for entry in self.entries.values():
            if entry.classname == classname:
                result.append(entry)
        return result

    def find_by_yolo_label(self, label):
        result = []
        for entry in self.entries.values():
            if entry.yolo_label == label:
                result.append(entry)
        return result

    def find_by_semantic_label(self, label):
        result = []
        for entry in self.entries.values():
            if label in entry.semantic_labels:
                result.append(entry)
        return result

    # dummy TODO
    def find_by_activity(self, activity):
        result = []
        for entry in self.entries.values():
            if activity in entry.activities:
                result.append(entry)
        return result

    # dummy TODO
    def find_by_name(self, name):
        result = []
        for entry in self.entries.values():
            if entry.name == name:
                result.append(entry)
        return result

    def find_by_status(self, status):
        result = []
        for entry in self.entries.values():
            if entry.status == status:
                result.append(entry)
        return result

    # dummy TODO
    def visible(self):
        return self.find_by_status(EntityStatus.VISIBLE)

    def persons(self):
        result = []
        for entry in self.entries.values():
            if entry.entity_type == EntityType.PERSON:
                result.append(entry)
        return result

    # dummy TODO
    def objects(self):
        result = []
        for entry in self.entries.values():
            if entry.entity_type == EntityType.OBJECT:
                result.append(entry)
        return result

    # dummy TODO
    def find_nearest(self, position, entity_type=None, classname=None):
        best = None
        best_dist = float("inf")
        for entry in self.entries.values():
            if entity_type is not None and entry.entity_type != entity_type:
                continue
            if classname is not None and entry.classname != classname:
                continue
            dist = _distance(entry.position, position)
            if dist < best_dist:
                best_dist = dist
                best = entry
        return best


def _distance(a, b):
    total = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        total += diff * diff
    return math.sqrt(total)


class BeliefStateAnnotator(robokudo.annotators.core.BaseAnnotator):

    _state = BeliefState()
    _next_id = 1

    def __init__(
        self,
        name="BeliefStateAnnotator",
        match_distance=0.3,
        probably_there_after=60.0,
        person_delete_after=120.0,
        object_delete_after=300.0,
        debug_json_path=str(Path.home() / "Desktop" / "belief_state.json"),
    ):
        super().__init__(name=name)
        self.match_distance = match_distance
        self.probably_there_after = probably_there_after
        self.person_delete_after = person_delete_after
        self.object_delete_after = object_delete_after
        self.debug_json_path = debug_json_path

    def update(self):
        now = time.time()
        cas = self.get_cas()

        for hyp in list(cas.annotations):
            if isinstance(hyp, robokudo.types.scene.HumanHypothesis):
                entity_type = EntityType.PERSON
            elif isinstance(hyp, robokudo.types.scene.ObjectHypothesis):
                entity_type = EntityType.OBJECT
            else:
                continue

            observation = self._extract_observation(cas, hyp, entity_type)
            if observation is None:
                continue

            self._integrate(observation, entity_type, now, hyp=hyp)

        self._age_entries(now)

        cas.set_ref("belief_state", self._state)

        if self.debug_json_path:
            self._dump_debug_json()

        self.feedback_message = f"Belief state: {len(self._state)} entries"
        return py_trees.common.Status.SUCCESS

    def _integrate(self, observation, entity_type, now, hyp=None):
        match = self._find_match(observation, entity_type)

        if match is None:
            entry = BeliefEntry(self._next_id, entity_type, observation["classname"])
            entry.yolo_label = observation["yolo_label"]
            entry.semantic_labels = list(observation["semantic_labels"])
            entry.activities = list(observation["activities"])
            entry.position = observation["position"]
            entry.orientation = observation["orientation"]
            entry.dimensions = observation["dimensions"]
            entry.last_seen = now
            entry.confidence = observation["confidence"]
            entry.name = observation["name"]
            self._state.entries[self._next_id] = entry
            self._next_id += 1
        else:
            entry = match
            entry.position = observation["position"]
            entry.orientation = observation["orientation"]
            entry.dimensions = observation["dimensions"]
            entry.last_seen = now
            entry.confidence = observation["confidence"]
            entry.status = EntityStatus.VISIBLE

            if observation["yolo_label"]:
                entry.yolo_label = observation["yolo_label"]

            for label in observation["semantic_labels"]:
                if label not in entry.semantic_labels:
                    entry.semantic_labels.append(label)

            entry.activities = list(observation["activities"])

            if observation["name"] != "anonymous":
                entry.name = observation["name"]

        if hyp is not None:
            hyp.annotations.append(entry)

    def _extract_observation(self, cas, hyp, entity_type):
        yolo_label = ""
        confidence = 0.0
        semantic_labels = []
        activities = []
        name = "anonymous"
        position = None
        orientation = [0.0, 0.0, 0.0, 1.0]
        dimensions = [0.0, 0.0, 0.0]

        for ann in hyp.annotations:
            if isinstance(ann, robokudo.types.human.ActivityAnnotation):
                if ann.type and ann.type not in activities:
                    activities.append(ann.type)
                continue

            if isinstance(ann, robokudo.types.annotation.Classification):
                source = getattr(ann, "source", "") or ""
                if ann.classification_type == "face":
                    name = ann.classname
                elif "ZeroShot" in source:
                    if ann.classname and ann.classname not in semantic_labels:
                        semantic_labels.append(ann.classname)
                else:
                    if not yolo_label:
                        yolo_label = ann.classname
                        confidence = float(ann.confidence)

            if isinstance(ann, robokudo.types.annotation.PoseAnnotation):
                try:
                    world_pose = transform_pose_from_cam_to_world(cas, ann)
                    position = list(world_pose.translation)
                    orientation = list(world_pose.rotation)
                except Exception:
                    position = list(ann.translation)
                    orientation = list(ann.rotation)
            elif isinstance(ann, robokudo.types.annotation.PositionAnnotation) and position is None:
                pose_like = robokudo.types.annotation.PoseAnnotation()
                pose_like.translation = list(ann.translation)
                pose_like.rotation = [0.0, 0.0, 0.0, 1.0]
                try:
                    world_pose = transform_pose_from_cam_to_world(cas, pose_like)
                    position = list(world_pose.translation)
                except Exception:
                    position = list(ann.translation)

            if isinstance(ann, robokudo.types.annotation.BoundingBox3DAnnotation):
                dimensions = [ann.x_length, ann.y_length, ann.z_length]

        if position is None:
            return None

        if yolo_label:
            classname = yolo_label
        elif entity_type == EntityType.PERSON:
            classname = "human"
        else:
            classname = "unknown"

        observation = {}
        observation["classname"] = classname
        observation["yolo_label"] = yolo_label
        observation["semantic_labels"] = semantic_labels
        observation["activities"] = activities
        observation["confidence"] = confidence
        observation["name"] = name
        observation["position"] = position
        observation["orientation"] = orientation
        observation["dimensions"] = dimensions

        return observation

    def _find_match(self, observation, entity_type):
        best = None
        best_dist = self.match_distance
        for entry in self._state.entries.values():
            if entry.entity_type != entity_type:
                continue
            if entry.classname != observation["classname"]:
                continue
            dist = _distance(entry.position, observation["position"])
            if dist <= best_dist:
                best_dist = dist
                best = entry
        return best

    def _age_entries(self, now):
        to_delete = []
        for entry_id, entry in self._state.entries.items():
            age = now - entry.last_seen
            if entry.entity_type == EntityType.PERSON and age > self.person_delete_after:
                to_delete.append(entry_id)
            elif entry.entity_type == EntityType.OBJECT and age > self.object_delete_after:
                to_delete.append(entry_id)
            elif age > self.probably_there_after:
                entry.status = EntityStatus.PROBABLY_THERE

        for entry_id in to_delete:
            del self._state.entries[entry_id]

    def _dump_debug_json(self):
        try:
            payload = {}
            for entry in self._state.entries.values():
                payload[str(entry.id)] = entry.to_dict()
            with open(self.debug_json_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            self.rk_logger.warning(f"Could not write belief state debug JSON: {exc}")
