from py_trees.composites import Sequence
import robokudo.analysis_engine
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.query import QueryAnnotator
import robokudo.annotators
from robokudo.idioms import pipeline_init
from robokudo.pipeline import Pipeline
from robokudo.tree_components.query_based_task_scheduler import QueryBasedScheduler
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from robokudo_robocup_gpsr.descriptors.analysis_engines.gsam_pipeline import GSamPipeline
from robokudo_robocup_gpsr.descriptors.analysis_engines.subtree_waving_raising_pipeline import \
    Subtree as Subtree_WavingRaisingPipeline
from robokudo_robocup_gpsr.descriptors.analysis_engines.subtree_human_posture_pipeline import \
    Subtree as Subtree_HumanPosturePipeline
from robokudo_robocup_gpsr.descriptors.analysis_engines.subtree_object_detection_pipeline import \
    Subtree as Subtree_ObjectDetectionPipeline
from robokudo_robocup_gpsr.descriptors.analysis_engines.subtree_pointing_pipeline import \
    Subtree as Subtree_PointingPipeline
from robokudo_robocup_gpsr.descriptors.analysis_engines.subtree_human_activity_pipeline import \
    Subtree as Subtree_Human_Activity
from robokudo_suturo.descriptors.analysis_engines.subtree_human_attribute_pipeline import \
    Subtree as Subtree_Human_Attribute
from robokudo_robocup_gpsr.descriptors.analysis_engines.subtree_human_near_object import \
    Subtree as Subtree_HumanNearObjectPipeline, FURNITURE
from robokudo_robocup_gpsr.descriptors.analysis_engines.subtree_human_in_room import \
    Subtree as Subtree_HumanInRoomPipeline
from robokudo_robocup_gpsr.descriptors.analysis_engines.subtree_obj_in_room import \
    Subtree as Subtree_ObjInRoomPipeline
from robokudo_robocup_receptionist.annotators.subtree_faces import \
    Subtree as Subtree_FacesPipeline
from robokudo_robocup_gpsr.descriptors.analysis_engines.subtree_belief_lookup import \
    Subtree as Subtree_BeliefLookup
from robokudo_robocup_gpsr.descriptors.analysis_engines.subtree_door_pipeline import \
    Subtree as Subtree_DoorPipeline
from robokudo_robocup_gpsr.descriptors.analysis_engines.subtree_door_pipeline_gsam import \
    Subtree as Subtree_DoorGsamPipeline


class AnalysisEngine(robokudo.analysis_engine.AnalysisEngineInterface):

    def __init__(self):
        super(AnalysisEngine, self).__init__()
        self.cr = None

    def name(self):
        return "gpsr"

    def configure_collection_reader(self):
        raise "Trying to instantiate gpsr_implementation directly. You must overwrite this method."

    @catch_and_raise_to_blackboard
    def implementation(self):
        """
        Creates a generic pipeline based on the entries within the received query.
        """
        self.configure_collection_reader()

        # The actual attributes for GPSR can be found in the CommandGenerator. See
        # https://github.com/RoboCupAtHome/CommandGenerator/tree/master for details. Check the rulebook to see
        # if the repository is the current correct one for GPSR.
        waving_raising_attributes = ["waving", "raising right arm", "raising left arm"]
        human_posture_attributes = ["sitting", "standing", "laying"]
        pointing_attributes = ["pointing right", "pointing left"]
        clothes_attributes = ["t shirt", "shirt", "blouse", "sweater", "coat", "jacket"]
        color_attributes = ["blue", "yellow", "black", "white", "red", "orange", "gray"]
        object_attributes = ["biggest", "largest", "smallest", "heaviest", "lightest", "thinnest"]
        furniture = ["chair","couch"]
        rooms = ["iai_kitchen", "living_room", "bedroom", "office", "hallway"]

        def filter_query_input(type):
            """
            Filters the query input to ensure it is valid.
            :param type: The type of the query input.
            :return: True if the query input is valid, False otherwise.
            """
            if len(type) > 1:
                return None
            elif len(type) == 1:
                return type[0]
            return None

        def task_selection(query):
            if not query.obj.type:
                return None

            attribute = filter_query_input(query.obj.attribute)
            color = filter_query_input(query.obj.color)
            description = filter_query_input(query.obj.description)

            #TODO query.obj.type = human base case so anpassen, dass wir noch ein fall haben, dass er filtern soll
            if query.obj.type == 'human':
                if color in color_attributes and description in clothes_attributes:
                    if query.obj.location in rooms:
                        return 'GSamPipelineWithRegion'
                    return 'GSamPipeline'
                elif query.obj.location in rooms:
                    if attribute in waving_raising_attributes + human_posture_attributes + pointing_attributes:
                        return 'HumanActivityInRoomPipeline'
                    return 'HumanInRoomPipeline'
                elif query.obj.location in furniture:
                    return 'HumanNearObjectPipeline'
                elif attribute in waving_raising_attributes:
                    return 'WavingRaisingPipeline'
                elif attribute in human_posture_attributes:
                    return 'HumanPosturePipeline'
                elif attribute in pointing_attributes:
                    return 'PointingPipeline'
                elif attribute == "Attribute":
                    return 'Attribute'
                elif len(query.obj.attribute) >= 2:
                    return 'HumanActivityFiltered'
                else:
                    return 'HumanActivity'
            elif query.obj.type == 'recall':
                return 'BeliefStateLookupPipeline'
            elif query.obj.type == 'face':
                return 'FacePipeline'
            elif query.obj.type == 'door':
                return 'DoorPipeline'
            elif query.obj.type == 'doorgsam':
                return 'DoorGsamPipeline'
            elif query.obj.type == 'object':
                if query.obj.location in rooms:
                    return 'ObjInRoomPipeline'
                elif attribute in object_attributes:
                    return 'GSamPipeline'
                elif attribute and not attribute in object_attributes:
                    return None
                else:
                    return 'ObjectDetectionPipeline'
            else:
                return None

        task_trees = {
            'WavingRaisingPipeline': Subtree_WavingRaisingPipeline().implementation(),
            'HumanPosturePipeline': Subtree_HumanPosturePipeline(human_posture_attributes).implementation(),
            'ObjectDetectionPipeline': Subtree_ObjectDetectionPipeline().implementation(),
            'PointingPipeline': Subtree_PointingPipeline().implementation(),
            'GSamPipeline': GSamPipeline(object_attributes=object_attributes, color_attributes=color_attributes,
                                         clothes_attributes=clothes_attributes),
            'GSamPipelineWithRegion': GSamPipeline(object_attributes=object_attributes, color_attributes=color_attributes,
                                                   clothes_attributes=clothes_attributes, with_region_filter=True),
            'HumanActivityFiltered': Subtree_Human_Activity(with_filter=True).implementation(),
            'HumanActivity' : Subtree_Human_Activity().implementation(),
            'Attribute' : Subtree_Human_Attribute().implementation(),
            'HumanNearObjectPipeline': Subtree_HumanNearObjectPipeline().implementation(),
            'HumanInRoomPipeline': Subtree_HumanInRoomPipeline().implementation(),
            'HumanActivityInRoomPipeline': Subtree_HumanInRoomPipeline(with_activity=True).implementation(),
            'ObjInRoomPipeline': Subtree_ObjInRoomPipeline().implementation(),
            'FacePipeline': Subtree_FacesPipeline(ros_pkg_path='robokudo_robocup_gpsr').implementation(),
            'BeliefStateLookupPipeline': Subtree_BeliefLookup().implementation(),
            'DoorPipeline': Subtree_DoorPipeline().implementation(),
            'DoorGsamPipeline': Subtree_DoorGsamPipeline().implementation(),
        }

        task_scheduling = Sequence("Task Scheduling", memory=True)
        task_scheduling.add_child(QueryBasedScheduler(tasks=task_trees, filter_fn=task_selection))

        seq = Pipeline('GPSR')
        seq.add_children(
            [
                pipeline_init(),
                QueryAnnotator(),
                self.cr,
                ImagePreprocessorAnnotator(),
                task_scheduling,
            ]
        )
        return seq
