import logging
import queue

import rclpy.action
import py_trees
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
import queue
import logging
import robokudo_msgs.msg
import geometry_msgs
import robokudo.annotators.core
import robokudo.types.scene
import robokudo.types.cv
from rclpy.action import ActionServer
from robokudo.utils.query import QueryHandler
from robokudo.cas import CASViews
import robokudo.defs
import robokudo.utils.error_handling
from robokudo.types.annotation import PoseAnnotation, PositionAnnotation
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world
from robokudo_msgs.msg import ShapeSize
from geometry_msgs.msg import Vector3
from robokudo_msgs.action import Query
from robokudo.annotators.gpsr.belief_state import BeliefEntry
from robokudo.annotators.gpsr.gpsr_results import camera_header


# class QueryAnnotator(robokudo.annotators.core.BaseAnnotator):
#     """
#     This Annotator spawns an Action Server that listens for Queries from external ROS nodes.
#     It will then annotate the CAS and put the Query into CASViews.QUERY.
#
#     The Annotator and the Actionserver are type-agnostic, which means that you are not bound
#     to a specific type of query. You can pass these from your AE to this QueryAnnotator.
#
#     Look up the QueryReply Class to see how to the result is generated.
#     """
#
#     def __init__(self, name="QueryAnnotator",
#                  feedback_instance=Query.Feedback(),
#                  result_instance=Query.Result(),
#                  action_type=Query,
#                  action_name="~query",
#                  ):
#         super().__init__(name)
#         self.query_service = None
#         self.feedback_instance = feedback_instance
#         self.result_instance = result_instance
#         self.action_type = action_type
#         self.action_name = action_name
#
#     def initialise(self):
#         self.rk_logger.debug("%s.initialise()" % self.__class__.__name__)
#
#         blackboard = py_trees.blackboard.Blackboard()
#         self.action_server = blackboard.get(BBIdentifier.QUERY_SERVER)
#         if self.action_server is None:
#             # TODO the name should be under the node's namespace
#             self.action_server = QueryActionServer(name=self.action_name,
#                                                    feedback_instance=self.feedback_instance,
#                                                    result_instance=self.result_instance,
#                                                    action_type=self.action_type, )
#             blackboard.set(BBIdentifier.QUERY_SERVER, self.action_server)
#
#     def update(self):
#         self.rk_logger.debug("%s.update()" % (self.__class__.__name__))
#
#         assert self.action_server is not None  # this should be initialized by now
#
#         query = self.action_server.new_query
#         if query:
#             self.feedback_message = 'Query: ' + str(query)
#             self.get_cas().set(CASViews.QUERY, query)
#             self.action_server.start_processing()
#             return py_trees.common.Status.SUCCESS
#
#         self.feedback_message = 'Waiting for query'
#
#         return py_trees.common.Status.RUNNING
#
#
# class QueryReply(robokudo.annotators.core.BaseAnnotator):
#     """
#     A test class which simply generates an empty Query Answer to check
#     if the Action server can reply properly.
#     Create a single, empty Object Designator that will be send to the caller.
#     """
#
#     def __init__(self, name="QueryReply"):
#         super().__init__(name=name)
#
#     def initialise(self):
#         self.rk_logger.debug("%s.initialise()" % self.__class__.__name__)
#
#     def update(self):
#         self.rk_logger.debug("%s.update()" % (self.__class__.__name__))
#         blackboard = py_trees.blackboard.Blackboard()
#         result = Query.Result()
#         od = robokudo_msgs.msg.ObjectDesignator()
#
#         pose_stamped = geometry_msgs.msg.PoseStamped()
#
#         pose_stamped.pose.position.x = 1
#         pose_stamped.pose.position.y = 2
#         pose_stamped.pose.position.z = 3
#
#         pose_stamped.pose.orientation.x = 0
#         pose_stamped.pose.orientation.y = 0
#         pose_stamped.pose.orientation.z = 0
#         pose_stamped.pose.orientation.w = 1
#
#         od.pose.append(pose_stamped)
#         result.res = [od]
#         blackboard.set(BBIdentifier.QUERY_ANSWER, result)
#
#         return py_trees.common.Status.SUCCESS


# class QueryActionServer(Node):
#     def __init__(self, name,
#                  feedback_instance=Query.Feedback(),
#                  result_instance=Query.Result(),
#                  action_type=Query,
#                  ):
#         super().__init__('query_action_server_node')
#
#         """
#         Construct the action server to handle queries coming in via ROS actions.
#         Please be aware, that action server callbacks are happening async to your Behavior Tree.
#         This means, that changes in the state of the action server can happen at any time during the tree tick.
#
#         :param name: the name of the action i.e the topic you can reach it with. For example: /robokudo/query
#         :param feedback_instance: An instance of the feedback message for your Action
#         :param result_instance:  An instance of the result message for your Action
#         :param action_type: The type of your action. Not an instance! e.g. robokudo_msgs.msg.QueryAction
#         """
#         self._action_name = name
#         #self._as = actionlib.SimpleActionServer(self._action_name, action_type, execute_cb=self.execute_cb, auto_start=False)
#         self._as = ActionServer(node=self, action_type=Query, action_name=self._action_name, execute_callback=self.execute_cb, goal_callback=self.goal_cb,
#             handle_accepted_callback=self.handle_accepted_cb)
#         self.reset_bookkeeping_vars()
#         self.new_query = None
#         self.query = None
#         self.preempt_message_printed = False
#         self._feedback = feedback_instance
#         self._result = result_instance
#
#     def reset_bookkeeping_vars(self):
#         self.query = None
#         self.new_query = None
#         self.preempt_message_printed = False
#         py_trees.blackboard.Blackboard().set(BBIdentifier.QUERY_ANSWER, None)
#         py_trees.blackboard.Blackboard().set(BBIdentifier.QUERY_FEEDBACK, queue.Queue())
#         py_trees.blackboard.Blackboard().set(BBIdentifier.QUERY_PREEMPT_REQUESTED, False)
#         py_trees.blackboard.Blackboard().set(BBIdentifier.QUERY_PREEMPT_ACK, False)
#         # feedback_queue : queue.Queue = py_trees.Blackboard().get(BBIdentifier.QUERY_FEEDBACK)
#         # feedback_queue.queue.clear()
#
#     def send_result(self, result):
#         self._result = result
#         self.reset_bookkeeping_vars()
#
#         self._as.set_succeeded(self._result)
#
#     def is_active(self):
#         return self.query is not None
#
#     def start_processing(self):
#         """
#         Tell the ActionServer that we are now starting the execution and it can start the monitoring/response process.
#         """
#         self.new_query = None
#
#     def goal_cb(self, goal_request):
#         """Called when a new goal is received."""
#         self.get_logger().info(f'Received goal request: {goal_request}')
#         return GoalResponse.ACCEPT  # oder REJECT
#
#     def handle_accepted_cb(self, goal_handle):
#         """Called when a goal is accepted, runs in a separate thread."""
#         self.get_logger().info('Goal accepted, executing...')
#         # Starte die Ausführung in einem separaten Thread
#         goal_handle.execute()
#
#     def execute_cb(self, goal_handle):
#         """Die Callback-Funktion für ROS2 Actions."""
#         self.new_query = goal_handle.request
#         self.query = goal_handle.request
#
#         feedback_queue: queue.Queue = py_trees.blackboard.Blackboard().get('QUERY_FEEDBACK')
#
#         rate = self.create_rate(10)  # z.B. 10 Hz
#
#         while True:
#             # Preempt / cancel handling
#             if goal_handle.is_cancel_requested:
#                 self.reset_bookkeeping_vars()
#                 goal_handle.canceled()
#                 return Query.Result()
#
#             # Feedback senden
#             try:
#                 feedback = feedback_queue.get_nowait()
#                 goal_handle.publish_feedback(feedback)
#             except queue.Empty:
#                 pass
#
#             # Prüfen, ob Ergebnis bereit ist
#             answer = py_trees.blackboard.Blackboard().get('QUERY_ANSWER')
#             if answer is not None:
#                 break
#
#             rate.sleep()
#
#         self.reset_bookkeeping_vars()
#         goal_handle.succeed()
#         return answer

# """
# def execute_cb(self, goal):
#     # Warning: Exception happening in this callback might not crash your program and hide.
#     # When it is not executed completely, look for lines that throw.
#
#     # publish info to the console for the user
#     rk_logger = logging.getLogger(robokudo.defs.PACKAGE_NAME)
#     rk_logger.info(f"Received query: {goal}")
#     self.new_query = goal
#     self.query = goal
#
#     rate = self.create_rate(50)  # in hz
#
#     # Wait until the QueryAnnotator has called start_processing on the AS.
#     rk_logger.info(f"Begin waiting for new_query")
#     while self.new_query is not None:
#         rate.sleep()
#
#     feedback_queue: queue.Queue = py_trees.blackboard.Blackboard().get(BBIdentifier.QUERY_FEEDBACK)
#
#     rk_logger.info(f"Start watching")
#     while True:
#         # Check if preempt is ready to be completed
#         preempted_acknowledged = py_trees.blackboard.Blackboard().get(BBIdentifier.QUERY_PREEMPT_ACK)
#         if preempted_acknowledged is True:
#             self.reset_bookkeeping_vars()
#             self._as.set_preempted()
#             return
#
#         # Check for errors
#         if robokudo.utils.error_handling.get_blackboard_exception() is not None:
#             rk_logger.info(f"QI catched error")
#             # Atleast one of the annotators failed - Abort and reset exception conditions
#             self.reset_bookkeeping_vars()
#
#             exception_text = str(robokudo.utils.error_handling.get_blackboard_exception())
#             robokudo.utils.error_handling.clear_blackboard_exception()
#             self._as.set_aborted(text=exception_text)
#             return
#
#         # Check for preempt request and put request info on the Tree
#         # Atleast one of the Annotators/Behaviors have to acknowledge in order
#         # finish the preemption
#         if self._as.is_preempt_requested() and not self.preempt_message_printed:
#             rk_logger.info(f"Query Interface received preempt request")
#             self.preempt_message_printed = True
#             py_trees.blackboard.Blackboard().set(BBIdentifier.QUERY_PREEMPT_REQUESTED, True)
#             continue
#
#         # Check for feedback messages to be sent
#         try:
#             self._feedback.status = feedback_queue.get_nowait()
#             self._as.publish_feedback(self._feedback)
#         except queue.Empty:
#             pass
#
#         answer = py_trees.blackboard.Blackboard().get(BBIdentifier.QUERY_ANSWER)
#         if answer is not None:
#             break
#
#         rate.sleep()
#     rk_logger.info(f"Send: {answer}")
#     self.send_result(answer)
# """


class GenerateAllQueryResult(robokudo.annotators.core.BaseAnnotator):
    """
    This class reads in the annotations done by the previous Annotators and generates
    Object Designators from them.

    These will be placed into the Blackboard so that a running Query Action Server can
    pick the information up and send it as a query reply.
    """

    def __init__(self, name="GenerateQueryResult"):
        super().__init__(name=name)

    def update(self):

        query = self.get_cas().get(CASViews.QUERY)
        queried_location = query.obj.location
        if queried_location == "":
            filter_by_region = False
        else:
            filter_by_region = True

        annotations = self.get_cas().annotations
        object_hypotheses_count = 0
        query_result = []
        result = Query.Result()

        for annotation in annotations:
            if not isinstance(annotation, robokudo.types.scene.ObjectHypothesis):
                continue

            object_hypotheses_count += 1
            region = False
            if filter_by_region:
                for attr in annotation.annotations:
                    if isinstance(attr, robokudo.types.annotation.LocationAnnotation):
                        if attr.name == queried_location:
                            region = True

            object_designator = robokudo_msgs.msg.ObjectDesignator()
            belief_ids = [
                a for a in annotation.annotations if isinstance(a, BeliefEntry)
            ]
            if belief_ids:
                object_designator.uid = str(belief_ids[0].id)
            if filter_by_region == region:
                for oh_annotation in annotation.annotations:
                    # TODO Rewrite with a generic conversion module for each of the types

                    if isinstance(
                        oh_annotation, robokudo.types.annotation.Classification
                    ):
                        if oh_annotation.source == "CLIPAnnotator":
                            object_designator.attribute.append(oh_annotation.classname)
                        else:
                            object_designator.type = oh_annotation.classname
                    if isinstance(
                        oh_annotation, robokudo.types.annotation.SemanticColor
                    ):
                        object_designator.color.append(oh_annotation.color)

                    if isinstance(oh_annotation, robokudo.types.cv.BoundingBox3D):
                        object_designator.size = f"x: {oh_annotation.x_length}, y: {oh_annotation.y_length}, z: {oh_annotation.z_length}"
                        size = ShapeSize()
                        vector = Vector3()
                        vector.x = oh_annotation.x_length
                        vector.y = oh_annotation.y_length
                        vector.z = oh_annotation.z_length
                        size.dimensions = vector
                        size.radius = 0
                        object_designator.shape_size.append(size)

                    if isinstance(
                        oh_annotation, robokudo.types.annotation.PoseAnnotation
                    ):
                        ps = geometry_msgs.msg.PoseStamped()

                        pose_map = transform_pose_from_camera_to_world(
                            self.get_cas(), oh_annotation
                        )
                        # TODO create PoseStamped and add it to the list
                        ps.pose.position.x = pose_map.translation[0]
                        ps.pose.position.y = pose_map.translation[1]
                        ps.pose.position.z = pose_map.translation[2]

                        ps.pose.orientation.x = pose_map.rotation[0]
                        ps.pose.orientation.y = pose_map.rotation[1]
                        ps.pose.orientation.z = pose_map.rotation[2]
                        ps.pose.orientation.w = pose_map.rotation[3]

                        # We assume that the pose annotation is in CAMERA coordinates
                        ps.header = camera_header(self.get_cas())
                        ps.header.frame_id = "map"
                        object_designator.pose.append(ps)

                        object_designator.pose_source.append(oh_annotation.source)

                    if isinstance(
                        oh_annotation, robokudo.types.annotation.PositionAnnotation
                    ):
                        pos = PoseAnnotation()
                        pos.source = oh_annotation.source
                        pos.translation.insert(0, oh_annotation.translation[0])
                        pos.translation.insert(1, oh_annotation.translation[1])
                        pos.translation.insert(2, oh_annotation.translation[2])

                        pos.rotation.insert(0, 0)
                        pos.rotation.insert(1, 0)
                        pos.rotation.insert(2, 0)
                        pos.rotation.insert(3, 1)

                        pose_map = transform_pose_from_camera_to_world(
                            self.get_cas(), pos
                        )

                        # Must be a PoseStamped due to type specification in ros message
                        ps = geometry_msgs.msg.PoseStamped()
                        ps.pose.position.x = pose_map.translation[0]
                        ps.pose.position.y = pose_map.translation[1]
                        ps.pose.position.z = pose_map.translation[2]

                        ps.pose.orientation.x = pose_map.rotation[0]
                        ps.pose.orientation.y = pose_map.rotation[1]
                        ps.pose.orientation.z = pose_map.rotation[2]
                        ps.pose.orientation.w = pose_map.rotation[3]

                        # We assume that the pose annotation is in CAMERA coordinates
                        ps.header = camera_header(self.get_cas())
                        ps.header.frame_id = "map"
                        object_designator.pose.append(ps)

                        object_designator.pose_source.append(oh_annotation.source)

                query_result.append(object_designator)

        result.res = query_result
        QueryHandler.send_answer(result)
        self.feedback_message = (
            f"Send result for {object_hypotheses_count} object hypotheses"
        )
        return py_trees.common.Status.SUCCESS


class GenerateCharacteristicResult(robokudo.annotators.core.BaseAnnotator):
    """
    This class reads in the annotations done by the previous Annotators and generates
    Object Designators from them.

    These will be placed into the Blackboard so that a running Query Action Server can
    pick the information up and send it as a query reply.
    """

    def __init__(self, name="GenerateCharacteristicResult"):
        super().__init__(name=name)

    def update(self):

        hypotheses = self.get_cas().filter_annotations_by_type(
            robokudo.types.scene.HumanHypothesis
        )
        query_result = []
        result = Query.Result()
        object_designator = robokudo_msgs.msg.ObjectDesignator()

        if hypotheses:
            belief_ids = [
                a for a in hypotheses[0].annotations if isinstance(a, BeliefEntry)
            ]
            if belief_ids:
                object_designator.uid = str(belief_ids[0].id)
            for annotation in hypotheses[0].annotations:
                if not isinstance(annotation, robokudo.types.annotation.Classification):
                    continue

                if annotation.source == "CLIPAnnotator":
                    object_designator.attribute.append(annotation.classname)
                else:
                    object_designator.type = annotation.classname
        query_result.append(object_designator)
        result.res = query_result
        QueryHandler.send_answer(result)
        self.feedback_message = f"Send result for {len(hypotheses)} human hypotheses"
        return py_trees.common.Status.SUCCESS
