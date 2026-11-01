import json

import py_trees
from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.utils.query import QueryHandler
from robokudo_msgs.action import Query
from robokudo.annotators.gpsr.belief_state import BeliefStateAnnotator


class BeliefStateLookupAnnotator(BaseAnnotator):
    """
    Returns BeliefState entries as JSON in text_result.

    With 3 cases:
    uid set: single entry by ID
    description[0]: all entries matching classname/yolo_label/semantic_label
    nothing set: all known entries persons and objects
    """

    def __init__(self, name="BeliefStateLookupAnnotator", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def update(self):
        query = self.get_cas().get(CASViews.QUERY)
        uid = query.obj.uid.strip()
        result = Query.Result()
        state = BeliefStateAnnotator._state

        if query.obj.description:
            category = query.obj.description[0].strip()
        else:
            category = ""

        if uid:
            try:
                entry = state.find_by_id(int(uid))
                if entry:
                    entries = [entry]
                else:
                    entries = []
            except ValueError:
                entries = []
        elif category in ("human", "person"):
            entries = list(state.persons())
        elif category:
            entries = (
                state.find_by_classname(category)
                or state.find_by_yolo_label(category)
                or state.find_by_semantic_label(category)
            )
        else:
            entries = list(state)

        if not entries:
            result.text_result = json.dumps({"error": "no entries found"})
            QueryHandler.send_answer(result)
            self.feedback_message = "No entries found."
            return py_trees.common.Status.SUCCESS

        payload = []
        for e in entries:
            payload.append(e.to_dict())

        if len(payload) == 1:
            to_send = payload[0]
        else:
            to_send = payload
        result.text_result = json.dumps(to_send)
        QueryHandler.send_answer(result)
        self.feedback_message = f"Returned {len(entries)} entry/entries."
        return py_trees.common.Status.SUCCESS
