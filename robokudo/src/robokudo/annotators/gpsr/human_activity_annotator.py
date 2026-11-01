"""
This module implements a RoboKudo Annotator: The HumanActivityAnnotator.
This annotator uses a pretrained CLIP model to classify human activities with a zeroshot approach.
It distinguishes between mutually exclusive postures (e.g.
sitting, standing) and gestures (raising, waving, pointing).

Gesture direction (left/right) is not determined here. That is the job of the
downstream HumanPointingAnnotator.

Author: Ali Talha Sükür
Created at: 2026-04-14
"""

from timeit import default_timer

import numpy as np
import py_trees
import torch
from PIL import Image, ImageDraw, ImageOps
from transformers import CLIPModel, CLIPProcessor

import robokudo.annotators.core
import robokudo.types.human
import robokudo.types.scene
from robokudo.cas import CASViews
from robokudo.types.annotation import Encoding
from robokudo.utils.error_handling import catch_and_raise_to_blackboard


class HumanActivityAnnotator(robokudo.annotators.core.ThreadedAnnotator):
    """Uses CLIP to classify human posture and gestures for each detected person."""

    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                # text descriptions for posture classification using the full body crop
                # we use softmax and take the highest score, so exactly one posture per person
                self.posture_classes = (
                    "a person standing upright with arms at their sides",
                    "a person sitting upright with torso vertical",
                    "a person lying down with their body mostly horizontal",
                    "a person crouching or squatting low",
                )
                self.posture_labels = ("Standing", "Sitting", "Lying", "Crouching")

                # text descriptions for gesture classification using the upper body crop
                # we use softmax and take the highest score above the threshold
                self.gesture_classes = (
                    "a person waving hello with an open palm",
                    "a person with their arm extended out pointing at something",
                    "a person with one arm raised up in the air",
                    "a person standing with both arms down at their sides",
                )
                self.gesture_labels = ("Waving", "Pointing", "Raising", "No gesture")

                self.gesture_threshold = 0.65  # if no gesture score is above this, we use "No gesture"
                self.save_top_k = 1  # we only save the best gesture per person, important sum of the softmax is 1

                self.clip_model = "openai/clip-vit-large-patch14"
                self.clip_processor = "openai/clip-vit-large-patch14"

        parameters = Parameters()

    def __init__(self, name="HumanActivityAnnotator", descriptor=Descriptor()) -> None:
        self.parameters = descriptor.parameters
        super().__init__(name, descriptor)
        self.model = CLIPModel.from_pretrained(self.parameters.clip_model)
        self.processor = CLIPProcessor.from_pretrained(self.parameters.clip_processor)

    @catch_and_raise_to_blackboard
    def compute(self):
        start = default_timer()
        self.analyze_human_hypotheses()
        self.feedback_message = f"Processing took {(default_timer() - start):.4f}s"
        return py_trees.common.Status.SUCCESS

    def cutout_human_rois(self, human_hypotheses, upper_body=False):
        """Extracts person crops from the color image.
        If upper_body_only=True, only the upper 60% used.
        """
        img = Image.fromarray(self.get_cas().get(CASViews.COLOR_IMAGE))

        #Image height is not required
        img_width, _ = img.size
        crops = []
        for hh in human_hypotheses:
            #getting the coordinates from the roi bounding boxes
            x = hh.roi.roi.pos.x
            y = hh.roi.roi.pos.y
            w = hh.roi.roi.width
            h = hh.roi.roi.height

            if upper_body:
                padding    = int(w * 0.2)
                left   = max(0, x - padding)
                right  = min(img_width, x + w + padding)
                bottom = y + int(h * 0.6)
            else:
                left = x
                right = x + w
                bottom =  y + h

            crop = img.crop((left, y, right, bottom))

            if upper_body:
                crop = ImageOps.autocontrast(crop)

            crops.append(crop)

        return crops

    def analyze_human_hypotheses(self):
        """Runs CLIP on all detected people and writes annotations for each one."""

        human_hypotheses = self.get_cas().filter_annotations_by_type(robokudo.types.scene.HumanHypothesis)
        if not human_hypotheses:
            return

        crops_full  = self.cutout_human_rois(human_hypotheses)
        crops_upper = self.cutout_human_rois(human_hypotheses, upper_body=True)

        # prepare the inputs for clip
        inputs_posture = self.processor(
            text=list(self.parameters.posture_classes), images=crops_full,
            return_tensors="pt", padding=True
        )
        inputs_gesture = self.processor(
            text=list(self.parameters.gesture_classes), images=crops_upper,
            return_tensors="pt", padding=True
        )

        with torch.no_grad():
            probs_posture  = self.model(**inputs_posture).logits_per_image.softmax(dim=1).detach().numpy()
            probs_gesture  = self.model(**inputs_gesture).logits_per_image.softmax(dim=1).detach().numpy()
            raw_feats      = self.model.get_image_features(pixel_values=inputs_posture["pixel_values"]).pooler_output
            image_features = (raw_feats / raw_feats.norm(dim=-1, keepdim=True)).detach().cpu().numpy()

        for hh, posture_dist, gesture_dist, img_feat in zip(
            human_hypotheses, probs_posture, probs_gesture, image_features
        ):
            self.annotate_person(hh, posture_dist, gesture_dist, img_feat)

        self.visualize_humans(human_hypotheses)

    def annotate_person(self, hh, posture_dist, gesture_dist, img_feat):
        """Writes posture, gesture and CLIP embedding annotations for one person."""

        # save the CLIP embedding to the BeliefState
        clip_encoding = Encoding()
        clip_encoding.encoding = img_feat
        clip_encoding.source = type(self).__name__
        hh.annotations.append(clip_encoding)

        # posture: we always pick the label with the highest softmax score
        posture_label = self.parameters.posture_labels[posture_dist.argmax()]
        posture_ann = robokudo.types.human.ActivityAnnotation()
        posture_ann.type = posture_label
        posture_ann.source = type(self).__name__
        hh.annotations.append(posture_ann)

        # to visualize the posture winner and the values of each one
        for i, (label, score) in enumerate(zip(self.parameters.posture_labels, posture_dist)):
            self.rk_logger.info(
                f"  Posture HH {hh.id}: {label} = {score:.2f}"
                + ("  <-- WINNER" if i == posture_dist.argmax() else "")
            )

        # to visualize the gesture winner and the values of each one
        # if nothing is above the threshold we write "No gesture"
        self.rk_logger.info(f"  Gesture scores HH {hh.id}:")
        for label, score in zip(self.parameters.gesture_labels, gesture_dist):
            self.rk_logger.info(
                f"    {label} = {score:.2f}"
                + ("  <-- THRESHOLD" if score >= self.parameters.gesture_threshold else "")
            )

        # Select top k gesture labels above threshold, highest scores first
        # Add each valid gesture as an ActivityAnnotation for this person
        gesture_count = 0
        for idx in gesture_dist.argsort()[::-1]:
            if gesture_count >= self.parameters.save_top_k:
                break
            label = self.parameters.gesture_labels[idx]
            if label == "No gesture":
                continue
            if gesture_dist[idx] >= self.parameters.gesture_threshold:
                gesture_ann = robokudo.types.human.ActivityAnnotation()
                gesture_ann.type = label
                gesture_ann.source = type(self).__name__
                hh.annotations.append(gesture_ann)
                self.rk_logger.info(f"Gesture for HH {hh.id}: {label} ({gesture_dist[idx]:.2f})")
                gesture_count += 1

        # If no gesture passed the threshold annotate this person with "No gesture"
        if gesture_count == 0:
            no_gesture = robokudo.types.human.ActivityAnnotation()
            no_gesture.type = "No gesture"
            no_gesture.source = type(self).__name__
            hh.annotations.append(no_gesture)
            self.rk_logger.info(f"Gesture for HH {hh.id}: No gesture")

    def visualize_humans(self, human_hypotheses):
        """Draws bounding boxes with posture and gesture labels into the output image."""
        img = Image.fromarray(self.get_cas().get_copy(CASViews.COLOR_IMAGE))
        draw = ImageDraw.Draw(img)

        for hh in human_hypotheses:
            x, y = hh.roi.roi.pos.x, hh.roi.roi.pos.y
            w, h = hh.roi.roi.width, hh.roi.roi.height
            draw.rectangle((x, y, x + w, y + h), outline=(0, 255, 0), width=2)

            activities = [a for a in hh.annotations if isinstance(a, robokudo.types.human.ActivityAnnotation)]
            postures = [a for a in activities if a.type in self.parameters.posture_labels]
            gestures = [a for a in activities if a.type in self.parameters.gesture_labels]

            if postures:
                draw.text((x, y - 20), postures[0].type, fill=(0, 255, 0))
            if gestures:
                draw.text((x, y - 40), ", ".join(g.type for g in gestures), fill=(255, 255, 0))

        self.get_annotator_output_struct().set_image(np.array(img))
