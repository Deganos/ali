from timeit import default_timer
from PIL import Image
import py_trees

import robokudo.annotators.core
import robokudo.utils.cv_helper
import robokudo.utils.error_handling
import robokudo.utils.file_loader
import robokudo.types.annotation
import robokudo.types.human
import robokudo.types.scene
from robokudo.cas import CASViews

import os
os.environ["QT_API"] = "pyqt5"

class StoreFaces(robokudo.annotators.core.ThreadedAnnotator):
    class Descriptor(robokudo.annotators.core.BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self):
                self.ros_pkg_path = None
                self.data_path = None
                self.noise = 0

        parameters = Parameters()

    def __init__(self, name="StoreFaces", descriptor=Descriptor()):
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

    @robokudo.utils.error_handling.catch_and_raise_to_blackboard
    def compute(self):
        if self.descriptor.parameters.ros_pkg_path is None or self.descriptor.parameters.ros_pkg_path is None:
            raise 'Path is not set'

        start_timer = default_timer()

        # Get all human hypothesis
        human_hypotheses = self.get_cas().filter_annotations_by_type(robokudo.types.scene.HumanHypothesis)

        for annotation in human_hypotheses:
            # Check if the human_hypothesis already has a classification
            if self._check_if_class_exists(annotation):
                continue

            # Searching for Face annotations
            for face_annotation in annotation.annotations:
                if isinstance(face_annotation, robokudo.types.human.FaceAnnotation):
                    # Get the roi of face, extracted point is top left
                    width = face_annotation.roi.roi.width
                    length = face_annotation.roi.roi.height
                    x = face_annotation.roi.roi.pos.x
                    y = face_annotation.roi.roi.pos.y

                    # Define area of points with face
                    color_image = self.get_cas().get(CASViews.COLOR_IMAGE)
                    noise = self.descriptor.parameters.noise
                    x_min = x - noise
                    x_max = x + width + noise
                    y_min = y - noise
                    y_max = y + length + noise

                    # Load Path to data
                    file_loader = robokudo.utils.file_loader.FileLoader()
                    full_path = file_loader.get_path_to_file_in_ros_package(
                        ros_pkg_name=self.descriptor.parameters.ros_pkg_path,
                        relative_path=self.descriptor.parameters.data_path,
                    )
                    file_path = full_path

                    # Store unknown image under a unique id
                    file_name = self._get_id(full_path) + '.png'
                    file = os.path.join(file_path, file_name)
                    cropped_image = color_image[y_min:y_max,x_min:x_max]
                    image = Image.fromarray(cropped_image)
                    image.save(file)

                    # Write Annotation with id of people
                    classification = robokudo.types.annotation.Classification()
                    classification.source = self.name
                    classification.classification_type = 'face'
                    classification.classname = file_name.split(".")[0]
                    annotation.annotations.append(classification)


        end_timer = default_timer()
        self.feedback_message = f'Processing took {(end_timer - start_timer):.4f}s'
        return py_trees.common.Status.SUCCESS

    def _check_if_class_exists(self, human_hypothesis):
        '''
        This Method returns true, if for a human hypothesis a file already exist and false if not
        '''
        if not isinstance(human_hypothesis, robokudo.types.scene.HumanHypothesis):
            raise("Check if class exists receives a wrong type")


        for classification in human_hypothesis.annotations:
            #print(type(classification))
            if isinstance(classification, robokudo.types.annotation.Classification):
                print(f'Detect {classification.classname}, it\'s a familiar face')
                return True
        print('Detect unfamiliar face, create a new file for his face!')
        return False

    def _get_id(self, path):
        """
        This method iterates over the files in data and return an unambiguous id.
        """
        names = []
        for filename in os.listdir(path):
            names.append(filename.split(".")[0])
        print(len(names))
        for i in range(1000):
            if not str(i) in names:
                return str(i)
        raise "Coudn't find Id to store faces"

