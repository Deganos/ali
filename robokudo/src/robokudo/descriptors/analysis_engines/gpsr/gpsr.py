import robokudo.analysis_engine
import robokudo.descriptors.camera_configs.config_hsr_ros2
import robokudo.io.camera_interface
import robokudo.io.storage_reader_interface
import robokudo_robocup_gpsr.descriptors.analysis_engines.gpsr_implementation
import robokudo_robocup_gpsr.descriptors.camera_configs.config_kinect_xbox360
from robokudo.annotators.collection_reader import CollectionReaderAnnotator


class AnalysisEngine(robokudo_robocup_gpsr.descriptors.analysis_engines.gpsr_implementation.AnalysisEngine):
    """Analysis Engine for the RoboCup@Home 'GPSR' Challenge. Configured to load data
    from the HSR camera. For the actual implementation, checkout gpsr_implementation.py
    in the same module."""

    def configure_collection_reader(self):
        # --- HSR (robot) ---
        # hsr_camera_config = robokudo.descriptors.camera_configs.config_hsr_ros2.CameraConfig()
        # camera_config = hsr_camera_config

        # --- Xbox 360 Kinect (development machine) ---
        # See the docstring of config_kinect_xbox360 for the driver and TF commands.
        kinect_camera_config = \
            robokudo_robocup_gpsr.descriptors.camera_configs.config_kinect_xbox360.CameraConfig()
        camera_config = kinect_camera_config

        config = CollectionReaderAnnotator.Descriptor(
            camera_config=camera_config,
            camera_interface=robokudo.io.camera_interface.KinectCameraInterface(camera_config))

        # Perceive Subtree
        self.cr = CollectionReaderAnnotator(descriptor=config)
