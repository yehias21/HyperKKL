import logging
from comet_ml import Experiment
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
import os


class Logger:
    def __init__(self, local_path, env_path=None):
        """
        :param type: local, global or both
        :param env_path: .env path file
        """
        # Load Comet credentials from .env file
        load_dotenv(dotenv_path=env_path)
        comet_api_key = os.getenv("COMET_API_KEY")
        comet_project_name = os.getenv("COMET_PROJECT_NAME")
        comet_workspace = os.getenv("COMET_WORKSPACE")
        # Initialize Comet.ml experiment
        self.comet_experiment = None
        if comet_api_key and comet_project_name and comet_workspace:
            self.comet_experiment = Experiment(
                api_key=comet_api_key,
                project_name=comet_project_name,
                workspace=comet_workspace
            )

        # Initialize TensorBoard writer
        self.tensorboard_writer = SummaryWriter(log_dir=local_path)

    def log_parameters(self, parameters):
        # Log parameters to both Comet.ml and TensorBoard
        if self.comet_experiment:
            self.comet_experiment.log_parameters(parameters)
        self.tensorboard_writer.add_hparams(parameters, {})

    def log_metric(self, name, value, step):
        # Log metrics to both Comet.ml and TensorBoard
        if self.comet_experiment:
            self.comet_experiment.log_metric(name, value, step=step)
        self.tensorboard_writer.add_scalar(name, value, global_step=step)

    def end(self):
        # Close Comet.ml experiment and TensorBoard writer
        if self.comet_experiment:
            self.comet_experiment.end()
        self.tensorboard_writer.close()
