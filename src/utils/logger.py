from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import datetime


class TBLogger:
    def __init__(self, log_dir="outputs/logs", experiment_name=None):
        """
        TensorBoard Logger

        Args:
            log_dir (str): base directory for logs
            experiment_name (str): optional name for experiment
        """

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        if experiment_name:
            self.log_path = Path(log_dir) / f"{experiment_name}_{timestamp}"
        else:
            self.log_path = Path(log_dir) / timestamp

        self.writer = SummaryWriter(self.log_path)

    # Log scalar metrics
    def log_metrics(self, model_name, metrics: dict, step=0):
        """
        Logs metrics like accuracy, log_loss

        Example:
            logger.log_metrics("xgboost", {"accuracy": 0.91, "log_loss": 0.3})
        """
        for key, value in metrics.items():
            if value is not None:
                self.writer.add_scalar(f"{model_name}/{key}", value, step)

    # Log parameters (optional)
    def log_params(self, model_name, params: dict):
        """
        Logs model hyperparameters as text
        """
        param_str = "\n".join([f"{k}: {v}" for k, v in params.items()])
        self.writer.add_text(f"{model_name}/params", param_str)

    # Log message (general)
    def log_text(self, tag, text):
        self.writer.add_text(tag, text)

    # Close logger
    def close(self):
        self.writer.close()


"""
How to use it 


from src.utils.logger import TBLogger

# 1. Initialize Logger
logger = TBLogger(log_dir="outputs/logs", experiment_name="cancer_classification")

# 2. Log Metrics
metrics = {"accuracy": 0.85, "log_loss": 0.42}
logger.log_metrics("random_forest", metrics)

# 3. Log Hyperparameters
params = {"n_estimators": 200, "max_depth": 10}
logger.log_params("random_forest", params)

# 4. Close when finished
logger.close()
"""