from codecarbon import EmissionsTracker
from pathlib import Path
import datetime


class CarbonTracker:
    def __init__(self, output_dir="outputs/logs", experiment_name=None):
        """
        Wrapper around CodeCarbon tracker

        Args:
            output_dir (str): where emission logs are stored
            experiment_name (str): optional name for experiment
        """

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        if experiment_name:
            self.output_path = Path(output_dir) / f"{experiment_name}_{timestamp}"
        else:
            self.output_path = Path(output_dir) / timestamp

        self.output_path.mkdir(parents=True, exist_ok=True)

        self.tracker = EmissionsTracker(
            output_dir=str(self.output_path),
            log_level="warning"   # keeps console clean
        )

    # Start tracking
    def start(self):
        self.tracker.start()

    # Stop tracking
    def stop(self):
        emissions = self.tracker.stop()
        return emissions

    # Convenience method
    def track(self, func, *args, **kwargs):
        """
        Runs a function while tracking emissions
        """
        self.start()
        result = func(*args, **kwargs)
        emissions = self.stop()

        return result, emissions
    

"""
How to use it 

from src.utils.tracking import CarbonTracker

# 1. Initialize Tracker
tracker = CarbonTracker(experiment_name="model_training_run")

# 2. Start Tracking
tracker.start()

# -- Your Training Code Here --
# model.fit(X_train, y_train)

# 3. Stop Tracking and get emissions (kg CO2)
emissions = tracker.stop()
print(f"Emissions: {emissions} kg CO2")

# OR use the convenience method
def train_logic(data):
    # training steps
    return "model_success"

result, emissions = tracker.track(train_logic, data="my_dataset")
"""