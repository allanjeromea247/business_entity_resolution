from src.pipeline import run_prediction_pipeline
from src.utils import setup_logger
from src.config import LOGS_DIR

if __name__ == "__main__":
    setup_logger('src', LOGS_DIR / 'predict.log')
    run_prediction_pipeline()
