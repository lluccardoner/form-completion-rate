import argparse

from utils import DATASET_DIR


def get_args():
    parser = argparse.ArgumentParser(description="Train a model to predict form completion rates")

    parser.add_argument("datasetPath", default=str(DATASET_DIR / "completion_rate.csv"),
                        nargs='?',help="Path where the dataset csv is stored")
    parser.add_argument("-d", "--debug", help="Enable debug logs", action="store_true")

    return parser.parse_args()
