import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train a model to predict form completion rates")

    parser.add_argument("datasetPath", help="Path where the dataset csv is stored")
    parser.add_argument("-d", "--debug", help="Enable debug logs", action="store_true")

    return parser.parse_args()
