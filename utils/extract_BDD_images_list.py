"""
This file has the information about the BDD 100K images. And a method to extract a specific list required image names.
"""
import json
from tqdm import tqdm
from pathlib import Path

train_labels_path = "C:/Others/BDD/bdd100k_labels_release/bdd100k\labels/bdd100k_labels_images_train.json"
val_labels_path = "C:/Others/BDD/bdd100k_labels_release/bdd100k\labels/bdd100k_labels_images_val.json"

# list of weather types in BDD
types_of_weather_list = ['clear', 'rainy', 'undefined', 'snowy', 'overcast', 'partly cloudy', 'foggy']
types_of_weather_count_train = {'clear': 37344, 'rainy': 5070, 'undefined': 8119, 'snowy': 5549, 'overcast': 8770,
                                'partly cloudy': 4881, 'foggy': 130}
types_of_weather_count_val = {'clear': 5346, 'rainy': 738, 'undefined': 1157, 'snowy': 769, 'overcast': 1239,
                              'partly cloudy': 738, 'foggy': 13}

# list of type of time of day
types_of_timeofday_list = ['daytime', 'dawn/dusk', 'night', 'undefined']
types_of_timeofday_count_train = {'daytime': 36728, 'dawn/dusk': 5027, 'night': 27971, 'undefined': 137}
types_of_timeofday_count_val = {'daytime': 5258, 'dawn/dusk': 778, 'night': 3929, 'undefined': 35}


def get_list_of_images(weather_type: str, timeofday: str, json_labels_file_path: Path) -> list[str]:
    """
    :param weather_type: type of weather in the image
    :param timeofday: time of day of the image
    :param json_labels_file_path: path to labels json file.
    :return: filtered list of image names with respect to weather_type and timeofday from json_labels_file.
    """
    imgs_list = []
    with open(json_labels_file_path, "r") as labels_file_path:
        labels = json.load(labels_file_path)
        for label in tqdm(labels):
            if label["attributes"]["weather"] == weather_type and label["attributes"]["timeofday"] == timeofday:
                imgs_list.append(label["name"])
    return imgs_list


print(len(get_list_of_images(weather_type="clear", timeofday="night", json_labels_file_path=Path(val_labels_path))))
