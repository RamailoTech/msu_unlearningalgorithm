from abc import ABC, abstractmethod


class BaseConfig(ABC):

    @abstractmethod
    def __init__(self):
        pass

    def validate_config(self):
        pass

    def to_dict(self):
        result = {}
        for attr_name, attr_value in self.__dict__.items():
            if hasattr(attr_value, "to_dict") and callable(attr_value.to_dict):
                result[attr_name] = attr_value.to_dict()
            elif isinstance(attr_value, list):
                result[attr_name] = [
                    item.to_dict() if hasattr(item, "to_dict") else item
                    for item in attr_value
                ]
            elif isinstance(attr_value, dict):
                dict_val = {}
                for k, v in attr_value.items():
                    dict_val[k] = v.to_dict() if hasattr(v, "to_dict") else v
                result[attr_name] = dict_val
            else:
                result[attr_name] = attr_value
        return result
