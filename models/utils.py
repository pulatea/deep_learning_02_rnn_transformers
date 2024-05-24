from models import model_1
from models import model_2_test
from models import model_2
from models import model_3
from models import model_4
from models import model_5
from models import model_6


def get_model_class(model_name):
    model_classes_dict = {'model_1': model_1.Model,
                          'model_2': model_2.Model,
                          'model_3': model_3.Model,
                          'model_4': model_4.Model,
                          'model_5': model_5.Model,
                          'model_6': model_6.Model}

    return model_classes_dict[model_name]
