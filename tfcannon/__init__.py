from tfcannon.model import TFCannon


def load_model(file):
    """
    Load a trained Cannon model

    :param regularizer: Regularization
    :type regularizer: float
    :return: TFCannon model

    :History: 2019-Aug-02 - Written - Henry Leung (University of Toronto)
    """
    import h5py
    import numpy as np

    _model = TFCannon()
    _model.trained_flag = True

    data = h5py.File(f'{file}', 'r')

    _model.coeffs = np.array(data['ceoffs'])
    _model.scatter = np.array(data['scatter'])
    _model.npixel = np.array(data['npixel'])
    _model.nlabels = np.array(data['nlabels'])
    _model.labels_median = np.array(data['labels_median'])
    _model.labels_std = np.array(data['labels_std'])
    _model.l1_regularization = np.array(data['l1_regularization'])
    try:
        _model.label_names = np.array(data['label_names'])
    except KeyError:
        pass

    return _model
