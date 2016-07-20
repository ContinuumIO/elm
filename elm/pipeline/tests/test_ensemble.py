from sklearn.datasets.samples_generator import make_blobs
#from elm.pipeline.tests.util import train_with_synthetic_data
from elm.model_selection import PARTIAL_FIT_MODEL_DICT
from elm.model_selection import get_args_kwargs_defaults

ensemble_kwargs = {}
executor = None

def make_sample(batch_size, n_features=2, centers=3):

    X, y = make_blobs(n_samples=batch_size,
                      centers=centers,
                      n_features=n_features,
                      random_state=0)

    df = pd.DataFrame(X)
    df.columns = ['band_{}'.format(idx + 1) for idx in range(df.shape[1])]

    return df, y

action_data = [(make_sample, (), {})]
fit_kwargs =
for model_name, model_init_class in PARTIAL_FIT_MODEL_DICT.item():
    _, model_init_kwargs = get_args_kwargs_defaults(model_init_class)
    for fit_func in ('fit', 'partial_fit'):
        fit_args = (action_data,)
        ensemble(executor,
                 model_init_class,
                 model_init_kwargs,
                 fit_func,
                 fit_args,
                 fit_kwargs,
                 model_selection_func,
                 model_selection_kwargs,
                 **ensemble_kwargs)