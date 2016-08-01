import pytest
from elm.pipeline.predict import predict_step

@pytest.mark.skip # TODO finish this test
def test_predict_step():
    models = [MiniBatchKMeans(n_clusters=n_clusters) for _ in range(5)]

    predict_step(config, step, executor,
                 models=None,
                 serialize=None,
                 to_cube=True)
