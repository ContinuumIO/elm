import pytest
from elm.pipeline.predict import predict

@pytest.mark.skip # TODO finish this test
def test_predict():
    models = [MiniBatchKMeans(n_clusters=n_clusters) for _ in range(5)]

    predict(config, step, client,
                 models=None,
                 serialize=None,
                 to_cube=True)
