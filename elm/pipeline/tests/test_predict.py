import pytest
from elm.pipeline.predict_many import predict_many

@pytest.mark.skip # TODO finish this test
def test_predict_many():
    models = [MiniBatchKMeans(n_clusters=n_clusters) for _ in range(5)]

    predict_many(config, step, client,
                 models=None,
                 serialize=None,
                 to_cube=True)
