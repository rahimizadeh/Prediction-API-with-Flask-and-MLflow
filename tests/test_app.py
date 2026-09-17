import numpy as np

from Application_Server import create_app


class DummyModel:
    def predict(self, frame):
        assert list(frame.columns) == ["experience", "age", "interview_score"]
        return np.array([12345.67])


def test_health():
    client = create_app(model=DummyModel()).test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_success():
    client = create_app(model=DummyModel()).test_client()
    response = client.post(
        "/predict",
        json={"experience": 4, "age": 30, "interview_score": 8},
    )
    assert response.status_code == 200
    assert response.get_json() == {"prediction": 12345.67}


def test_predict_missing_field():
    client = create_app(model=DummyModel()).test_client()
    response = client.post("/predict", json={"experience": 4, "age": 30})
    assert response.status_code == 400
    assert "interview_score" in response.get_json()["error"]
