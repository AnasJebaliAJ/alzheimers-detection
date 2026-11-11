
import pytest
from predict import predict_patient


@pytest.fixture
def valid_patient():
    return {
        "M/F": "F",
        "Age": 75,
        "EDUC": 12,
        "SES": 2.0,
        "MMSE": 27.0,
        "eTIV": 1500,
        "nWBV": 0.72,
        "ASF": 1.17
    }


def test_prediction(valid_patient):
    result = predict_patient(valid_patient)

    assert result["prediction"] in [
        "Demented", "Nondemented"
    ]

    probabilities = result["probabilities"]

    assert sum(probabilities.values()) == pytest.approx(1.0)

    for probability in probabilities.values():
        assert 0 <= probability <= 1


def test_missing_features():
    with pytest.raises(ValueError, match="Missing features"):
        predict_patient({"Age": 75})


def test_invalid_gender(valid_patient):
    valid_patient["M/F"] = "X"

    with pytest.raises(ValueError, match="M/F must be"):
        predict_patient(valid_patient)


@pytest.mark.parametrize("age", [-5, 150])
def test_invalid_age(valid_patient, age):
    valid_patient["Age"] = age

    with pytest.raises(ValueError, match="Age must be"):
        predict_patient(valid_patient)
