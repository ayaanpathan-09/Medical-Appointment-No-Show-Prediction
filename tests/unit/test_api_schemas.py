import pytest
from pydantic import ValidationError

from src.api.schemas import PredictRequest


def test_valid_predict_request():
    req = PredictRequest(
        appointment_id="A1",
        patient_id="P1",
        age=34,
        gender="F",
        scheduled_date="2024-01-01",
        appointment_date="2024-01-10",
        sms_received=True,
        chronic_condition=False,
    )
    assert req.age == 34


def test_invalid_age_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(
            appointment_id="A1", patient_id="P1", age=-5, gender="F",
            scheduled_date="2024-01-01", appointment_date="2024-01-10", sms_received=True,
        )


def test_invalid_gender_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(
            appointment_id="A1", patient_id="P1", age=30, gender="X",
            scheduled_date="2024-01-01", appointment_date="2024-01-10", sms_received=True,
        )
