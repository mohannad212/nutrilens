"""Environment-based settings. See .env.example for the documented variable list."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- Backend ----
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    secret_key: str = "change-me-32+chars"
    access_token_expire_minutes: int = 60
    jwt_algorithm: str = "HS256"

    # ---- Database ----
    database_url: str = "postgresql+psycopg://nutrilens:password@localhost:5432/nutrilens"

    # ---- USDA FoodData Central ----
    usda_api_key: str = ""
    usda_base_url: str = "https://api.nal.usda.gov/fdc/v1"
    usda_rate_limit_per_hour: int = 3600

    # ---- ML ----
    model_dir: Path = Path("./models")
    classifier_weights: str = "food101_efficientnet_b4.pth"
    confidence_threshold: float = 0.60
    device: str = "cpu"

    # ---- Security policy ----
    max_failed_logins: int = 5
    max_upload_mb: int = 10
    min_image_width: int = 640
    min_image_height: int = 480

    # ---- Recommendations ----
    min_meals_for_recommendation: int = 7
    recommendation_window_days: int = 7


@lru_cache
def get_settings() -> Settings:
    return Settings()
