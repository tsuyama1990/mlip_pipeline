from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Global settings for the MLIP-AutoPipe application.
    """

    qe_command: str = ""


settings = Settings()
