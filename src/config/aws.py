import boto3
from botocore.config import Config
from config.envs import envs

DEFAULT_CLIENT_CONFIG = Config(
    read_timeout=300,
    connect_timeout=10,
    retries={"max_attempts": 2, "mode": "adaptive"},
)


class AWSClientManager:
    def __init__(self, **kwargs):
        self.session = boto3.Session(region_name=envs.aws_region, **kwargs)

    def get_client(self, service_name: str, **kwargs) -> "boto3.client":
        kwargs.setdefault("config", DEFAULT_CLIENT_CONFIG)
        return self.session.client(service_name, region_name=envs.aws_region, **kwargs)


aws_client_manager = AWSClientManager()
