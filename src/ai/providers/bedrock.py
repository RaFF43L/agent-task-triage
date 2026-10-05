"""Provider Bedrock (default): Claude via Converse API + Titan embeddings."""

from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

from config import envs
from config.aws import DEFAULT_CLIENT_CONFIG, aws_client_manager

from .base import ModelProvider, ProviderRegistry


@ProviderRegistry.register
class BedrockProvider(ModelProvider):
    name = "bedrock"

    def model_large(self, **kwargs):
        return ChatBedrockConverse(
            model=envs.model,
            region_name=envs.aws_region,
            config=DEFAULT_CLIENT_CONFIG,
            **kwargs,
        )

    def model_fast(self, **kwargs):
        return ChatBedrockConverse(
            model=envs.model_fast,
            region_name=envs.aws_region,
            config=DEFAULT_CLIENT_CONFIG,
            **kwargs,
        )

    def model_judge(self, **kwargs):
        return ChatBedrockConverse(
            model=envs.model_judge or envs.model,
            region_name=envs.aws_region,
            config=DEFAULT_CLIENT_CONFIG,
            **kwargs,
        )

    def embeddings(self, **kwargs):
        client = aws_client_manager.get_client("bedrock-runtime")
        return BedrockEmbeddings(
            client=client,
            model_id=envs.embed_model,
            model_kwargs={
                "dimensions": envs.embed_dimensions,
                "normalize": True,
            },
            **kwargs,
        )
