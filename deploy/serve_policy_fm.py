import logging
import socket
import sys

sys.path.append(".")

import click
import torch
import yaml

from deploy.remote_policy.rdt_fm_policy import RDTFMPolicy
from deploy.remote_policy.websocket_policy import WebsocketPolicyServer


DTYPE_BY_NAME = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@click.command()
@click.option("--input", "-i", required=True, help="Path to RDT-FM checkpoint.")
@click.option("--pretrained_vision_language_model_name_or_path", "-vlm", required=True, help="Path to Qwen2.5-VL checkpoint.")
@click.option("--normalizer_path", "-np", required=True, help="Path to normalizer checkpoint.")
@click.option("--model_config", "-mc", required=True, help="Path to model_config yaml file.")
@click.option("--host", default="0.0.0.0", show_default=True, help="Server bind address.")
@click.option("--port", default=8000, show_default=True, type=int, help="Server port.")
@click.option("--device", default="cuda", show_default=True, help="Torch device for policy inference.")
@click.option("--dtype", type=click.Choice(sorted(DTYPE_BY_NAME)), default="bf16", show_default=True)
def main(
    input,
    pretrained_vision_language_model_name_or_path,
    normalizer_path,
    model_config,
    host,
    port,
    device,
    dtype,
):
    logging.basicConfig(level=logging.INFO, force=True)

    with open(model_config, "r") as f:
        config = yaml.safe_load(f)

    policy = RDTFMPolicy(
        config=config,
        pretrained_path=input,
        normalizer_path=normalizer_path,
        pretrained_vision_language_model_name_or_path=pretrained_vision_language_model_name_or_path,
        device=torch.device(device),
        dtype=DTYPE_BY_NAME[dtype],
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Starting RDT2-FM policy server on host=%s ip=%s bind=%s port=%s", hostname, local_ip, host, port)
    logging.info("Policy metadata: %s", policy.metadata)

    server = WebsocketPolicyServer(policy=policy, host=host, port=port, metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
