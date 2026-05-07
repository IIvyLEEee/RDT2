import sys
import time

sys.path.append(".")

import click
import numpy as np

from deploy.remote_policy.websocket_policy import WebsocketClientPolicy


@click.command()
@click.option("--policy_server_host", required=True, help="Remote RDT2-FM policy server host.")
@click.option("--policy_server_port", default=8000, show_default=True, type=int)
@click.option("--instruction", default="Pick up the pink snack bag with the right hand.", show_default=True)
@click.option("--image_size", default=384, show_default=True, type=int)
@click.option("--num_warmup", default=0, show_default=True, type=int)
def main(policy_server_host, policy_server_port, instruction, image_size, num_warmup):
    client = WebsocketClientPolicy(host=policy_server_host, port=policy_server_port)
    print(f"Remote policy metadata: {client.get_server_metadata()}")

    observation = {
        "images": {
            "left_stereo": np.zeros((image_size, image_size, 3), dtype=np.uint8),
            "right_stereo": np.zeros((image_size, image_size, 3), dtype=np.uint8),
        },
        "state": np.zeros(20, dtype=np.float32),
        "instruction": instruction,
    }

    for _ in range(num_warmup):
        client.infer(observation)

    start = time.time()
    result = client.infer(observation)
    elapsed_ms = (time.time() - start) * 1000

    actions = result["actions"]
    print(f"Client round-trip: {elapsed_ms:.1f} ms")
    print(f"Actions shape: {actions.shape}, dtype: {actions.dtype}")
    print(f"Server timing: {result.get('server_timing', {})}")
    print(f"Policy timing: {result.get('policy_timing', {})}")


if __name__ == "__main__":
    main()
