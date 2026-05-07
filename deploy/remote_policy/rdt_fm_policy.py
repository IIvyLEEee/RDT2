import time
from typing import Dict

import numpy as np
import torch

from models.rdt_inferencer import RDTInferencer


class RDTFMPolicy:
    """Remote-serving wrapper around `RDTInferencer.step`."""

    def __init__(
        self,
        config: Dict,
        pretrained_path: str,
        normalizer_path: str,
        pretrained_vision_language_model_name_or_path: str,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        self._model = RDTInferencer(
            config=config,
            pretrained_path=pretrained_path,
            normalizer_path=normalizer_path,
            pretrained_vision_language_model_name_or_path=pretrained_vision_language_model_name_or_path,
            device=device,
            dtype=dtype,
        )
        self._config = config
        self.metadata = {
            "policy_type": "rdt2_fm",
            "camera_names": list(config["dataset"]["camera_names"]),
            "state_dim": int(config["common"]["state_dim"]),
            "action_dim": int(config["common"]["action_dim"]),
            "action_chunk_size": int(config["common"]["action_chunk_size"]),
            "fps": int(config["common"].get("fps", 30)),
        }

    def infer(self, obs: Dict) -> Dict:
        instruction = obs.get("instruction", obs.get("prompt"))
        if instruction is None:
            raise ValueError("Observation must include `instruction` or `prompt`.")

        images = obs.get("images")
        if not isinstance(images, dict):
            raise ValueError("Observation must include `images` as a dict.")

        state = obs.get("state")
        if state is None:
            state = np.zeros(self.metadata["state_dim"], dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)

        observations = {
            "images": {name: np.asarray(images[name], dtype=np.uint8) for name in self.metadata["camera_names"]},
            "state": state,
        }

        start = time.monotonic()
        action = self._model.step(observations=observations, instruction=instruction)
        policy_ms = (time.monotonic() - start) * 1000
        return {
            "actions": action.detach().cpu().numpy().astype(np.float32),
            "policy_timing": {"step_ms": policy_ms},
        }
