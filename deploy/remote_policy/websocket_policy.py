import asyncio
import http
import logging
import time
import traceback
from typing import Dict, Optional, Tuple

import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames
import websockets.sync.client

from deploy.remote_policy import msgpack_numpy


logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serve a policy object with an `infer(dict) -> dict` method."""

    def __init__(self, policy, host: str = "0.0.0.0", port: int = 8000, metadata: Optional[Dict] = None):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_start = time.monotonic()
                result = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_start

                result.setdefault("server_timing", {})
                result["server_timing"]["infer_ms"] = infer_time * 1000
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(result))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


class WebsocketClientPolicy:
    """Synchronous client used by the robot-side control loop."""

    def __init__(self, host: str, port: int = 8000, api_key: Optional[str] = None, reconnect_sleep_s: float = 5.0):
        if host.startswith("ws://") or host.startswith("wss://"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"

        self._api_key = api_key
        self._reconnect_sleep_s = reconnect_sleep_s
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def infer(self, obs: Dict) -> Dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        pass

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logger.info("Waiting for policy server at %s...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Still waiting for policy server...")
                time.sleep(self._reconnect_sleep_s)


def _health_check(connection: websocket_server.ServerConnection, request: websocket_server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
