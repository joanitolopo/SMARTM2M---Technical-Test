# client_send.py
import base64
import json
import asyncio
from websockets import connect
from PIL import Image
import io

WS_URL = "ws://localhost:8000/ws/infer"

async def send_image(path):
    with open(path, "rb") as f:
        b = f.read()
    b64 = "data:image/png;base64," + base64.b64encode(b).decode('ascii')
    async with connect(WS_URL) as websocket:
        await websocket.send(json.dumps({"image": b64}))
        resp = await websocket.recv()
        print("Server response:", resp)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python client_send.py path/to/image.png")
    else:
        asyncio.run(send_image(sys.argv[1]))
