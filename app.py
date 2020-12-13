from fastapi import FastAPI
from fastai.vision import *
from io import BytesIO
import aiohttp


app = FastAPI()
learner = load_learner("./clases", "export.pkl")


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

@app.get("/classify-url")
async def classify_url(url: str):
    print(url)
    bytes = await get_bytes(url)
    img = open_image(BytesIO(bytes))
    _, _, losses = learner.predict(img)

    return {
        "predictions": sorted(
            zip(learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    }


