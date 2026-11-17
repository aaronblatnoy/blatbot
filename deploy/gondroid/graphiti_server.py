import asyncio, logging, os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.llm_client import LLMConfig, OpenAIClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("memory")
state = {"g": None, "q": None, "done": 0, "failed": 0, "last_error": ""}

async def worker():
    while True:
        item = await state["q"].get()
        for attempt in range(3):
            try:
                await state["g"].add_episode(**item)
                state["done"] += 1
                break
            except Exception as exc:
                state["last_error"] = repr(exc)[:500]
                log.warning("episode %s attempt %d failed: %r", item.get("name"), attempt + 1, exc)
                if attempt == 2:
                    state["failed"] += 1
                else:
                    await asyncio.sleep(5 * (attempt + 1))
        state["q"].task_done()

@asynccontextmanager
async def lifespan(app):
    model = os.environ.get("MODEL_NAME") or "gpt-5.4-mini"
    llm = OpenAIClient(LLMConfig(api_key=os.environ["OPENAI_API_KEY"], model=model, small_model=model), reasoning=None, verbosity=None)
    state["g"] = Graphiti(os.environ["GRAPH_URI"], os.environ["GRAPH_USER"], os.environ["GRAPH_PASSWORD"], llm_client=llm)
    await state["g"].build_indices_and_constraints()
    state["q"] = asyncio.Queue()
    task = asyncio.create_task(worker())
    yield
    task.cancel()
    await state["g"].close()

app = FastAPI(lifespan=lifespan)

class Message(BaseModel):
    content: str
    role_type: str = "user"
    role: Optional[str] = None
    name: str = ""
    timestamp: Optional[datetime] = None
    source_description: str = ""

class AddMessages(BaseModel):
    group_id: str
    messages: List[Message]

class SearchQuery(BaseModel):
    query: str
    group_ids: Optional[List[str]] = None
    max_facts: int = 10

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "healthy", "queued": state["q"].qsize(), "done": state["done"],
            "failed": state["failed"], "last_error": state["last_error"]}

@app.post("/messages", status_code=202)
async def messages(body: AddMessages):
    for m in body.messages:
        who = m.role or m.role_type
        await state["q"].put(dict(
            name=m.name or "message", episode_body=who + ": " + m.content,
            source=EpisodeType.message, source_description=m.source_description,
            reference_time=m.timestamp or datetime.now(timezone.utc), group_id=body.group_id))
    return {"success": True, "queued": state["q"].qsize()}

@app.post("/search")
async def search(body: SearchQuery):
    edges = await state["g"].search(body.query, group_ids=body.group_ids, num_results=body.max_facts)
    names = {}
    try:
        ids = list({u for e in edges for u in (e.episodes or [])})
        if ids:
            for n in await EpisodicNode.get_by_uuids(state["g"].driver, ids):
                names[n.uuid] = n.name
    except Exception as exc:
        log.warning("episode names failed: %r", exc)
    def iso(v):
        return v.isoformat() if v else None
    return {"facts": [dict(uuid=e.uuid, name=e.name, fact=e.fact, group_id=e.group_id,
                           valid_at=iso(e.valid_at), invalid_at=iso(e.invalid_at),
                           created_at=iso(e.created_at), expired_at=iso(e.expired_at),
                           episodes=[names[u] for u in (e.episodes or []) if u in names]) for e in edges]}
