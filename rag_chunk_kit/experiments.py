from __future__ import annotations

from dataclasses import asdict,dataclass
from hashlib import sha1
from itertools import product
from pathlib import Path
import json

from .pipeline import evaluate,ingest
from .retrieval import SearchIndex,SentenceTransformerEmbedder


@dataclass(frozen=True)
class Experiment:
    id:str
    name:str
    corpus:str
    cases:str
    chunk_chars:int
    overlap:int
    top_k:int
    embedding_model:str|None

    def to_dict(self):
        return asdict(self)


def expand(config:dict)->list[Experiment]:
    rows=[]
    for chunk_chars,overlap,top_k,embedding_model in product(
        config.get("chunk_chars",[600]),
        config.get("overlap",[60]),
        config.get("top_k",[5]),
        config.get("embedding_models",[None]),
    ):
        payload={
            "name":str(config["name"]),
            "corpus":str(config["corpus"]),
            "cases":str(config["cases"]),
            "chunk_chars":int(chunk_chars),
            "overlap":int(overlap),
            "top_k":int(top_k),
            "embedding_model":embedding_model,
        }
        ident=sha1(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]
        rows.append(Experiment(id=ident,**payload))
    return rows


def run(experiment:Experiment)->dict:
    chunks=ingest(experiment.corpus,experiment.chunk_chars,experiment.overlap)
    embedder=SentenceTransformerEmbedder(experiment.embedding_model) if experiment.embedding_model else None
    index=SearchIndex.build(chunks,embedder)
    cases=json.loads(Path(experiment.cases).read_text(encoding="utf-8"))
    result=evaluate(index,cases,experiment.top_k,embedder)
    return {
        "experiment_id":experiment.id,
        "config":experiment.to_dict(),
        "chunks":len(chunks),
        "metrics":{key:result[key] for key in ("recall","mrr","ndcg")},
        "cases":result["cases"],
    }
