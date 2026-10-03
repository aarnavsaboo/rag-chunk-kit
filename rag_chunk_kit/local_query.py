from __future__ import annotations

from urllib.request import Request,urlopen
import json


class OllamaQueryExpander:
    def __init__(self,model:str,endpoint:str="http://127.0.0.1:11434"):
        self.model=model
        self.endpoint=endpoint.rstrip("/")

    def expand(self,query:str,count:int=3)->list[str]:
        if count<1:
            raise ValueError("count must be positive")
        prompt=f"""Generate {count} alternative search queries for retrieval.
Preserve the information need but vary wording and terminology.
Return a JSON object with a single key "queries" containing an array of strings.

ORIGINAL QUERY:
{query}
"""
        body=json.dumps({
            "model":self.model,
            "prompt":prompt,
            "stream":False,
            "format":"json",
            "options":{"temperature":0.6},
        }).encode()
        request=Request(
            self.endpoint+"/api/generate",
            data=body,
            headers={"Content-Type":"application/json"},
            method="POST",
        )
        with urlopen(request,timeout=600) as response:
            outer=json.load(response)
        data=json.loads(outer["response"])
        rows=data.get("queries",[]) if isinstance(data,dict) else []
        seen={query.casefold().strip()}
        output=[]
        for value in rows:
            item=str(value).strip()
            key=item.casefold()
            if item and key not in seen:
                seen.add(key)
                output.append(item)
        return output[:count]
