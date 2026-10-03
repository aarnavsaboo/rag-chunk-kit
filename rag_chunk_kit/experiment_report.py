def summarize(rows:list[dict])->list[dict]:
    output=[]
    for row in rows:
        cfg=row["config"]
        output.append({
            "experiment_id":row["experiment_id"],
            "chunk_chars":cfg["chunk_chars"],
            "overlap":cfg["overlap"],
            "top_k":cfg["top_k"],
            "embedding_model":cfg["embedding_model"],
            "chunks":row["chunks"],
            **row["metrics"],
        })
    return sorted(output,key=lambda x:(-x["ndcg"],-x["mrr"],x["chunk_chars"],x["overlap"]))
