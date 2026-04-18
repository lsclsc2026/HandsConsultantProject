import csv
from pathlib import Path

import networkx as nx

from app.core.config import ROOT_DIR


class GraphRetriever:
    def __init__(self) -> None:
        self.graph = nx.Graph()
        self.path = ROOT_DIR / "knowledge" / "graph_edges.csv"
        self._load_graph()

    def _load_graph(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                src = row.get("source", "").strip()
                dst = row.get("target", "").strip()
                relation = row.get("relation", "关联")
                weight = float(row.get("weight", "0.5"))
                if not src or not dst:
                    continue
                self.graph.add_edge(src, dst, relation=relation, weight=weight)

    def retrieve(self, query_text: str, profile_text: str, top_k: int = 6) -> list[str]:
        if self.graph.number_of_nodes() == 0:
            return []

        seed_text = f"{query_text} {profile_text}"
        seeds = [n for n in self.graph.nodes if n in seed_text]
        results: list[tuple[float, str]] = []

        for seed in seeds:
            for nbr in self.graph.neighbors(seed):
                edge = self.graph.get_edge_data(seed, nbr) or {}
                score = float(edge.get("weight", 0.5))
                relation = edge.get("relation", "关联")
                text = f"图谱规则：{seed} 与 {nbr} 为{relation}关系。"
                results.append((score, text))

        results.sort(key=lambda item: item[0], reverse=True)
        dedup: list[str] = []
        for _, text in results:
            if text not in dedup:
                dedup.append(text)
            if len(dedup) >= top_k:
                break
        return dedup
