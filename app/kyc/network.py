"""Synthetic-identity / fraud-ring detection on a shared-identifier graph.

Nodes are applications and identifiers (phone, IBAN, address, device,
employer blind indexes). Two applications that share an identifier are
connected through that identifier node. A connected component containing at
least ``min_size`` *distinct applicants* linked through strong identifiers
(phone/IBAN/device/address) is flagged as a potential fraud ring; the
component is returned as visualisation data (nodes + edges).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
from pydantic import BaseModel

STRONG_IDENTIFIERS = ("phone", "iban", "device", "address")


@dataclass(frozen=True)
class IdentityRecord:
    application_id: str
    applicant_key: str  # blind index of the TCKN
    identifiers: dict[str, str | None]


class GraphNode(BaseModel):
    id: str
    kind: str
    label: str


class GraphEdge(BaseModel):
    source: str
    target: str
    kind: str


class RingResult(BaseModel):
    ring_size: int  # distinct applicants in the component
    application_count: int
    shared_identifiers: list[str]
    flagged: bool
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def build_graph(records: Iterable[IdentityRecord]) -> nx.Graph:
    graph = nx.Graph()
    for rec in records:
        app_node = f"app:{rec.application_id}"
        graph.add_node(app_node, kind="application", applicant=rec.applicant_key)
        for kind in STRONG_IDENTIFIERS:
            value = rec.identifiers.get(kind)
            if value:
                id_node = f"{kind}:{value[:12]}"
                graph.add_node(id_node, kind=kind)
                graph.add_edge(app_node, id_node, kind=kind)
    return graph


def ring_for(
    application_id: str, records: Iterable[IdentityRecord], min_size: int = 3
) -> RingResult:
    graph = build_graph(records)
    node = f"app:{application_id}"
    if node not in graph:
        return RingResult(
            ring_size=1,
            application_count=1,
            shared_identifiers=[],
            flagged=False,
            nodes=[],
            edges=[],
        )
    component = nx.node_connected_component(graph, node)
    sub = graph.subgraph(component)
    apps = [n for n, d in sub.nodes(data=True) if d.get("kind") == "application"]
    applicants = {sub.nodes[n]["applicant"] for n in apps}
    shared = sorted(
        {
            sub.nodes[n]["kind"]
            for n in sub.nodes
            if sub.nodes[n].get("kind") != "application" and sub.degree(n) > 1
        }
    )
    nodes = [
        GraphNode(
            id=n,
            kind=d.get("kind", "?"),
            label=n.split(":", 1)[1][:8] if d.get("kind") != "application" else n[4:],
        )
        for n, d in sub.nodes(data=True)
    ]
    edges = [
        GraphEdge(source=u, target=v, kind=d.get("kind", "")) for u, v, d in sub.edges(data=True)
    ]
    return RingResult(
        ring_size=len(applicants),
        application_count=len(apps),
        shared_identifiers=shared,
        flagged=len(applicants) >= min_size and bool(shared),
        nodes=nodes,
        edges=edges,
    )
