"""RTM graph visualization using NetworkX + matplotlib.

Renders the traceability graph with a hierarchical layout reflecting the
flow: satellite requirements -> ADCS requirements -> design elements ->
evidence -> attestations.

Nodes are color-coded by type, edges labeled by relationship.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from rdflib import Graph

from ontology.prefixes import PROV, RTM, SYSML
from traceability.queries import query_to_dicts

# Color scheme
COLORS = {
    "sat_requirement": "#7FB3DE",  # light blue
    "requirement": "#4A90D9",      # blue
    "design_element": "#7BC67E",   # green
    "proof": "#F5A623",            # orange
    "simulation": "#E8C840",       # gold
    "attestation": "#D94A4A",      # red
}

# Layers for hierarchical layout (left to right)
_LAYER_X = {
    "attestation": 0,
    "sat_requirement": 1,
    "requirement": 2,
    "design_element": 3,
    "evidence": 4,
}


def _extract_graph_data(rdf_graph: Graph) -> tuple[nx.DiGraph, dict, dict]:
    """Extract nodes and edges from RDF into a NetworkX DiGraph.

    Returns (G, node_colors, node_types) where node_colors maps
    node_id -> hex color and node_types maps node_id -> type string.
    """
    G = nx.DiGraph()
    node_colors = {}
    node_types = {}

    # --- Satellite requirements ---
    q = """
    SELECT ?name WHERE {
        ?req a sysml:RequirementDefinition ;
             sysml:declaredName ?name .
        FILTER(STRSTARTS(?name, "SAT-"))
    }
    """
    for row in query_to_dicts(rdf_graph, q):
        n = row["name"]
        # Shorten for display
        short = n.replace("SAT-REQ-", "SAT:\n")
        G.add_node(n, label=short)
        node_colors[n] = COLORS["sat_requirement"]
        node_types[n] = "sat_requirement"

    # --- ADCS requirements ---
    q = """
    SELECT ?name WHERE {
        ?req a sysml:RequirementDefinition ;
             sysml:declaredName ?name .
        FILTER(STRSTARTS(?name, "REQ-"))
    }
    """
    for row in query_to_dicts(rdf_graph, q):
        n = row["name"]
        G.add_node(n, label=n)
        node_colors[n] = COLORS["requirement"]
        node_types[n] = "requirement"

    # --- Derivation edges ---
    q = """
    SELECT ?child ?parent WHERE {
        ?c sysml:declaredName ?child ;
           rtm:derivedFrom ?p .
        ?p sysml:declaredName ?parent .
    }
    """
    for row in query_to_dicts(rdf_graph, q):
        G.add_edge(row["parent"], row["child"], rel="derivedFrom")

    # --- Design elements ---
    q = """
    SELECT DISTINCT ?elementName WHERE {
        ?rel a sysml:SatisfyRequirementUsage ;
             sysml:satisfyingElement ?el .
        ?el sysml:declaredName ?elementName .
    }
    """
    for row in query_to_dicts(rdf_graph, q):
        n = row["elementName"]
        # Shorten reaction wheel names
        short = n.replace("ReactionWheel_", "RW-")
        G.add_node(n, label=short)
        node_colors[n] = COLORS["design_element"]
        node_types[n] = "design_element"

    # --- Satisfy edges ---
    q = """
    SELECT ?reqName ?elementName WHERE {
        ?req sysml:declaredName ?reqName ;
             sysml:ownedRelationship ?rel .
        ?rel a sysml:SatisfyRequirementUsage ;
             sysml:satisfyingElement ?el .
        ?el sysml:declaredName ?elementName .
        FILTER(STRSTARTS(?reqName, "REQ-"))
    }
    """
    for row in query_to_dicts(rdf_graph, q):
        G.add_edge(row["reqName"], row["elementName"], rel="satisfiedBy")

    # --- Evidence nodes (linked via rtm:addresses) ---
    q = """
    SELECT ?ev ?type ?hash ?reqName WHERE {
        ?ev a ?type ;
            rtm:contentHash ?hash ;
            rtm:addresses ?req .
        ?req sysml:declaredName ?reqName .
        FILTER(?type IN (rtm:ProofArtifact, rtm:SimulationResult))
    }
    """
    for row in query_to_dicts(rdf_graph, q):
        ev_type = row["type"].split("#")[-1]
        short_hash = row["hash"][:8]
        node_id = f"ev_{short_hash}"
        is_proof = ev_type == "ProofArtifact"
        label = f"Proof\n{short_hash}" if is_proof else f"Sim\n{short_hash}"
        color = COLORS["proof"] if is_proof else COLORS["simulation"]

        G.add_node(node_id, label=label)
        node_colors[node_id] = color
        node_types[node_id] = "evidence"
        G.add_edge(row["reqName"], node_id, rel="addresses")

    # --- Attestation nodes ---
    q = """
    SELECT ?reqName ?engineer ?timestamp WHERE {
        ?att a rtm:Attestation ;
             rtm:attests ?req ;
             prov:wasAssociatedWith ?agent ;
             prov:generatedAtTime ?timestamp .
        ?agent rdfs:label ?engineer .
        ?req sysml:declaredName ?reqName .
    }
    """
    for row in query_to_dicts(rdf_graph, q):
        att_id = f"att_{row['reqName']}"
        ts = row["timestamp"][:10] if row["timestamp"] else ""
        # Shorten engineer name for display
        eng = row["engineer"].split("(")[0].strip()
        label = f"Attested\n{eng}\n{ts}"
        G.add_node(att_id, label=label)
        node_colors[att_id] = COLORS["attestation"]
        node_types[att_id] = "attestation"
        G.add_edge(att_id, row["reqName"], rel="attests")

    return G, node_colors, node_types


def _hierarchical_layout(
    G: nx.DiGraph, node_types: dict,
) -> dict:
    """Compute a hierarchical layout anchored on ADCS requirements.

    The four ADCS requirements form the backbone. Everything else is
    positioned relative to the requirement it connects to, eliminating
    edge crossings between layers.

    Columns (left to right):
      attestations | sat requirements | ADCS requirements | design elements | evidence
    """
    x_cols = {
        "attestation": 0,
        "sat_requirement": 2.5,
        "requirement": 5.0,
        "design_element": 7.5,
        "evidence": 10.0,
    }
    req_y_spacing = 3.0  # vertical space between requirements

    pos = {}

    # --- Step 1: Place ADCS requirements as the vertical backbone ---
    reqs = sorted([n for n, t in node_types.items() if t == "requirement"])
    for i, req in enumerate(reqs):
        pos[req] = (x_cols["requirement"], -i * req_y_spacing)

    # --- Step 2: Place attestations aligned with their target requirement ---
    for node, ntype in node_types.items():
        if ntype != "attestation":
            continue
        # Find which requirement this attestation targets
        target_req = None
        for _, neighbor, data in G.edges(data=True):
            if data.get("rel") == "attests" and _ == node:
                target_req = neighbor
                break
        if target_req and target_req in pos:
            pos[node] = (x_cols["attestation"], pos[target_req][1])
        else:
            pos[node] = (x_cols["attestation"], 0)

    # --- Step 3: Place satellite requirements aligned with their derived ADCS req ---
    for node, ntype in node_types.items():
        if ntype != "sat_requirement":
            continue
        # Find which ADCS requirement derives from this
        target_req = None
        for _, neighbor, data in G.edges(data=True):
            if data.get("rel") == "derivedFrom" and _ == node:
                target_req = neighbor
                break
        if target_req and target_req in pos:
            pos[node] = (x_cols["sat_requirement"], pos[target_req][1])
        else:
            pos[node] = (x_cols["sat_requirement"], 0)

    # --- Step 4: Place design elements near the requirements they satisfy ---
    # Group design elements by which requirements connect to them
    design_nodes = sorted([n for n, t in node_types.items() if t == "design_element"])
    de_req_map: dict[str, list[str]] = {}  # design element -> list of connected reqs
    for u, v, data in G.edges(data=True):
        if data.get("rel") == "satisfiedBy" and v in design_nodes:
            de_req_map.setdefault(v, []).append(u)

    # Position each design element at the average Y of its connected requirements,
    # then spread them vertically to avoid overlap
    de_y_targets: list[tuple[float, str]] = []
    for de in design_nodes:
        connected_reqs = de_req_map.get(de, [])
        if connected_reqs:
            avg_y = sum(pos[r][1] for r in connected_reqs if r in pos) / len(connected_reqs)
        else:
            avg_y = 0
        de_y_targets.append((avg_y, de))

    # Sort by target Y position and spread with minimum spacing
    de_y_targets.sort(key=lambda t: t[0])
    min_de_spacing = 1.0
    for i, (target_y, de) in enumerate(de_y_targets):
        if i == 0:
            pos[de] = (x_cols["design_element"], target_y)
        else:
            prev_y = pos[de_y_targets[i - 1][1]][1]
            y = min(target_y, prev_y - min_de_spacing)
            pos[de] = (x_cols["design_element"], y)

    # --- Step 5: Place evidence nodes near their connected requirement ---
    ev_nodes = sorted([n for n, t in node_types.items() if t == "evidence"])
    ev_req_map: dict[str, str] = {}
    for u, v, data in G.edges(data=True):
        if data.get("rel") == "addresses" and v in ev_nodes:
            ev_req_map[v] = u

    # Group evidence by requirement, then stack vertically near that requirement
    ev_by_req: dict[str, list[str]] = {}
    for ev, req in ev_req_map.items():
        ev_by_req.setdefault(req, []).append(ev)

    for req, evs in ev_by_req.items():
        if req not in pos:
            continue
        req_y = pos[req][1]
        n = len(evs)
        ev_spacing = 1.0
        for i, ev in enumerate(sorted(evs)):
            offset = (i - (n - 1) / 2) * ev_spacing
            pos[ev] = (x_cols["evidence"], req_y + offset)

    return pos


def build_rtm_figure(
    rdf_graph: Graph,
    figsize: tuple[float, float] = (16, 10),
    title: str = "Requirements Traceability Matrix",
) -> plt.Figure:
    """Build a matplotlib figure of the RTM graph.

    Returns the Figure object for display in marimo or saving to file.
    """
    G, node_colors, node_types = _extract_graph_data(rdf_graph)
    pos = _hierarchical_layout(G, node_types)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges by type with different styles
    edge_styles = {
        "derivedFrom": {"style": "solid", "color": "#888888", "width": 1.5},
        "satisfiedBy": {"style": "solid", "color": "#555555", "width": 1.5},
        "addresses": {"style": "dashed", "color": "#999999", "width": 1.0},
        "attests": {"style": "solid", "color": COLORS["attestation"], "width": 2.0},
    }

    for rel_type, style in edge_styles.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("rel") == rel_type]
        if edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=edges, ax=ax,
                style=style["style"],
                edge_color=style["color"],
                width=style["width"],
                arrows=True,
                arrowsize=15,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.1",
                min_source_margin=20,
                min_target_margin=20,
            )

    # Draw nodes
    for node in G.nodes():
        x, y = pos[node]
        color = node_colors.get(node, "#cccccc")
        ntype = node_types.get(node, "")
        label = G.nodes[node].get("label", node)

        # Node shape/size by type
        if ntype == "attestation":
            bbox = dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor="#333", linewidth=1.5)
            fontcolor = "white"
            fontsize = 7
        elif ntype in ("sat_requirement", "requirement"):
            bbox = dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor="#333", linewidth=1.5)
            fontcolor = "white"
            fontsize = 9
        elif ntype == "design_element":
            bbox = dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor="#333", linewidth=1.5)
            fontcolor = "#1a1a1a"
            fontsize = 8
        else:  # evidence
            bbox = dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="#333", linewidth=1.0)
            fontcolor = "#1a1a1a"
            fontsize = 7

        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color=fontcolor, fontweight="bold",
                bbox=bbox, zorder=5)

    # Edge labels
    edge_labels = {(u, v): d["rel"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=6, font_color="#666666",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
    )

    # Legend — positioned below the graph
    legend_items = [
        mpatches.Patch(facecolor=COLORS["sat_requirement"], edgecolor="#333", label="Satellite Requirement"),
        mpatches.Patch(facecolor=COLORS["requirement"], edgecolor="#333", label="ADCS Requirement"),
        mpatches.Patch(facecolor=COLORS["design_element"], edgecolor="#333", label="Design Element"),
        mpatches.Patch(facecolor=COLORS["proof"], edgecolor="#333", label="Proof Artifact"),
        mpatches.Patch(facecolor=COLORS["simulation"], edgecolor="#333", label="Simulation Result"),
        mpatches.Patch(facecolor=COLORS["attestation"], edgecolor="#333", label="Attestation"),
    ]
    ax.legend(
        handles=legend_items, loc="upper center",
        bbox_to_anchor=(0.5, -0.02), ncol=6, fontsize=8, framealpha=0.9,
    )

    # Column headers — positioned above the graph content
    x_cols = {"attestation": 0, "sat_requirement": 2.5, "requirement": 5.0,
              "design_element": 7.5, "evidence": 10.0}
    col_labels = {
        "attestation": "Attestation",
        "sat_requirement": "Satellite\nRequirements",
        "requirement": "ADCS\nRequirements",
        "design_element": "Design\nElements",
        "evidence": "Evidence",
    }
    y_top = max(y for _, y in pos.values()) + 1.8
    for col_name, x in x_cols.items():
        ax.text(x, y_top, col_labels[col_name],
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#444")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=40)
    ax.axis("off")
    ax.margins(x=0.08, y=0.12)
    fig.tight_layout()

    return fig


# Keep build_dot for backwards compatibility with tests
def build_dot(graph: Graph) -> str:
    """Build a Graphviz DOT string from the RTM graph (legacy)."""
    G, node_colors, node_types = _extract_graph_data(graph)

    lines = [
        'digraph RTM {',
        '  rankdir=LR;',
        '  node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8];',
    ]
    for node in G.nodes():
        color = node_colors.get(node, "#cccccc")
        label = G.nodes[node].get("label", node).replace("\n", "\\n")
        fontcolor = "white" if node_types.get(node) in ("attestation", "requirement", "sat_requirement") else "black"
        lines.append(f'  "{node}" [fillcolor="{color}", fontcolor={fontcolor}, label="{label}"];')
    for u, v, d in G.edges(data=True):
        lines.append(f'  "{u}" -> "{v}" [label="{d.get("rel", "")}"];')
    lines.append("}")
    return "\n".join(lines)


def render_rtm(
    graph: Graph,
    output_path: str | Path = "output/rtm_graph",
    fmt: str = "png",
) -> Path:
    """Render the RTM graph to a file using matplotlib."""
    fig = build_rtm_figure(graph)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = output_path.with_suffix(f".{fmt}")
    fig.savefig(str(out_file), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file
