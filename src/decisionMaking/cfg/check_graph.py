import argparse
import networkx as nx
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------

def merge_nodes_pairwise(g, x_key="x", y_key="y"):
    nodes = list(g.nodes())          # snapshot; we mutate g while iterating
    seen_deleted = set()             # avoid double-processing

    for i, u in enumerate(nodes):
        if u in seen_deleted:
            continue
        xu, yu = float(g.nodes[u][x_key]), float(g.nodes[u][y_key])

        for v in nodes[i + 1:]:
            if v in seen_deleted:
                continue
            xv, yv = float(g.nodes[v][x_key]), float(g.nodes[v][y_key])

            if xu == xv and yu == yv:
                # ---- Duplicate found: merge v → u ------------------------
                if g.is_multigraph():
                    # Re-wire successors
                    for succ, keydict in g[v].items():
                        for k, edata in keydict.items():
                            g.add_edge(u, succ, **edata)
                    # Re-wire predecessors (Di graphs only)
                    if g.is_directed():
                        for pred, keydict in g.pred[v].items():
                            for k, edata in keydict.items():
                                g.add_edge(pred, u, **edata)
                else:
                    for succ, edata in g[v].items():
                        g.add_edge(u, succ, **edata)
                    if g.is_directed():
                        for pred, edata in g.pred[v].items():
                            g.add_edge(pred, u, **edata)

                g.remove_node(v)
                seen_deleted.add(v)


def draw_graph(path):
    G = nx.read_graphml(path)
    pos = {}
    for n, data in G.nodes(data=True):
        try:
            x = float(data.get("d1", data.get("x")))
            y = float(data.get("d2", data.get("y")))
        except (TypeError, ValueError):
            x, y = 0.0, 0.0
        pos[n] = (x, y)

    plt.figure(figsize=(30, 30))
    label_pos = {n: (x - 0.02, y + 0.1) for n, (x, y) in pos.items()}
    # Draw nodes and edges
    nx.draw(G, pos, node_size=10, with_labels=False, arrows=False, edge_color='gray')

    # Draw labels for each node at its position
    nx.draw_networkx_labels(G, label_pos, font_size=10)

    plt.gca().set_aspect("equal")
    plt.savefig("config/graph.png", dpi=300, bbox_inches='tight')
    # ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--infile", default="config/raw.graphml")
    ap.add_argument("-o", "--outfile", default="config/output.graphml",
                    help="Output .graphml file (default: %(default)s)")
    ap.add_argument("--x_key", default="x")
    ap.add_argument("--y_key", default="y")
    ap.add_argument("--img", default=True)
    args = ap.parse_args()

    g = nx.read_graphml(args.infile)
    merge_nodes_pairwise(g, x_key=args.x_key, y_key=args.y_key)
    nx.write_graphml(g, args.outfile)
    if args.img:
        draw_graph(args.outfile)
    print(f"Saved deduplicated graph to {args.outfile}")

if __name__ == "__main__":
    main()
