import json
from collections import deque
from typing import List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
from src.decisionMaking.planning.PathFinder import PathFinder
import itertools
from collections import defaultdict
import os

# Preplotting path
class Navigator:
    ANGLE_THRESHOLD = 30  # degrees considered "straight"
    TURN_BACK_THRESHOLD = 150  # degrees considered "turn back"

    def __init__(self, graphml_file: str, cfg_file: str) -> None:
        self.pp = PathFinder(graphml_file)
        with open(cfg_file, "r") as fp:
            cfg = json.load(fp)
        self.checkpoints = list(map(int, cfg["CHECKPOINTS"]))
        self.intersections = set(map(int, cfg["NODE_TYPE"].get("INTERSECTION", [])))
        self.first_checkpoint = cfg["FIRST_CHECKPOINT"]
        self.pre_forbidden_path = {tuple(p) for p in cfg["PRE_FORBIDDEN_PATH"]}
        self.always_forbidden_path = {tuple(p) for p in cfg["ALWAYS_FORBIDDEN_PATH"]}
        self.first_checkpoint_reached = False
        self.segment_paths: List[List[int]] = []
        self.graph_path = graphml_file
        self.direction_map = {}

    @property
    def forbidden_paths(self):
        if not self.first_checkpoint_reached:
            return self.always_forbidden_path.union(self.pre_forbidden_path)
        return self.always_forbidden_path

    def _turn_priority(self, prev: int, curr: int, nxt: int) -> int:
        angle = self.pp._signed_angle(
            self.pp._heading(prev, curr),
            self.pp._heading(curr, nxt),
        )
        if abs(angle) < self.ANGLE_THRESHOLD:
            return 0  # forward
        return 1 if angle > 0 else 2  # right before left

    def _direction(self, prev: int, curr: int, nxt: int) -> str:
        return {0: "forward", 1: "right", 2: "left"}[self._turn_priority(prev, curr, nxt)]

    def _bfs_next_checkpoint(self, start: int, prev: int, remaining: set) -> List[int]:
        queue = deque([[prev, start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            curr = path[-1]
            if curr in remaining:
                return path[1:]

            neighbours = self.pp.neigh.get(curr, [])
            if len(path) >= 2:
                prev_node = path[-2]


                # Filter out neighbors that require a U-turn
                neighbours = [
                    n for n in neighbours
                    if abs(
                        self.pp._signed_angle(
                            self.pp._heading(prev_node, curr),
                            self.pp._heading(curr, n)
                        )
                    ) < self.TURN_BACK_THRESHOLD
                ]
                neighbours = sorted(
                    neighbours,
                    key=lambda n: self._turn_priority(prev_node, curr, n)
                )

            for nbr in neighbours:
                if nbr not in visited:
                    if len(path) >= 2:
                        prev_node = path[-2]
                        curr_node = path[-1]
                        next_node = nbr
                        if (prev_node, curr_node, next_node) in self.forbidden_paths:
                            continue

                    visited.add(nbr)
                    queue.append(path + [nbr])


        raise RuntimeError("No remaining checkpoint is reachable from current node")


    def capture_all(self, start_node: int) -> List[Tuple[int, int]]:
        prev = self.pp._find_previous_node(start_node)
        current = start_node
        remaining = set(self.checkpoints)
        hops: List[Tuple[int, int]] = []
        self.full_path = []
        segment_lst = []
        cnt = 0

        # --- STEP 1: Must reach FIRST_CHECKPOINT first ---
        if current != self.first_checkpoint:
            try:
                path_to_first = self._bfs_next_checkpoint(current, prev, {self.first_checkpoint})
            except RuntimeError as e:
                print("Could not reach FIRST_CHECKPOINT:", e)
                return []

            # NOTE: discard all intermediate checkpoints passed before reaching first_checkpoint
            self.full_path = [path_to_first]
            self.first_checkpoint_reached = True
            hops.append((current, self.first_checkpoint))
            current = self.first_checkpoint
            remaining.discard(self.first_checkpoint)
            prev = path_to_first[-2] if len(path_to_first) > 1 else prev
        # else:
        #     # Already starting at first_checkpoint
        #     self.first_checkpoint_reached = True
        #     remaining.discard(self.first_checkpoint)

        # --- STEP 2: Normal checkpoint capture ---
        self.full_path[0].pop()
        while remaining:
            valid_targets = [
                cp for cp in remaining if not self._needs_uturn(current, cp, prev)
            ]
            if not valid_targets:
                print("No more forward-reachable checkpoints without U-turns.")
                break

            sub_path = self._bfs_next_checkpoint(current, prev, set(valid_targets))
            full_path = [prev] + sub_path

            for i in range(1, len(full_path) - 1):
                curr_node = full_path[i]
                if curr_node in self.intersections:
                    dir_name = self._direction(full_path[i - 1], curr_node, full_path[i + 1])
                    self.direction_map[curr_node] = dir_name
                    cnt += 1
                segment_lst.append(curr_node)

                if cnt == 4:
                    self.full_path.append(segment_lst)
                    segment_lst = []
                    cnt = 0

            next_cp = sub_path[-1]
            hops.append((current, next_cp))
            remaining.remove(next_cp)
            prev = full_path[-2] if len(full_path) > 1 else prev
            current = next_cp

        # Final segment
        if segment_lst or not self.full_path:
            segment_lst.append(current)
            self.full_path.append(segment_lst)

        return hops



    def _needs_uturn(self,
                     start_node: int,
                     checkpoint_node: int,
                     prev_node: int) -> bool:

        try:
            path = self.pp.find_optimal_path(start_node, checkpoint_node)
        except Exception:
            return True          # unreachable → skip

        if len(path) < 2:        # already on the checkpoint
            return False

        try:
            current_heading = self.pp._heading(prev_node, start_node)
        except Exception:
            return False         # cannot determine → assume OK

        first_heading = self.pp._heading(path[0], path[1])
        turn_angle = abs(self.pp._signed_angle(current_heading, first_heading))
        return turn_angle >= self.TURN_BACK_THRESHOLD
    
    def update_after_run(self, current_node: int, visited_checkpoints: List[int]) -> List[Tuple[int, int]]:
        remaining = [cp for cp in self.checkpoints if cp not in visited_checkpoints]

        # Update internal state
        self.checkpoints = remaining
        self.segment_paths = []
        self.full_path = []

        # Replan from current_node
        hops = self.capture_all(current_node)

        return hops
    
    def is_on_expected_path(self, prev_node: int, current_node: int) -> bool:
        """
        Returns True only if (prev_node → current_node) matches the planned next move.
        """
        if not self.full_path:
            return False

        # Flatten full path into one sequence
        path_nodes = list(itertools.chain.from_iterable(self.full_path))

        for i in range(len(path_nodes) - 1):
            if path_nodes[i] == prev_node:
                # We're expecting the next node to be:
                expected_next = path_nodes[i + 1]
                return current_node == expected_next

        return False  # prev_node not found or at the end of path

    def get_direction_map(self) -> dict:
        return self.direction_map
    
    def plot_segments(self, save_path: str = "segments_plot.png", show: bool = False) -> None:
        G = nx.read_graphml(self.graph_path)
        pos = {n: (float(d.get("d1", d.get("x"))), float(d.get("d2", d.get("y")))) for n, d in G.nodes(data=True)}

        base_name, ext = os.path.splitext(save_path)
        visited_nodes = set()
        colour_cycle = itertools.cycle(plt.colormaps.get_cmap("tab20").colors)
        edge_use = defaultdict(int)

        fig, ax = plt.subplots(figsize=(26, 18))
        nx.draw(G, pos, node_size=6, edge_color="lightgrey", with_labels=False, ax=ax)

        # Draw landmarks
        if isinstance(self.checkpoints[0], list):
            chk = [str(n) for region in self.checkpoints for n in region]
        else:
            chk = list(map(str, self.checkpoints))
        inter = list(map(str, self.intersections))
        nx.draw_networkx_nodes(G, pos, nodelist=chk, node_color="red", node_size=60, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=inter, node_color="dodgerblue", node_size=40, ax=ax)

        seg_handles = []
        last_node = None

        for seg_idx, nodes in enumerate(self.full_path, 1):
            if last_node is not None:
                nodes.insert(0, last_node)  # Ensure continuity with previous segment

            color = next(colour_cycle)
            seg_handles.append(plt.Line2D([0], [0], color=color, lw=3, label=f"Segment {seg_idx}"))

            for u, v in zip(nodes[:-1], nodes[1:]):
                e = (str(u), str(v))
                if e not in G.edges and (e[::-1]) in G.edges:
                    e = e[::-1]
                k = tuple(sorted(e))
                use_count = edge_use[k]
                edge_use[k] += 1
                base = 0.2 * (use_count // 2 + 1)
                rad = base if use_count % 2 else -base
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[e],
                    edge_color=[color],
                    width=3,
                    connectionstyle=f"arc3,rad={rad}",
                    ax=ax,
                    arrows=True,
                )

            # Only annotate 'Start' once
            if seg_idx == 1:
                s = str(nodes[0])
                if s in pos:
                    ax.text(*pos[s], "Start", fontsize=15, color="black", ha="center")

            visited_nodes.update(nodes)
            last_node = nodes[-1]

        # Add legend once
        landmarks = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="Checkpoint"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="dodgerblue", markersize=8, label="Intersection"),
        ]
        ax.legend(handles=seg_handles + landmarks, loc="lower right", fontsize=9, frameon=False)

        # Final save
        out_path = f"{base_name}{ext}"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)



if __name__ == "__main__":
    nav = Navigator("decisionMaking/cfg/output.graphml", "decisionMaking/cfg/cfg.json")
    hops = nav.capture_all(150)
    # print(len(nav.full_path))
    nav.plot_segments(save_path='map_plot.png')
    print('Saved to map_plot.png')
    # print(nav.full_path[0])
    # real_path = [195, 196, 197, 198, 199, 200, 292]
    # current_node = real_path[-1]
    # prev_node = real_path[-2]
    # if not nav.is_on_expected_path(prev_node, current_node): # Only check in intersections 
    #     nav.update_after_run(current_node, real_path[:-1])
    #     print("Updated path after run:", nav.full_path)
    # nav.plot_segments(save_path='b.png')
    # if not nav.is_on_expected_path(prev_node, current_node):
    #     nav.update_after_run(current_node, visited_checkpoints)

    
