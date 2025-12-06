import math
import networkx as nx

# Find shortest path between two nodes
# Input: [(start1, end1), (start2, end2), ...]
# 
class PathFinder:
    def __init__(self, config_file: str):
        self._load_graph(config_file)

    def _load_graph(self, file_path: str) -> None:
        try:
            g = nx.read_graphml(file_path)
        except Exception as exc:
            raise RuntimeError(f"Cannot read {file_path}: {exc}") from exc

        self.graph = g
        self.xy = {int(n): (float(d["x"]), float(d["y"])) for n, d in g.nodes(data=True)}
        self.neigh = {int(n): list(map(int, g.neighbors(n))) for n in g.nodes()}

        self.intersection_nodes = []
        for node in g.nodes():
            successors = list(g.successors(node)) if g.is_directed() else list(g.neighbors(node))
            predecessors = list(g.predecessors(node)) if g.is_directed() else list(g.neighbors(node))
            if len(successors) + len(predecessors) > 3:
                self.intersection_nodes.append(int(node))

    def _dist(self, a: int, b: int) -> float:
        x1, y1 = self.xy[a]
        x2, y2 = self.xy[b]
        return math.hypot(x2 - x1, y2 - y1)

    def _heading(self, a: int, b: int) -> float:
        x1, y1 = self.xy[a]
        x2, y2 = self.xy[b]
        return math.degrees(math.atan2(y2 - y1, x2 - x1))  # ° CCW from +x

    def _find_previous_node(self, curr: int) -> int:
        incoming = [int(u) for u, v in self.graph.edges() if int(v) == curr]
        if not incoming:
            raise ValueError(f"No incoming node found for node {curr}")
        return incoming[0]
    def _find_next_node(self, curr: int) -> int:
        outgoing = [int(v) for u, v in self.graph.edges() if int(u) == curr]
        if not outgoing:
            raise ValueError(f"No outgoing node found for node {curr}")
        return outgoing

    @staticmethod
    def _signed_angle(ref: float, leg: float) -> float:
        diff = (leg - ref + 180) % 360 - 180
        return -diff

    def _path_directions(self, path):
        if len(path) < 3:
            return []

        directions = []
        ref = self._heading(path[0], path[1])

        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            nxt = path[i + 1]

            if curr in self.intersection_nodes:
                angle_in = self._signed_angle(self._heading(prev, curr), self._heading(curr, nxt))
                turn_in = "go_forward" if angle_in == 0 else ("go_right" if angle_in > 0 else "go_left")
                directions.append(((prev, curr), turn_in, angle_in))

                angle_out = self._signed_angle(self._heading(curr, nxt), self._heading(nxt, path[i+2]) if i+2 < len(path) else self._heading(curr, nxt))
                turn_out = "go_forward" if angle_out == 0 else ("go_right" if angle_out > 0 else "go_left")
                directions.append(((curr, nxt), turn_out, angle_out))

                ref = self._heading(curr, nxt)

        return directions

    def process_route(self, start_end):
        if not (isinstance(start_end, (list, tuple)) and len(start_end) == 2):
            raise ValueError("Input must be a tuple like (curr, destination)")

        curr, dest = map(int, start_end)
        prev = self._find_previous_node(curr)

        sub = nx.shortest_path(
            self.graph,
            str(curr),
            str(dest),
            weight=lambda u, v, d: self._dist(int(u), int(v)),
        )
        sub = list(map(int, sub))
        full_path = [prev] + sub

        return full_path, self._path_directions(full_path)

    def find_optimal_path(self, start: int, end: int):
        sub = nx.shortest_path(
            self.graph,
            str(start),
            str(end),
            weight=lambda u, v, d: self._dist(int(u), int(v)),
        )
        return list(map(int, sub))

    def modify_path(self, base_path, modified_from, modified_to, destination):
        if modified_from not in base_path:
            raise ValueError(f"Modified 'from' node {modified_from} not found in base path.")

        idx_from = base_path.index(modified_from)
        new_base_path = base_path[:idx_from + 1]

        new_segment = self.find_optimal_path(modified_from, modified_to)
        final_segment = self.find_optimal_path(modified_to, destination)

        modified_path = new_base_path + new_segment[1:] + final_segment[1:]
        modified_directions = self._path_directions(modified_path)

        return modified_path, modified_directions

if __name__ == "__main__":
    import time
    start_time = time.time()
    dp = PathFinder("output.graphml")

    # start_end = (224, 189)
    # path, directions = dp.process_route(start_end)

    # for (a, b), turn, angle in directions:
    #     print(f"({a}, {b}): {turn:>11s}   angle: {angle:+.2f}°")

    # print("\nModifying path...")

    # This modify path adjust whole path + new adjusted path (in case of no entry)
    # new_path, new_directions = dp.modify_path(path, 290, 302, 189)

    # for (a, b), turn, angle in new_directions:
    #     print(f"({a}, {b}): {turn:>11s}   angle: {angle:+.2f}°")
    
    
    nodes_des_list = [(313, 397), (337, 82)]
    for start, end in nodes_des_list:
        path, directions = dp.process_route((start, end))
        print(time.time() - start_time)

        print(f"Path from {start} to {end}:")
        print('-----------')
        for (a, b), turn, angle in directions:
            print(f"({a}, {b}): {turn:>11s}   angle: {angle:+.2f}°")

