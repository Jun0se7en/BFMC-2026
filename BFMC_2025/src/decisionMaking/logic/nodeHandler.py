import json
import networkx as nx
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from collections import deque
from src.decisionMaking.planning.Navigator import Navigator

# Use to find closest node and its area

class NodeHandler:
    def __init__(self, cfg_path: Union[str, Path], graph_path: Union[str, Path]) -> None:
        self.cfg_path = Path(cfg_path)
        self.graph_path = Path(graph_path)
        self._load_config()
        self.graph: nx.MultiDiGraph = nx.read_graphml(self.graph_path)

    def _load_config(self) -> None:
        with self.cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.node_area: Dict[str, Union[List[int], List[List[int]]]] = cfg.get("NODE_AREA", {})
        self.node_type: Dict[str, List[int]] = cfg.get("NODE_TYPE", {})

    def get_current_node(self, coordinates: Tuple[float, float]) -> Tuple[str, Tuple[float, float], float]:
        x, y = coordinates
        closest_node, min_dist = None, float("inf")

        for node_id, attr in self.graph.nodes(data=True):
            node_x, node_y = float(attr.get("x", 0)), float(attr.get("y", 0))
            dist = ((x - node_x) ** 2 + (y - node_y) ** 2) ** 0.5
            if dist < min_dist:
                closest_node, min_dist = node_id, dist

        if closest_node is None:
            raise ValueError("No nodes found in the graph")

        return closest_node, (float(self.graph.nodes[closest_node]["x"]), float(self.graph.nodes[closest_node]["y"])), min_dist

    def get_neighbors(self, node_id: Union[int, str]) -> List[str]:
        return list(self.graph.neighbors(str(node_id)))

    def get_successors(self, node_id: Union[int, str]) -> List[str]:
        return list(self.graph.successors(str(node_id)))

    def mapping_area(self, start_end: List[int]) -> List[int]:
        """Find nodes along path from start to end using BFS traversal."""
        if not (isinstance(start_end, list) and len(start_end) == 2):
            raise ValueError("Input must be a list of two node IDs: [start, end]")

        start, end = map(str, start_end)
        if start not in self.graph or end not in self.graph:
            raise ValueError("Start or end node not in graph")

        parent = {start: None}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            if current == end:
                break
            for succ in self.graph.successors(current):
                if succ not in parent:
                    parent[succ] = current
                    queue.append(succ)

        if end not in parent:
            return []

        path = []
        current = end
        while current is not None:
            path.append(int(current))
            current = parent[current]

        return list(reversed(path))

    def check_area_next(self, current_node: int) -> Optional[List[str]]:
        current_str = str(current_node)
        if current_str not in self.graph:
            return None

        matched_areas = set()

        for succ in self.graph.successors(current_str):
            next_node = int(succ)

            for area, values in self.node_area.items():
                if isinstance(values[0], list):
                    # Area defined by node ranges
                    for start, end in values:
                        if start <= next_node <= end or end <= next_node <= start:
                            matched_areas.add(area)
                else:
                    # Area defined by simple node list
                    if next_node in values:
                        matched_areas.add(area)

        if matched_areas:
            return sorted(matched_areas)
        return None

    # def get_node_area(self, node_id: Union[int, str]) -> Optional[str]:
    #     """Return the area name if node_id appears in any area segment path."""
    #     try:
    #         node_id = int(node_id)
    #     except ValueError:
    #         return None

    #     for area, segments in self.node_area.items():
    #         if isinstance(segments[0], int):
    #             segments = [segments]  # Convert single range to list of one

    #         for segment in segments:
    #             if len(segment) != 2:
    #                 continue
    #             path = self.mapping_area(segment)
    #             if node_id in path:
    #                 return area
    #     return None


if __name__ == "__main__":
    handler = NodeHandler("decisionMaking/cfg/cfg.json", "decisionMaking/cfg/output.graphml")
    nav = Navigator("decisionMaking/cfg/output.graphml", "decisionMaking/cfg/cfg.json")
    x, y = 2.8, 1.6  # Replace with test coordinates

    node_id, coords, distance = handler.get_current_node((x, y))
    print(f"Closest Node: {node_id} at {coords}, distance: {distance:.2f}")
    print(f"Successors: {handler.get_successors(int(node_id))}")
    nav.capture_all(int(node_id))
    print(nav.full_path)
    # for path in nav.full_path:
    #     breakpoint()
    #     handler.find_area_nodes(path)
        
    # if handler.check_area_next(current_node) is not None:
        
    
    # nav.plot_segments()
    # parking_nodes = handler.mapping_area([228, 239]) 
    # print("Parking Area Nodes:", parking_nodes)
    # print(f"Area Type: {area}")
    
