import math
from typing import List, Tuple

class RandomStartHandler:
    def __init__(self, graphml_path: str):
        import networkx as nx

        self.G = nx.read_graphml(graphml_path)

        self.graph_positions = {
            int(n): (float(data['x']), float(data['y']))
            for n, data in self.G.nodes(data=True)
        }

        self.graph_edges = [
            (int(u), int(v)) for u, v in self.G.edges()
        ]

    def _euclidean_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _custom_angle_between(self, p_from, p_to):
        """Custom angle in degrees: Ox is 0/180, Oy is 90, negative for Q3/Q4"""
        dx = p_to[0] - p_from[0]
        dy = p_to[1] - p_from[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle  # already conforms to sign convention

    def _angular_difference(self, a, b):
        return min((360 - abs(a - b)) % 360, abs(a - b))

    def _average_pose(self, data: List[Tuple[float, float, float]]):
        xs = [p[0] for p in data]
        ys = [p[1] for p in data]
        thetas = [p[2] for p in data]

        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)

        # Use average direction from first to last position, not theta values
        motion_angle = self._custom_angle_between((xs[0], ys[0]), (xs[-1], ys[-1]))

        return (avg_x, avg_y, motion_angle)

    def find_best_matching_edge(self, gps_list: List[Tuple[float, float, float]]) -> Tuple[int, int]:
        """
        Input: list of (x, y, theta) samples (at least 1)
        Output: best matching edge (prev_node, current_node)
        """
        if not gps_list:
            raise ValueError("At least one GPS sample is required")

        if len(gps_list) == 1:
            x, y, _ = gps_list[0]
            best_pair = None
            best_distance = float('inf')

            for u, v in self.graph_edges:
                pu = self.graph_positions[u]
                pv = self.graph_positions[v]
                mx, my = (pu[0] + pv[0]) / 2, (pu[1] + pv[1]) / 2
                dist = self._euclidean_distance((x, y), (mx, my))
                if dist < best_distance:
                    best_distance = dist
                    best_pair = (u, v)

            return best_pair

        avg_x, avg_y, motion_angle = self._average_pose(gps_list)

        best_pair = None
        best_score = float('inf')

        for u, v in self.graph_edges:
            pu = self.graph_positions[u]
            pv = self.graph_positions[v]

            mx, my = (pu[0] + pv[0]) / 2, (pu[1] + pv[1]) / 2
            midpoint_distance = self._euclidean_distance((avg_x, avg_y), (mx, my))

            edge_angle = self._custom_angle_between(pu, pv)
            angle_diff = self._angular_difference(edge_angle, motion_angle)

            score = midpoint_distance + (angle_diff / 90.0)
            if score < best_score:
                best_score = score
                best_pair = (u, v)

        return best_pair


if __name__ == "__main__":
    rsh = RandomStartHandler("decisionMaking/cfg/output.graphml")
    
    gps_data = [
        (2.88, 2.18, 80),
        (2.87, 2.23, 82),
        (2.87, 2.26, 84),
        (2.86, 2.27, 85),
        (2.87, 2.28, 85)
    ]

    prev_node, current_node = rsh.find_best_matching_edge(gps_data)
    print("Best matching edge:", prev_node, "→", current_node)


