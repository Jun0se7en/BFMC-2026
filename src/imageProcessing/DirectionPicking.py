import math
import json
import heapq

class DirectionPicking:
    def __init__(self, config_file):
        self.load_data(config_file)
        self.forced_edge_directions =  {(11, 14): "go_right",
                                        (11, 7): "go_left",
                                        (9, 7): "go_right",
                                        (9, 14): "go_left",
                                        (13, 11): "go_right",
                                        (13, 12): "go_left",
                                        (8, 12): "go_right",
                                        (8, 10): "go_left",
                                        } 

    def load_data(self, config_file):
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            self.dict_nodes_coordination = {int(k): tuple(v) for k, v in data["dict_nodes_coordination"].items()}
            self.neighbor_nodes = {int(k): v for k, v in data["neighbor_nodes"].items()}
            self.from_lst = data["from_lst"]
            self.to_lst = data["to_lst"]
        except FileNotFoundError:
            print("Error: Config file not found at:", config_file)

    def compute_distance(self, frm, to):
        x1, y1 = self.dict_nodes_coordination[frm]
        x2, y2 = self.dict_nodes_coordination[to]
        return math.hypot(x2 - x1, y2 - y1)

    def compute_heading(self, frm, to):
        x1, y1 = self.dict_nodes_coordination[frm]
        x2, y2 = self.dict_nodes_coordination[to]
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

    def relative_turn(self, reference_heading, leg_heading):
        # Compute the smallest difference in angles.
        diff = (leg_heading - reference_heading + 180) % 360 - 180
        if diff == 0:
            return "go_forward"
        return "go_left" if diff > 0 else "go_right"

    def find_optimal_path(self, start, destination):
        # Allow both states at the start.
        best = {(start, 0): 0, (start, 1): 0}
        parent = {(start, 0): None, (start, 1): None}
        pq = [(0, start, 0), (0, start, 1)]
        final_state = None

        while pq:
            cost, current, state = heapq.heappop(pq)
            if current == destination:
                final_state = (current, state)
                break
            if cost > best.get((current, state), float('inf')):
                continue

            expected_next = self.to_lst if state == 0 else self.from_lst
            for neighbor in self.neighbor_nodes.get(current, []):
                # If neighbor isn't destination and isn't in the expected list, skip.
                if neighbor != destination and neighbor not in expected_next:
                    continue

                new_cost = cost + self.compute_distance(current, neighbor)
                new_state = (state + 1) % 2
                if new_cost < best.get((neighbor, new_state), float('inf')):
                    best[(neighbor, new_state)] = new_cost
                    parent[(neighbor, new_state)] = (current, state)
                    heapq.heappush(pq, (new_cost, neighbor, new_state))

        if final_state is None:
            raise ValueError(f"No optimal path found from {start} to {destination}.")

        # Backtrack to build the path.
        path = []
        cur = final_state
        while cur is not None:
            path.append(cur[0])
            cur = parent[cur]
        return path[::-1]

    def find_path_directions(self, traveled_nodes):
        directions = []
        start = traveled_nodes[0]
        if start not in self.neighbor_nodes or not self.neighbor_nodes[start]:
            raise ValueError(f"Node {start} has no neighbor.")

        # Use the first neighbor to set an initial reference heading.
        current_ref = self.compute_heading(start, self.neighbor_nodes[start][0])

        for i in range(len(traveled_nodes) - 1):
            frm = traveled_nodes[i]
            to = traveled_nodes[i + 1]
            leg_heading = self.compute_heading(frm, to)
            # If movement is strictly horizontal or vertical, mark as "go_forward"
            if (self.dict_nodes_coordination[frm][0] == self.dict_nodes_coordination[to][0] or
                self.dict_nodes_coordination[frm][1] == self.dict_nodes_coordination[to][1]):
                turn = "go_forward"
            else:
                turn = self.relative_turn(current_ref, leg_heading)
            directions.append(turn)
            current_ref = leg_heading

        return directions

    def process_route(self, start, destination):
        # Direct neighbor check.
        if destination in self.neighbor_nodes.get(start, []):
            path = [start, destination]
        else:
            path = self.find_optimal_path(start, destination)
        directions = self.find_path_directions(path)
        return path, directions

    def modify_path(self, base_path, modified_from, modified_to, destination):
        if modified_from not in base_path:
            raise ValueError(f"Modified 'from' node {modified_from} not found in base path.")

        idx_from = base_path.index(modified_from)
        new_base_path = base_path[:idx_from + 1]

        new_segment = self.find_optimal_path(modified_from, modified_to)
        final_segment = self.find_optimal_path(modified_to, destination)

        modified_path = new_base_path + new_segment[1:] + final_segment[1:]
        modified_directions = self.find_path_directions(modified_path)

        for i in range(len(modified_path) - 1):
            edge = (modified_path[i], modified_path[i + 1])
            if edge in self.forced_edge_directions:
                modified_directions[i] = self.forced_edge_directions[edge]

        return modified_path, modified_directions

    def find_neighbor(self, node, direction):
        neighbors = self.neighbor_nodes.get(node, [])
        if len(neighbors) <= 1:
            return None
        node_coord = self.dict_nodes_coordination[node]
        if direction == "go_forward":
            for nbr in neighbors:
                nbr_coord = self.dict_nodes_coordination[nbr]
                if node_coord[0] == nbr_coord[0] or node_coord[1] == nbr_coord[1]:
                    return nbr
        for nbr in neighbors:
            nbr_coord = self.dict_nodes_coordination[nbr]
            if not (node_coord[0] == nbr_coord[0] or node_coord[1] == nbr_coord[1]):
                if (node, nbr) in self.forced_edge_directions and self.forced_edge_directions[(node, nbr)] == direction:
                    return nbr
        ref = self.compute_heading(node, neighbors[0])
        if direction == "go_left":
            target = (ref + 90) % 360
        elif direction == "go_right":
            target = (ref - 90) % 360
        elif direction == "go_forward":
            target = ref % 360
        else:
            raise ValueError("Direction must be 'go_left', 'go_right', or 'go_forward'.")
        best = None
        best_diff = float('inf')
        for nbr in neighbors:
            heading = self.compute_heading(node, nbr) % 360
            diff = min(abs(heading - target), 360 - abs(heading - target))
            if diff < best_diff:
                best_diff = diff
                best = nbr
        return best


if __name__ == '__main__':
    config_file = 'map_nodes.json'
    dp = DirectionPicking(config_file)

    # Process a normal route.
    start_node = 8
    destination_node = 20
    path, directions = dp.process_route(start_node, destination_node)
    print("Processed Route:")
    print("Path:", path)
    print("Directions:", directions[::2])

    # Modify the route: for example, change segment starting at 8 to go to 10
    new_path, new_directions = dp.modify_path(path, modified_from=8, modified_to=10, destination=destination_node)
    print("\nModified Route:")
    print("Path:", new_path)
    print("Directions:", new_directions[::2])

    print("Neighbor: ", dp.find_neighbor(9, 'go_left'))
