import numpy as np
import matplotlib.pyplot as plt

class IntersectionHandler:
    def __init__(self, graph):
        self.graph = graph

    def rotate_vector(self, vec, angle_rad):
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        return rot_matrix @ vec

    def bezier_curve(self, p0, p1, p2, num_points=5):
        return [
            (
                (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0],
                (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
            )
            for t in np.linspace(0, 1, num_points)
        ]

    def get_curve(self, prev, start, end, direction, num_points=5):
        x0, y0 = prev
        x1, y1 = start
        x2, y2 = end

        if direction == "forward":
            return [
                (
                    (1 - t) * x1 + t * x2,
                    (1 - t) * y1 + t * y2
                )
                for t in np.linspace(0, 1, num_points)
            ]

        # Compute heading vector from prev -> start
        heading = np.array([x1 - x0, y1 - y0])
        heading_norm = heading / np.linalg.norm(heading)

        # Midpoint between start and end
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Flip convention: 'right' makes the path curve leftward
        angle_rad = -np.pi / 2 if direction == "left" else np.pi / 2
        lateral_offset = self.rotate_vector(heading_norm, angle_rad) * 0.3
        ctrl_x = mid_x + lateral_offset[0]
        ctrl_y = mid_y + lateral_offset[1]

        return self.bezier_curve(start, (ctrl_x, ctrl_y), end, num_points=num_points)

    def update(self, prev_pos, current_pos, next_pos, direction, num_points=5):
        return self.get_curve(prev_pos, current_pos, next_pos, direction, num_points=num_points)

    def draw_path(self, path):
        x_vals, y_vals = zip(*path)

        plt.figure(figsize=(5, 5))
        plt.plot(x_vals, y_vals, marker='o')
        plt.title("Path Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')
        plt.grid(True)
        plt.show()
        
if __name__ == "__main__":
    # Example usage
    graph = None  # Replace with actual graph object
    handler = IntersectionHandler(graph)
    prev = (1.55, 4.02)
    start = (1.29, 4.02)
    end = (0.72, 4.59)
    direction = "forward"  # Can be 'forward', 'left', or 'right'

    curve_path = handler.update(prev, start, end, direction)
    print("Curve Path:", curve_path)
    handler.draw_path(curve_path)