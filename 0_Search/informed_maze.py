import sys
import numpy as np


class Node():
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action


class StackFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class InformedFrontier_MD(StackFrontier):
    def __init__(self, Manhattan_Matrix):
        super().__init__()
        self.Manhattan_Matrix = Manhattan_Matrix

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            # node = self.frontier[min(self.Manhattan_Matrix[self.frontier])]
            min_node = min(self.frontier, key=lambda node: self.Manhattan_Matrix[node.state[0]][node.state[1]])
            self.frontier.remove(min_node)
            return min_node


class InformedFrontier_Astar(StackFrontier):
    def __init__(self, Manhattan_Matrix):
        super().__init__()
        self.Manhattan_Matrix = Manhattan_Matrix
        # self.node = node

    def add(self, node):
        self.frontier.append(node)
        self.cost = self.cost_calculator(node)
        self.heuristic = self.Manhattan_Matrix[node.state[0]][node.state[1]] + self.cost


    def cost_calculator(self, node):
        cost = 0
        while node.parent is not None:
            cost += 1
            node = node.parent

        return cost

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            # node = self.frontier[min(self.Manhattan_Matrix[self.frontier])]
            min_node = min(self.frontier, key=lambda node: self.heuristic)
            self.frontier.remove(min_node)
            return min_node


def manhattan_matrix_calculator(contents, goal):
    rows, cols = len(contents), len(contents[0])
    manhattan_matrix = []

    # Calculate Manhattan distances for each cell
    for i in range(rows):
        row = []
        for j in range(cols):
            # Implement Manhattan distance calculation logic here
            if contents[i][j] == " " or contents[i][j] == "A" or contents[i][j] == "B":
                row.append(abs(goal[0] - i) + abs(goal[1] - j))
            else:
                row.append(99999)
        manhattan_matrix.append(row)

    return manhattan_matrix


class Maze():

    def __init__(self, filename):

        # Read file and set height and width of maze
        with open(filename) as f:
            self.contents = f.read()

        # Validate start and goal
        if self.contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if self.contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")

        # Determine height and width of maze
        self.contents = self.contents.splitlines()
        self.height = len(self.contents)
        self.width = max(len(line) for line in self.contents)

        # Keep track of walls
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if self.contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif self.contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif self.contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)
        self.solution = None

        self.manhattan_matrix = manhattan_matrix_calculator(self.contents, self.goal)

    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("|X|", end="")
                elif (i, j) == self.start:
                    print("|A|", end="")
                elif (i, j) == self.goal:
                    print("|B|", end="")
                elif solution is not None and (i, j) in solution:
                    print("|O|", end="")
                else:
                    print("| |", end="")
            print()
        print()

    def neighbors(self, state):
        row, col = state
        candidates = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]

        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result

    def solve(self):
        """Finds a solution to maze, if one exists."""

        # Keep track of number of states explored
        self.num_explored = 0

        # Initialize frontier to just the starting position
        start = Node(state=self.start, parent=None, action=None)
        frontier = InformedFrontier_Astar(self.manhattan_matrix)
        frontier.add(start)

        # Initialize an empty explored set
        self.explored = set()

        # Keep looping until solution found
        while True:

            # If nothing left in frontier, then no path
            if frontier.empty():
                raise Exception("no solution")

            # Choose a node from the frontier
            node = frontier.remove()
            self.num_explored += 1

            # If node is the goal, then we have a solution
            if node.state == self.goal:
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            # Mark node as explored
            self.explored.add(node.state)

            # Add neighbors to frontier
            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action)
                    frontier.add(child)

    def output_image(self, filename, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 50
        cell_border = 2

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width * cell_size, self.height * cell_size),
            "black"
        )
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):

                # Walls
                if col:
                    fill = (40, 40, 40)

                # Start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)

                # Goal
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)

                # Solution
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)

                # Explored
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)

                # Empty cell
                else:
                    fill = (237, 240, 252)

                # Draw cell
                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                      ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )
                bbox = draw.textbbox((j * cell_size, i * cell_size), str(self.manhattan_matrix[i][j]), font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                draw.text(
                    (j * cell_size + (cell_size - text_width) // 2, i * cell_size + (cell_size - text_height) // 2),
                    str(self.manhattan_matrix[i][j]),
                    fill=(0, 0, 0),
                    font=font
                )
        filename_image = filename + '_informed_Astar.png'
        img.save(filename_image)


if len(sys.argv) != 2:
    sys.exit("Usage: python maze.py maze.txt")

m = Maze(sys.argv[1])
print("Maze:")
m.print()
# print('Manhattan matrix :', m.manhattan_matrix)
print('goal is at : ', m.goal)

print("Solving...")
m.solve()
print("States Explored:", m.num_explored)
print("Solution:")
m.print()
m.output_image(sys.argv[1], show_explored=True)
