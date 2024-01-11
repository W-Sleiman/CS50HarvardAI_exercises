import sys


class Node():
    def __init__(self, state, parent, action, manh_dist=0, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.manh_dist = manh_dist
        self.cost = cost


# Depth-First Search (DFS) Frontier
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


# Breadth-First Search (BFS) Frontier
class QueueFrontier(StackFrontier):

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node

class InformedFrontier_Astar(StackFrontier):
    def __init__(self, maze):
        super().__init__()
        # self.manhattan_matrix = Manhattan_Matrix
        self.maze = maze


    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            heuristic = []
            for i, node in enumerate(self.frontier):
                print('node.manh_dist = ',node.manh_dist)
                print()
                heuristic.append(self.maze.nodes[node.state].manh_dist + cost_calculator(node))     # adding +1 to take into account that this node hasn't been reached/explored yet, it is one node/step away
            print('heuristic  = ',heuristic)
                # print('Frontier heuristic = ',heuristic)

            min_node = self.frontier[heuristic.index(min(heuristic))]
            print('min_node.state  = ',min_node.state)
            self.frontier.remove(min_node)

        return min_node




def manhattan_distance_matrix_calculator(maze, contents):
    maze = maze
    rows, cols = len(contents), len(contents[0])
    walls_dist = (rows + cols) * 99
    manhattan_matrix = []

    # Calculate Manhattan distances for each cell
    for i in range(rows):
        row = []
        for j in range(cols):
            # Implement Manhattan distance calculation logic here
            if contents[i][j] == " " or contents[i][j] == "A" or contents[i][j] == "B":

                distance = abs(maze.goal[0] - i) + abs(maze.goal[1] - j)
                row.append(distance)

                node = maze.nodes.get((i, j))
                if node:
                    # Update node's manhattan distance attribute
                    print(node.state,'.manh_dist = ',node.manh_dist)
                    node.manh_dist = distance
                    print(node.state,'.manh_dist = ',node.manh_dist)
                    # Update manhattan distance in nodes dictionary
                    maze.nodes[node.state].manh_dist = distance

            else:
                row.append(walls_dist)

                node = maze.nodes.get((i, j))
                if node:
                    # Update node's manhattan distance attribute
                    node.manh_dist = walls_dist
                    # Update manhattan distance in nodes dictionary
                    maze.nodes[node.state].manh_dist = walls_dist

        manhattan_matrix.append(row)

    return manhattan_matrix


def cost_calculator(node):
    cost = 0

    # Count node parents
    while node.parent is not None:
        cost += 1
        node = node.parent

    # Set node's cost attribute
    node.cost = cost

    return cost

class Maze():

    def __init__(self, filename):

        # Read file and set height and width of maze
        with open(filename) as f:
            contents = f.read()

        # Validate start and goal
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")

        # Determine height and width of maze
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        # Keep track of walls
        self.walls = []
        # Create a dictionary with node.state as keys and nodes as values
        self.nodes = {}

        # Iterate over all the elements of contents
        for i in range(self.height):
            row = []
            for j in range(self.width):

                # Initialize node and populate it into dictionary
                node = Node(state=(i, j), parent=None, action=None, manh_dist=0, cost=0)
                self.nodes[(i, j)] = node

                # Find and distinguish between start, goal, wall, and path nodes
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.manh_dist_matrix = manhattan_distance_matrix_calculator(self, contents)
        for i in range(self.height):
            for j in range(self.width):
                Node(state=(i, j), parent=None, action=None, manh_dist=self.manh_dist_matrix[i][j], cost=0)


        self.solution = None

        # print('manh_dist_matrix = ',self.manh_dist_matrix)
        # print('node(5,0).manh_dist = ',self.nodes[(5, 0)].manh_dist)
        # print('node(5,0).manh_dist = ',node.state[(5, 0)].manh_dist)
        # print('node(4,1).manh_dist = ',self.nodes[(4,1)].manh_dist)
        # print('node(3,1).manh_dist = ',self.nodes[(3,1)].manh_dist)
        # print('node(4,2).manh_dist = ',self.nodes[(4,2)].manh_dist)
        # print()

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
                    print("|*|", end="")
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
        # frontier = InformedFrontier_Astar(self)
        frontier = StackFrontier()
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

            # Update costs in dictionary
            node.cost = cost_calculator(node)
            self.nodes[node.state].cost = node.cost
            # print("Cost:", node.cost)

            # Update node's manhattan distance from the nodes dictionary
            node.manh_dist = self.nodes[node.state].manh_dist


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
                node = self.nodes[(i, j)]
                text = str(node.manh_dist) + '+' + str(node.cost)
                # print('text = ',text)

                bbox = draw.textbbox((j * cell_size, i * cell_size), str(text), font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                # for node in node.state():
                draw.text(
                    (j * cell_size + (cell_size - text_width) // 2, i * cell_size + (cell_size - text_height) // 2),
                    str(text),
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
print("Solving...")
m.solve()
print("States Explored:", m.num_explored)
print("Solution:")
m.print()
m.output_image("mazeY", show_explored=True)
# print(m.nodes)
