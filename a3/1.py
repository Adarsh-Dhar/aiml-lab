from collections import deque

def find_path_bfs(maze):
    rows = len(maze)
    cols = len(maze[0])
    start = (0, 0)
    end = (rows - 1, cols - 1)
    
    queue = deque([(start, [start])])
    visited = {start}
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        (curr_row, curr_col), path = queue.popleft()
        
        if (curr_row, curr_col) == end:
            return path
            
        for dx, dy in directions:
            next_row, next_col = curr_row + dx, curr_col + dy
            
            if (0 <= next_row < rows and 
                0 <= next_col < cols and 
                maze[next_row][next_col] == 0 and 
                (next_row, next_col) not in visited):
                
                visited.add((next_row, next_col))
                queue.append(((next_row, next_col), path + [(next_row, next_col)]))
    
    return "No path found."

maze = [
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 0]
]

result = find_path_bfs(maze)
print(result)