def has_cycle(graph):
    visited = set()
    path = set()
    
    def dfs(node):
        visited.add(node)
        path.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in path:
                return True
                
        path.remove(node)
        return False
    
    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    
    return False

graph = {
    'A': ['B'],
    'B': ['C'],
    'C': ['D'],
    'D': ['E'],
    'E': ['C']
}

result = has_cycle(graph)
print(result)