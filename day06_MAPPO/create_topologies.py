import json

topologies = {
    "topology_full.json": {
        "mode": "static",
        "adj_matrix": [[0,1,1],[1,0,1],[1,1,0]],
        "n_agents": 3,
        "description": "Fully connected"
    },
    "topology_ring.json": {
        "mode": "static",
        "adj_matrix": [[0,1,0],[0,0,1],[1,0,0]],
        "n_agents": 3,
        "description": "Ring"
    },
    "topology_dynamic.json": {
        "mode": "dynamic",
        "comm_radius": 1.0,
        "n_agents": 3,
        "description": "Distance-based dynamic"
    }
}

for filename, content in topologies.items():
    with open(filename, "w") as f:
        json.dump(content, f, indent=4)
    print(f"Oluşturuldu: {filename}")