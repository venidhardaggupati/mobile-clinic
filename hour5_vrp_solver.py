"""
hour5_vrp_solver.py — The Logic Engine (V2.0 Fine-Tuned)
"""
from typing import Any
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

def solve_routing(
    data: dict[str, Any],
    fleet_size: int = 2,
    max_time: int = 480,  
) -> dict[str, Any]:
    
    data["num_vehicles"] = fleet_size
    num_nodes = len(data["time_matrix"])
    num_vehicles = data["num_vehicles"]
    depot = data["depot"]
    cases = data.get("cases", [0] * num_nodes)
    prizes = data.get("prizes", [0.0] * num_nodes)
    village_ids = data["village_ids"]

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # ── 1. TIME CALLBACK ──────────────────────────────────────────────────────
    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        travel_time = int(data["time_matrix"][from_node][to_node])

        if from_node == depot:
            service_time = 0
        else:
            # FIX 1: Cap service time at 120 mins max so massive outbreaks don't break the shift
            raw_service_time = int(30 + (cases[from_node] * 1.5))
            service_time = min(120, raw_service_time)

        return int(travel_time + service_time)

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # ── 2. TIME BUDGET DIMENSION ──────────────────────────────────────────────
    routing.AddDimension(
        int(transit_callback_index),
        slack_max=0,
        capacity=int(max_time), 
        fix_start_cumul_to_zero=True,
        name="Time",
    )

    # ── 3. PRIZE-COLLECTING DISJUNCTIONS ──────────────────────────────────────
    for node_idx in range(num_nodes):
        if node_idx == depot:
            continue
            
        # FIX 2: Multiply by 10,000. Skipping a node is now mathematically devastating.
        penalty = int(prizes[node_idx] * 10000) 
        
        if penalty == 0:
            penalty = 500 # Even stable villages have a base penalty to encourage visits
            
        routing.AddDisjunction([manager.NodeToIndex(node_idx)], penalty)

    # ── 4. SOLVE METRICS ──────────────────────────────────────────────────────
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = 5

    solution = routing.SolveWithParameters(search_params)

    # ── 5. EXTRACT MULTI-VAN ROUTES ───────────────────────────────────────────
    if not solution:
        return {"route_ids": {}, "total_time": 0, "status": "NO_SOLUTION"}

    route_ids = {}
    vehicle_route_times = []
    time_dimension = routing.GetDimensionOrDie("Time")

    for vehicle_id in range(num_vehicles):
        van_name = f"Van_{vehicle_id + 1}"
        index = routing.Start(vehicle_id)
        route_nodes = []

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(village_ids[node])
            index = solution.Value(routing.NextVar(index))

        end_node = manager.IndexToNode(index)
        route_nodes.append(village_ids[end_node])

        route_time = int(solution.Min(time_dimension.CumulVar(index)))
        vehicle_route_times.append(route_time)

        non_depot_visits = [v for v in route_nodes if v != village_ids[depot]]
        if non_depot_visits:
            route_ids[van_name] = route_nodes

    total_time = int(max(vehicle_route_times)) if vehicle_route_times else 0

    return {
        "route_ids": route_ids,
        "total_time": total_time,
        "status": "SUCCESS",
    }