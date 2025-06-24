import random
import numpy as np
import math
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import time
import json
import os
import copy

# =====================================================================================
# DATASET
# =====================================================================================
n = 333; m = 3; K = 1; H = 3; T = 4; D = 2.0
ah = [2, 3, 4]; Qj = [120, 240, 660]; uj = [0.5, 1.0, 2.5]
CPj = [100, 180, 400]; CTj = [10, 10, 10]; CRj = [10, 10, 10]
Mj = [200, 200, 100]; Wjk = [[1]]; vehicle_capacity = 0.3
random.seed(42)
np.random.seed(42)
qi = [[random.uniform(0.01, 0.015)] for _ in range(n)]
Ui = [5 for _ in range(n)]; pij = [[0 for _ in range(m)] for _ in range(n)]
coordinates = [(40.9934906, 29.0379007), (40.973529, 29.0692579), (40.9696854, 29.0405382), (40.9643193, 29.0874018), (40.9683333, 29.0624902), (40.9663229, 29.0627214), (40.9712774, 29.0449134), (40.9850652, 29.0705225), (40.982768, 29.0572445), (40.9851221, 29.070817), (40.9683733, 29.084596), (40.9655572, 29.0681502), (40.9727526, 29.0511152), (40.9728716, 29.0424841), (40.9775785, 29.0792621), (40.974994, 29.0492322), (40.9687838, 29.0898996), (40.9661099, 29.0687338), (40.976943, 29.0512945), (40.9647646, 29.0646844), (40.959501, 29.07985), (40.9774233, 29.0860104), (40.972658, 29.0883856), (40.9752288, 29.0913353), (40.967665, 29.0845597), (41.0040407, 29.0445659), (40.958706, 29.108915), (40.958254, 29.09519), (40.9550448, 29.0949366), (40.9533928, 29.0953548), (40.9562612, 29.0896511), (40.9606523, 29.1041404), (40.9630001, 29.0705511), (40.9640958, 29.0739307), (40.965889, 29.07044), (40.9736124, 29.0560377), (40.9716168, 29.0596813), (40.9682063, 29.0660375), (40.9664107, 29.0694354), (40.9668609, 29.0868133), (40.9688728, 29.0616667), (40.9642725, 29.0687619), (40.9679421, 29.0605447), (40.9671744, 29.065601), (40.962972, 29.0725599), (40.9681477, 29.0574772), (40.9665695, 29.0579573), (40.9710178, 29.0555245), (40.9653145, 29.0630134), (40.9691395, 29.0686384), (40.9698704, 29.0593539), (40.9695894, 29.0564737), (40.9707793, 29.0630168), (40.9691445, 29.0639708), (40.984522, 29.026883), (40.9863111, 29.025563), (40.980807, 29.0255278), (40.9829009, 29.0324503), (40.9814896, 29.0211381), (40.9860086, 29.048691), (40.9872865, 29.0509737), (40.9741572, 29.0422626), (40.976119, 29.042435), (40.980667, 29.0414115), (40.973875, 29.047776), (40.9715352, 29.0444635), (40.9682326, 29.0426974), (40.9742322, 29.0458506), (40.9810381, 29.0382226), (40.9758231, 29.0513099), (40.980524, 29.053441), (40.9751293, 29.0806254), (40.9657425, 29.076342), (40.9687622, 29.0780962), (40.9720855, 29.075886), (40.9756011, 29.072485), (40.9777395, 29.0730858), (40.9753834, 29.0757117), (40.9746635, 29.0772057), (40.9725199, 29.0764198), (40.9679895, 29.0817266), (40.9682325, 29.0747287), (40.970999, 29.0693294), (40.9696381, 29.0721175), (40.9709838, 29.0796867), (40.9723113, 29.0686816), (40.9733119, 29.0745302), (40.9719195, 29.0774095), (40.9673302, 29.0764939), (40.9753551, 29.0775502), (40.9736695, 29.076767), (40.980457, 29.053951), (40.9807444, 29.0595367), (40.979233, 29.060318), (40.9764977, 29.064726), (40.9801107, 29.0704304), (40.9773048, 29.0570243), (40.983386, 29.0616392), (40.9996633, 29.0458211), (41.0077442, 29.0357904), (41.0061417, 29.0360981), (41.0098747, 29.0414724), (41.0058907, 29.0375546), (40.9683705, 29.0979463), (40.96887, 29.091105), (40.974684, 29.0929081), (40.9665391, 29.0983566), (40.9692165, 29.1014022), (40.9628677, 29.0970958), (40.983952, 29.072011), (40.9865379, 29.0333536), (40.9799844, 29.0794932), (40.9782797, 29.0809889), (40.9797496, 29.0865378), (40.9805395, 29.084778), (40.979562, 29.0877545), (40.9838291, 29.0897364), (40.9836018, 29.0905099), (40.9828226, 29.0869141), (40.981405, 29.0868128), (40.9836521, 29.0849143), (40.9851727, 29.0852247), (40.9848505, 29.0837082), (40.9832975, 29.0826779), (40.9866663, 29.0816549), (40.9863726, 29.0777396), (40.9849603, 29.0770477), (40.983135, 29.0762886), (40.9811127, 29.0898774), (40.9844418, 29.0789732), (40.962289, 29.076904), (40.9614298, 29.0785038), (40.9588703, 29.0839347), (40.960541, 29.079928), (40.962746, 29.075772), (40.9601146, 29.0875742), (40.9638528, 29.0889265), (40.9615868, 29.0865295), (40.9579865, 29.0857267), (40.9854194, 29.0362753), (40.9869229, 29.0434108), (41.0097118, 29.0461345), (40.977557, 29.0695671), (40.9766688, 29.0683901), (40.9751194, 29.0695252), (40.9740897, 29.0674099), (40.975188, 29.065518), (40.9732044, 29.0652001), (40.9749902, 29.0605632), (40.9734338, 29.061245), (40.9773435, 29.061155), (40.9706826, 29.093362), (40.9772475, 29.0545119), (40.9810522, 29.0609195), (40.9816395, 29.0621088), (40.9820495, 29.0635949), (40.9805353, 29.0619934), (40.9791104, 29.0652899), (40.9774256, 29.0625274), (40.9781177, 29.0608594), (40.9572193, 29.0823394), (40.9761033, 29.0890288), (40.9844993, 29.064081), (40.9884927, 29.0763153), (41.0009213, 29.0416074), (40.998868, 29.037774), (40.9804833, 29.0424684), (41.0056731, 29.0462114), (41.0039173, 29.0452956), (41.0045482, 29.0478649), (41.0058907, 29.0416154), (41.0023625, 29.0387243), (41.0002556, 29.0363483), (40.9617005, 29.0959295), (40.9601295, 29.0858528), (40.957443, 29.1024582), (40.9590565, 29.0989557), (40.9582516, 29.0802539), (40.9647709, 29.0862848), (40.9592079, 29.091026), (40.972653, 29.054678), (40.9755537, 29.0471304), (40.9604642, 29.0973598), (40.961244, 29.1008302), (40.975554, 29.083587), (40.9723877, 29.0464127), (40.9741634, 29.0428546), (40.9846583, 29.0331613), (40.9761463, 29.0498798), (40.9585875, 29.0920614), (40.9687678, 29.0718063), (40.9591489, 29.092041), (40.9786487, 29.05063), (40.9635307, 29.1005114), (40.9646937, 29.088525), (40.9613515, 29.0997748), (40.9932661, 29.0355221), (40.9753579, 29.0555213), (40.9716266, 29.0848333), (40.9710365, 29.0945461), (40.9708988, 29.0997899), (40.9862937, 29.0517891), (40.9734662, 29.0950797), (40.9614393, 29.0780577), (40.9720503, 29.088676), (40.9760135, 29.0795047), (40.9812415, 29.0756688), (40.9691395, 29.0831858), (40.9770529, 29.079255), (40.9824048, 29.0515442), (40.9814957, 29.0544299), (40.9867095, 29.053216), (40.9728855, 29.0518235), (41.0029372, 29.0450058), (40.9794534, 29.0921575), (40.9963307, 29.046744), (40.995681, 29.044205), (40.9874446, 29.0581344), (40.9597345, 29.094298), (40.9629788, 29.0898139), (40.9660829, 29.087387), (40.9616658, 29.1050172), (40.9606959, 29.1037009), (40.9576523, 29.1053959), (40.9585885, 29.1078163), (40.9597304, 29.1076202), (40.9658314, 29.1031739), (40.9715118, 29.0833729), (40.9837186, 29.0614638), (40.9814697, 29.0908318), (40.9747311, 29.0842898), (40.9812416, 29.0517564), (40.9579531, 29.0925061), (40.9709061, 29.0820057), (40.9939724, 29.0318017), (40.9698475, 29.078356), (40.9654045, 29.1003824), (40.9924343, 29.032476), (40.9676598, 29.0595031), (40.9764646, 29.0511586), (40.9874177, 29.021539), (40.9648501, 29.1007323), (40.970247, 29.071031), (40.9677102, 29.0836261), (40.9999816, 29.0313675), (40.9649529, 29.0980566), (40.9676742, 29.0898132), (40.9620749, 29.1062992), (40.9751647, 29.0792032), (40.9596975, 29.0961362), (40.9875049, 29.0653835), (40.9827749, 29.078256), (40.9577756, 29.1023897), (40.9575213, 29.0810586), (40.9668727, 29.0857825), (40.9760186, 29.0427638), (40.9685024, 29.0873269), (40.9731011, 29.0410283), (40.9921139, 29.0391372), (40.9762004, 29.0555338), (40.9545607, 29.0966793), (40.9571738, 29.094669), (40.9988424, 29.0502643), (40.9830148, 29.056061), (40.9781965, 29.0728274), (40.9942872, 29.0379297), (40.9713562, 29.0978204), (40.9638581, 29.0978986), (40.9831843, 29.0568501), (40.9664857, 29.0791607), (40.969817, 29.100101), (40.980973, 29.0660388), (40.970148, 29.0864974), (40.9852406, 29.0714248), (40.977581, 29.0659155), (40.981056, 29.086303), (40.983635, 29.083127), (40.9720931, 29.0708406), (40.9662094, 29.0908207), (40.9836513, 29.0444478), (40.9714884, 29.0707658), (40.9760497, 29.0752193), (40.9638122, 29.0802411), (40.9655526, 29.0772831), (40.9775196, 29.0462485), (40.9658564, 29.0781604), (40.9646309, 29.0782536), (40.9698639, 29.089215), (40.9680131, 29.0939548), (40.9683254, 29.0972113), (40.975982, 29.044486), (40.977615, 29.0790194), (40.965666, 29.08969), (40.9680688, 29.0743708), (40.966597, 29.0934468), (40.9729944, 29.0675765), (40.982983, 29.062974), (40.97954, 29.0416779), (40.986902, 29.0639907), (40.96012, 29.0755972), (40.9812024, 29.0781805), (40.977281, 29.047881), (40.9800653, 29.0392456), (40.9782148, 29.0478349), (40.9729873, 29.0821396), (40.9771033, 29.0735078), (40.9736467, 29.0488101), (40.9872802, 29.0403819), (41.0027792, 29.0289318), (40.9658074, 29.0805368), (40.9894382, 29.0554518), (40.978885, 29.069129), (40.9680086, 29.0832203), (40.9754406, 29.0409436), (40.9811971, 29.0449927), (40.976485, 29.0905647), (40.9668397, 29.0969217), (40.9670603, 29.0906395), (40.9977655, 29.0471327), (40.9747751, 29.0810591), (40.9822228, 29.0578427), (40.9723367, 29.0473961), (40.9950222, 29.0336035), (40.9656647, 29.0949973), (41.0016498, 29.0359117), (40.9716512, 29.0520931), (40.9761626, 29.0617872), (40.9677661, 29.0798625), (40.9650472, 29.0979052), (40.9728659, 29.0761899), (40.9639335, 29.1034061), (40.963235, 29.1000691), (40.979127, 29.075433)]
IFs = [(40.9935974, 29.0379968), (40.9743889, 29.0905937), (41.0009544, 29.032716), (40.9534609, 29.0953113), (40.9804911, 29.0260378), (40.9847034, 29.0552783), (40.9832167, 29.0458334), (40.9689129, 29.0913704), (40.9588217, 29.0724344)]
depot = (40.9750726987988, 29.06994947117117)
nodes = coordinates + [depot] + IFs

# depot_idx is defined globally.
depot_idx = n 

d_ij = np.zeros((len(nodes), len(nodes)))
def euclidean(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
for i in range(len(nodes)):
    for j in range(len(nodes)):
        d_ij[i][j] = euclidean(nodes[i], nodes[j])

def route_cost(route):
    if not route or len(route) < 2: return 0
    return sum(d_ij[route[i]][route[i + 1]] for i in range(len(route) - 1))

def solution_cost(solution):
    return sum(route_cost(r) for r in solution)

def calculate_route_details(route, k):
    load, length = 0, 0
    for i in range(len(route) - 1):
        start_node, end_node = route[i], route[i+1]
        length += d_ij[start_node][end_node]
        if end_node < n:
            load += qi[end_node][k]
        elif end_node > n: # IF (Depot is already n)
            load = 0
    return load, length

def two_opt(route):
    if len(route) <= 3: return route
    best_route = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route)):
                if j - i == 1: continue
                new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                if route_cost(new_route) < route_cost(best_route):
                    best_route = new_route
                    improved = True
                    break
            if improved:
                break
    return best_route

# CORRECTION: Capacity check has been added to this function.
def inter_route_swap(solution, k, D_constraint, capacity_constraint):
    """
    Swaps a customer between two different routes.
    The swap is applied if it does not violate LENGTH and CAPACITY constraints.
    """
    if len(solution) < 2: return None
    new_solution = copy.deepcopy(solution)
    r1_idx, r2_idx = random.sample(range(len(new_solution)), 2)
    
    route1 = new_solution[r1_idx]
    route2 = new_solution[r2_idx]

    custs1 = [node for node in route1[1:-1] if node < n]
    custs2 = [node for node in route2[1:-1] if node < n]
    if not custs1 or not custs2: return None

    node1 = random.choice(custs1)
    node2 = random.choice(custs2)
    
    node1_idx_in_r1 = route1.index(node1)
    node2_idx_in_r2 = route2.index(node2)

    route1[node1_idx_in_r1] = node2
    route2[node2_idx_in_r2] = node1

    # NEW: Both load and length are now checked.
    load1, len1 = calculate_route_details(route1, k)
    load2, len2 = calculate_route_details(route2, k)

    if len1 > D_constraint or len2 > D_constraint or load1 > capacity_constraint or load2 > capacity_constraint:
        return None # Invalidate this move if any constraint is violated.

    return new_solution

# CORRECTION: This is the final, correctly working version that fixes all previous errors.
def inter_route_relocate(solution, k, D_constraint, capacity_constraint):
    """
    Relocates a customer from one route to the best position in another.
    This version fixes indexing errors and correctly manages empty routes.
    """
    if len(solution) < 2:
        return None

    # Create a deep copy to prevent changes from affecting the original solution
    new_solution = copy.deepcopy(solution)

    # Select the source route (route_from) and the destination route (route_to)
    r_from_idx, r_to_idx = random.sample(range(len(new_solution)), 2)
    
    # Select the customer to be moved (only customer nodes, not depot or IF)
    custs_from = [node for node in new_solution[r_from_idx][1:-1] if node < n]
    if not custs_from:
        return None

    node_to_move = random.choice(custs_from)
    
    # Remove the customer from the old route
    route_from = new_solution[r_from_idx]
    route_from.remove(node_to_move)

    # If the route becomes empty after removing the customer (only [depot, depot] remains),
    # remove this route entirely from the solution. This allows for route consolidation.
    if len(route_from) < 3:
        new_solution.pop(r_from_idx)
        # Since we deleted a route, update the index of 'route_to' if it was affected
        if r_to_idx > r_from_idx:
            r_to_idx -= 1
            
    route_to = new_solution[r_to_idx]
    
    best_insert_pos = -1
    min_cost_increase = float('inf')

    # Find the best position to insert a customer (after the depot and before the depot)
    for i in range(1, len(route_to) + 1):
        
        # Create the trial route
        temp_route_to = route_to[:i] + [node_to_move] + route_to[i:]
        
        # Check the constraints
        load, length = calculate_route_details(temp_route_to, k)
        
        if length <= D_constraint and load <= capacity_constraint:
            # Calculate the cost change
            prev_node = route_to[i-1]
            next_node = route_to[i] if i < len(route_to) else depot_idx # If inserting at the end of the list, the next node is the depot
            
            cost_increase = d_ij[prev_node][node_to_move] + d_ij[node_to_move][next_node] - d_ij[prev_node][next_node]
            
            if cost_increase < min_cost_increase:
                min_cost_increase = cost_increase
                best_insert_pos = i
    
    # If a valid insertion position was found, apply the move
    if best_insert_pos != -1:
        route_to.insert(best_insert_pos, node_to_move)
        return new_solution
    
    # If no suitable position was found, invalidate the move
    return None

# CORRECTION: This function now operates on a "stop if no improvement" logic instead of fixed iterations.
def vns_for_a_days_plan(initial_solution, k, D_constraint, capacity_constraint, max_iter=5000, stagnation_limit=500):
    """
    Optimizes the entire route plan for a single day as a whole.
    It uses a dynamic stopping criterion: it stops if no improvement is found
    for a certain number of iterations. max_iter serves as a safety net.
    """
    if not initial_solution:
        return []

    best_solution = initial_solution
    best_cost = solution_cost(best_solution)
    
    neighborhoods = [inter_route_swap, inter_route_relocate]
    
    # NEW: Counter for iterations with no improvement
    non_improvement_counter = 0
    
    print(f"     VNS Initial Cost: {best_cost:.4f}, Number of Routes: {len(best_solution)}")
    print(f"     Optimization started (stagnation_limit={stagnation_limit}, max_iter={max_iter})")

    # max_iter still serves as a general upper limit.
    for i in range(max_iter):
        nh = random.choice(neighborhoods)
        candidate_solution = nh(best_solution, k, D_constraint, capacity_constraint)
        
        if candidate_solution is None:
            # An invalid or unsuccessful move also does not count as an improvement.
            non_improvement_counter += 1
            continue
        
        # Further improve the candidate with local search
        for r_idx in range(len(candidate_solution)):
            candidate_solution[r_idx] = two_opt(candidate_solution[r_idx])
        
        candidate_cost = solution_cost(candidate_solution)
        
        # If there is an improvement...
        if candidate_cost < best_cost:
            best_solution = candidate_solution
            best_cost = candidate_cost
            # NEW: Reset the counter
            non_improvement_counter = 0
        else:
            # NEW: If no improvement, increment the counter
            non_improvement_counter += 1
            
        # NEW: Check the stopping criterion
        if non_improvement_counter >= stagnation_limit:
            print(f"     -> VNS stopping: No improvement found for {stagnation_limit} iterations.")
            break
            
    # Indicate why the loop terminated when it's done
    else: # This 'else' block runs if the for loop completes without a break
        print(f"     -> VNS stopping: Maximum iteration limit ({max_iter}) reached.")

    print(f"     VNS Final Cost: {best_cost:.4f}, Number of Routes: {len(best_solution)}")
    return best_solution

# CORRECTION: This function has been completely rewritten to generate a much better initial solution for VNS.
def solve_pvrp_vns_inter_route(routing_plan, D_constraint, max_iter=500):
    total_routing_cost, detailed_routes_list, final_route_counter = 0, [], 0
    cost_multiplier = 10000 
    
    for k in range(K):
        for d in range(1, T + 1):
            unserved_customers = set(routing_plan[k][d])
            if not unserved_customers:
                continue
            
            print(f"\n--- Day {d}, Waste {k} Planning Started. Number of Customers: {len(unserved_customers)} ---")

            initial_daily_routes = []
            for cust in unserved_customers:
                # Check that each route individually adheres to the D constraint
                if d_ij[depot_idx][cust] + d_ij[cust][depot_idx] <= D_constraint:
                    initial_daily_routes.append([depot_idx, cust, depot_idx])
                else:
                    print(f"WARNING: Customer {cust} exceeds the route length constraint even alone! Could not be routed.")

            if not initial_daily_routes:
                print("WARNING: No valid initial routes could be created for this day.")
                continue

            print(f"   Initial Plan Created: {len(initial_daily_routes)} small routes are being passed to VNS.")
            
            # Optimize this generated initial plan with the holistic VNS
            optimized_daily_routes = vns_for_a_days_plan(initial_daily_routes, k, D_constraint, vehicle_capacity, max_iter=max_iter)

            
            # Save the results
            for route_nodes in optimized_daily_routes:
                if len(route_nodes) <= 2: continue
                final_route_counter += 1
                route_info = {"route_id": final_route_counter, "day": d, "waste_type": k, "nodes": route_nodes, "optimized_cost": route_cost(route_nodes)}
                detailed_routes_list.append(route_info)
                total_routing_cost += (route_cost(route_nodes) * cost_multiplier)
                
    return total_routing_cost, detailed_routes_list

def extract_plan_from_fihk_vars(solution, fihk_vars):
    plan = {k: {d: [] for d in range(1, T + 1)} for k in range(K)}
    for i in range(n):
        for k in range(K):
            for h in range(H):
                if solution.get_value(fihk_vars[i, h, k]) > 0.5:
                    num_visits = int(T / ah[h])
                    if num_visits > 0:
                        days_to_visit = random.sample(range(1, T + 1), min(T, num_visits))
                        for d in days_to_visit:
                            plan[k][d].append(i)
    return plan

# EDIT: VISUALIZATION FUNCTION ADDED
def plot_routes_by_day(coordinates, depot, IFs, detailed_routes, output_folder="daily_routes"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Folder created: {output_folder}")
    routes_by_day = {}
    for route in detailed_routes:
        day = route['day']
        if day not in routes_by_day: routes_by_day[day] = []
        routes_by_day[day].append(route)
    route_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']
    all_nodes = coordinates + [depot] + IFs
    for day, routes_for_this_day in sorted(routes_by_day.items()):
        fig, ax = plt.subplots(figsize=(18, 18))
        x_coords = [c[1] for c in coordinates]; y_coords = [c[0] for c in coordinates]
        ax.scatter(x_coords, y_coords, c='blue', label='Site (Customer)', s=20, alpha=0.3)
        ax.scatter(depot[1], depot[0], c='green', marker='*', s=300, label='Depot', zorder=5)
        if_x = [c[1] for c in IFs]; if_y = [c[0] for c in IFs]
        ax.scatter(if_x, if_y, c='red', marker='^', s=150, label='Intermediate Facility (IF)', zorder=5)
        for i, route_info in enumerate(routes_for_this_day):
            color = route_colors[i % len(route_colors)]
            route_nodes_indices = route_info['nodes']
            route_x = [all_nodes[idx][1] for idx in route_nodes_indices]; route_y = [all_nodes[idx][0] for idx in route_nodes_indices]
            ax.plot(route_x, route_y, color=color, linewidth=2, marker='o', markersize=4, label=f"Route {route_info['route_id']}")
        ax.set_title(f'Optimized Vehicle Routes - Day {day}', fontsize=20)
        ax.set_xlabel('Longitude', fontsize=14); ax.set_ylabel('Latitude', fontsize=14)
        ax.legend(loc='best', fontsize='medium'); ax.grid(True)
        fig.tight_layout(rect=[0, 0, 1, 1])
        filename = os.path.join(output_folder, f"day_{day}_routes.png")
        try:
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            print(f"✅ Map for Day {day} successfully saved as '{filename}'.")
        except Exception as e:
            print(f"❌ ERROR: Map for Day {day} could not be saved. Error: {e}")
        plt.close(fig)

# =====================================================================================
# RFBAS MODEL FUNCTIONS
# =====================================================================================
def calculate_routing_cost_estimates():
    C_route = {}
    for i in range(n):
        cost = d_ij[depot_idx][i] + d_ij[i][depot_idx]
        for k in range(K):
            C_route[i, k] = cost
    return C_route

def solve_rfbas_milp(C_route):
    mdl = Model("RFBAS_Integrated_MILP")
    # Variables...
    xijk = mdl.integer_var_dict(((i, j, k) for i in range(n) for j in range(m) for k in range(K)), name="x")
    fihk = mdl.binary_var_dict(((i, h, k) for i in range(n) for h in range(H) for k in range(K)), name="fihk")
    z_plus_ij = mdl.continuous_var_dict(((i, j) for i in range(n) for j in range(m)), name="zplus")
    z_minus_ij = mdl.continuous_var_dict(((i, j) for i in range(n) for j in range(m)), name="zminus")
    wj = mdl.continuous_var_dict(range(m), name="w")

    bin_cost_expr = mdl.sum(CPj[j] * wj[j] for j in range(m)) + \
                      mdl.sum(CRj[j] * z_minus_ij[i, j] for i in range(n) for j in range(m)) + \
                      mdl.sum(CTj[j] * z_plus_ij[i, j] for i in range(n) for j in range(m))
    routing_cost_expr = mdl.sum(fihk[i, h, k] * (T / ah[h]) * C_route[i, k]
                                for i in range(n) for h in range(H) for k in range(K))
    mdl.minimize(bin_cost_expr + routing_cost_expr)
    
    # Constraints (same as BAFRS)
    # ...
    for i in range(n):
        for k in range(K):
            mdl.add_constraint(mdl.sum(Qj[j] * xijk[i, j, k] * Wjk[k][0] for j in range(m)) >= qi[i][k] * mdl.sum(ah[h] * fihk[i, h, k] for h in range(H)))
            mdl.add_constraint(mdl.sum(fihk[i, h, k] for h in range(H)) == 1)
    for i in range(n):
        mdl.add_constraint(mdl.sum(xijk[i, j, k] * uj[j] for j in range(m) for k in range(K)) <= Ui[i])
    for j in range(m):
        mdl.add_constraint(wj[j] <= Mj[j])
    for i in range(n):
        for j in range(m):
            mdl.add_constraint(z_plus_ij[i, j] >= mdl.sum(xijk[i, j, k] for k in range(K)) - pij[i][j])
            mdl.add_constraint(z_minus_ij[i, j] >= pij[i][j] - mdl.sum(xijk[i, j, k] for k in range(K)))

    for j in range(m):
         mdl.add_constraint(wj[j] >= mdl.sum(pij[i][j] + z_plus_ij[i,j] - z_minus_ij[i,j] for i in range(n)))

    print("   Solving RFBAS MILP model...")
    sol = mdl.solve(log_output=False)
    return sol, fihk if sol else (None, None)

# CORRECTION 3: Correctly working function instead of flawed logic.
def solve_allocation_with_fixed_fihk(fihk_values):
    """Solves only the container allocation problem for given fixed fihk values and returns its cost."""
    mdl = Model("Allocation_Subproblem")
    # Variables related only to containers
    xijk = mdl.integer_var_dict(((i, j, k) for i in range(n) for j in range(m) for k in range(K)), name="x")
    z_plus_ij = mdl.continuous_var_dict(((i, j) for i in range(n) for j in range(m)), name="zplus")
    z_minus_ij = mdl.continuous_var_dict(((i, j) for i in range(n) for j in range(m)), name="zminus")
    wj = mdl.continuous_var_dict(range(m), name="w")

    # Objective function includes only the container cost
    mdl.minimize(mdl.sum(CPj[j] * wj[j] for j in range(m)) + \
                 mdl.sum(CRj[j] * z_minus_ij[i, j] for i in range(n) for j in range(m)) + \
                 mdl.sum(CTj[j] * z_plus_ij[i, j] for i in range(n) for j in range(m)))

    # Constraints use fihk as a fixed value
    for i in range(n):
        for k in range(K):
            fixed_freq_term = sum(ah[h] for h, val in enumerate(fihk_values[i,:,k]) if val > 0.5)
            mdl.add_constraint(mdl.sum(Qj[j] * xijk[i, j, k] * Wjk[k][0] for j in range(m)) >= qi[i][k] * fixed_freq_term)
    
    # ... Other constraints (Ui, Mj, z, w) remain the same
    # ...
    for i in range(n):
        mdl.add_constraint(mdl.sum(xijk[i, j, k] * uj[j] for j in range(m) for k in range(K)) <= Ui[i])
    for j in range(m):
        mdl.add_constraint(wj[j] <= Mj[j])
    for i in range(n):
        for j in range(m):
            mdl.add_constraint(z_plus_ij[i, j] >= mdl.sum(xijk[i, j, k] for k in range(K)) - pij[i][j])
            mdl.add_constraint(z_minus_ij[i, j] >= pij[i][j] - mdl.sum(xijk[i, j, k] for k in range(K)))

    for j in range(m):
         mdl.add_constraint(wj[j] >= mdl.sum(pij[i][j] + z_plus_ij[i,j] - z_minus_ij[i,j] for i in range(n)))


    sol = mdl.solve(log_output=False)
    return sol.objective_value if sol else float('inf')


# EDIT: MAIN EXECUTION FUNCTION UPDATED TO CALL VISUALIZATION
def run_rfbas_model():
    print("\n🚚 Running CORRECTED RFBAS (Route First, Bin Allocation Second)...")
    start_time = time.time()
    print("RFBAS Step 1: Estimating routing costs...")
    routing_cost_estimates = calculate_routing_cost_estimates()
    
    print("RFBAS Step 2: Finding optimal frequencies with Integrated MILP...")
    solution, fihk_vars = solve_rfbas_milp(routing_cost_estimates)
    if not solution:
        print("❌ RFBAS analysis stopped."); return

    fihk_values = np.zeros((n, H, K))
    for (i, h, k), var in fihk_vars.items():
        if solution.get_value(var) > 0.5:
            fihk_values[i, h, k] = 1

    final_routing_plan = extract_plan_from_fihk_vars(solution, fihk_vars)
    
    print("RFBAS Step 3: Calculating final and realistic costs...")
    # EDIT: We are capturing the 'detailed_routes' list returned from VNS.
    final_routing_cost, detailed_routes = solve_pvrp_vns_inter_route(final_routing_plan, D, max_iter=10000)
    final_bin_cost = solve_allocation_with_fixed_fihk(fihk_values)
    
    total_cost = final_bin_cost + final_routing_cost
    duration = time.time() - start_time
    
    print("\n--- RFBAS RESULTS ---")
    print(f"✅ Total Time: {duration:.2f} seconds")
    print(f"💰 Total Cost: {total_cost:.2f} (Container: {final_bin_cost:.2f}, Routing: {final_routing_cost:.2f})")
    
    # NEW STEP: Visualization
    print("\nRFBAS Step 4: Visualizing Daily Routes...")
    plot_routes_by_day(
        coordinates=coordinates,
        depot=depot,
        IFs=IFs,
        detailed_routes=detailed_routes,
        output_folder="rfbas_daily_routes" # Output folder
    )

    # --- NEW: Save results to a JSON file ---
    final_analysis_data = {
        "analysis_results": {
            "total_cost": round(total_cost, 2),
            "total_execution_time_sec": round(duration, 2),
            "milp_solution": { # We are simplifying this part for RFBAS
                "solution_time_sec": round(duration, 2), # An approximate value
                "total_container_cost": round(final_bin_cost, 2)
            },
            "vns_routing": {
                "total_routing_cost": round(final_routing_cost, 2),
                "total_route_count": len(detailed_routes),
                "routes": detailed_routes
            }
        },
        "coordinates": coordinates,
        "depot": depot,
        "IFs": IFs
    }
    try:
        with open("rfbas_results.json", "w", encoding="utf-8") as f:
            json.dump(final_analysis_data, f, ensure_ascii=False, indent=4)
        print("✅ Analysis results successfully saved to 'rfbas_results.json'.")
    except Exception as e:
        print(f"❌ ERROR: Results could not be saved to JSON file. Error: {e}")
    # --- END: JSON Saving ---

    return total_cost

# =====================================================================================
# MAIN EXECUTION BLOCK
# =====================================================================================
if __name__ == "__main__":
    # Now we are only running the RFBAS model.
    run_rfbas_model()
    print("\nRFBAS model execution finished.")