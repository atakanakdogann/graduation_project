import random
import numpy as np
import math
from docplex.mp.model import Model
import time
import pandas as pd
import copy
import matplotlib.pyplot as plt
import json



MASTER_COORDINATES = [(40.9934906, 29.0379007), (40.973529, 29.0692579), (40.9696854, 29.0405382), (40.9643193, 29.0874018), (40.9683333, 29.0624902), (40.9663229, 29.0627214), (40.9712774, 29.0449134), (40.9850652, 29.0705225), (40.982768, 29.0572445), (40.9851221, 29.070817), (40.9683733, 29.084596), (40.9655572, 29.0681502), (40.9727526, 29.0511152), (40.9728716, 29.0424841), (40.9775785, 29.0792621), (40.974994, 29.0492322), (40.9687838, 29.0898996), (40.9661099, 29.0687338), (40.976943, 29.0512945), (40.9647646, 29.0646844), (40.959501, 29.07985), (40.9774233, 29.0860104), (40.972658, 29.0883856), (40.9752288, 29.0913353), (40.967665, 29.0845597), (41.0040407, 29.0445659), (40.958706, 29.108915), (40.958254, 29.09519), (40.9550448, 29.0949366), (40.9533928, 29.0953548), (40.9562612, 29.0896511), (40.9606523, 29.1041404), (40.9630001, 29.0705511), (40.9640958, 29.0739307), (40.965889, 29.07044), (40.9736124, 29.0560377), (40.9716168, 29.0596813), (40.9682063, 29.0660375), (40.9664107, 29.0694354), (40.9668609, 29.0868133), (40.9688728, 29.0616667), (40.9642725, 29.0687619), (40.9679421, 29.0605447), (40.9671744, 29.065601), (40.962972, 29.0725599), (40.9681477, 29.0574772), (40.9665695, 29.0579573), (40.9710178, 29.0555245), (40.9653145, 29.0630134), (40.9691395, 29.0686384), (40.9698704, 29.0593539), (40.9695894, 29.0564737), (40.9707793, 29.0630168), (40.9691445, 29.0639708), (40.984522, 29.026883), (40.9863111, 29.025563), (40.980807, 29.0255278), (40.9829009, 29.0324503), (40.9814896, 29.0211381), (40.9860086, 29.048691), (40.9872865, 29.0509737), (40.9741572, 29.0422626), (40.976119, 29.042435), (40.980667, 29.0414115), (40.973875, 29.047776), (40.9715352, 29.0444635), (40.9682326, 29.0426974), (40.9742322, 29.0458506), (40.9810381, 29.0382226), (40.9758231, 29.0513099), (40.980524, 29.053441), (40.9751293, 29.0806254), (40.9657425, 29.076342), (40.9687622, 29.0780962), (40.9720855, 29.075886), (40.9756011, 29.072485), (40.9777395, 29.0730858), (40.9753834, 29.0757117), (40.9746635, 29.0772057), (40.9725199, 29.0764198), (40.9679895, 29.0817266), (40.9682325, 29.0747287), (40.970999, 29.0693294), (40.9696381, 29.0721175), (40.9709838, 29.0796867), (40.9723113, 29.0686816), (40.9733119, 29.0745302), (40.9719195, 29.0774095), (40.9673302, 29.0764939), (40.9753551, 29.0775502), (40.9736695, 29.076767), (40.980457, 29.053951), (40.9807444, 29.0595367), (40.979233, 29.060318), (40.9764977, 29.064726), (40.9801107, 29.0704304), (40.9773048, 29.0570243), (40.983386, 29.0616392), (40.9996633, 29.0458211), (41.0077442, 29.0357904), (41.0061417, 29.0360981), (41.0098747, 29.0414724), (41.0058907, 29.0375546), (40.9683705, 29.0979463), (40.96887, 29.091105), (40.974684, 29.0929081), (40.9665391, 29.0983566), (40.9692165, 29.1014022), (40.9628677, 29.0970958), (40.983952, 29.072011), (40.9865379, 29.0333536), (40.9799844, 29.0794932), (40.9782797, 29.0809889), (40.9797496, 29.0865378), (40.9805395, 29.084778), (40.979562, 29.0877545), (40.9838291, 29.0897364), (40.9836018, 29.0905099), (40.9828226, 29.0869141), (40.981405, 29.0868128), (40.9836521, 29.0849143), (40.9851727, 29.0852247), (40.9848505, 29.0837082), (40.9832975, 29.0826779), (40.9866663, 29.0816549), (40.9863726, 29.0777396), (40.9849603, 29.0770477), (40.983135, 29.0762886), (40.9811127, 29.0898774), (40.9844418, 29.0789732), (40.962289, 29.076904), (40.9614298, 29.0785038), (40.9588703, 29.0839347), (40.960541, 29.079928), (40.962746, 29.075772), (40.9601146, 29.0875742), (40.9638528, 29.0889265), (40.9615868, 29.0865295), (40.9579865, 29.0857267), (40.9854194, 29.0362753), (40.9869229, 29.0434108), (41.0097118, 29.0461345), (40.977557, 29.0695671), (40.9766688, 29.0683901), (40.9751194, 29.0695252), (40.9740897, 29.0674099), (40.975188, 29.065518), (40.9732044, 29.0652001), (40.9749902, 29.0605632), (40.9734338, 29.061245), (40.9773435, 29.061155), (40.9706826, 29.093362), (40.9772475, 29.0545119), (40.9810522, 29.0609195), (40.9816395, 29.0621088), (40.9820495, 29.0635949), (40.9805353, 29.0619934), (40.9791104, 29.0652899), (40.9774256, 29.0625274), (40.9781177, 29.0608594), (40.9572193, 29.0823394), (40.9761033, 29.0890288), (40.9844993, 29.064081), (40.9884927, 29.0763153), (41.0009213, 29.0416074), (40.998868, 29.037774), (40.9804833, 29.0424684), (41.0056731, 29.0462114), (41.0039173, 29.0452956), (41.0045482, 29.0478649), (41.0058907, 29.0416154), (41.0023625, 29.0387243), (41.0002556, 29.0363483), (40.9617005, 29.0959295), (40.9601295, 29.0858528), (40.957443, 29.1024582), (40.9590565, 29.0989557), (40.9582516, 29.0802539), (40.9647709, 29.0862848), (40.9592079, 29.091026), (40.972653, 29.054678), (40.9755537, 29.0471304), (40.9604642, 29.0973598), (40.961244, 29.1008302), (40.975554, 29.083587), (40.9723877, 29.0464127), (40.9741634, 29.0428546), (40.9846583, 29.0331613), (40.9761463, 29.0498798), (40.9585875, 29.0920614), (40.9687678, 29.0718063), (40.9591489, 29.092041), (40.9786487, 29.05063), (40.9635307, 29.1005114), (40.9646937, 29.088525), (40.9613515, 29.0997748), (40.9932661, 29.0355221), (40.9753579, 29.0555213), (40.9716266, 29.0848333), (40.9710365, 29.0945461), (40.9708988, 29.0997899), (40.9862937, 29.0517891), (40.9734662, 29.0950797), (40.9614393, 29.0780577), (40.9720503, 29.088676), (40.9760135, 29.0795047), (40.9812415, 29.0756688), (40.9691395, 29.0831858), (40.9770529, 29.079255), (40.9824048, 29.0515442), (40.9814957, 29.0544299), (40.9867095, 29.053216), (40.9728855, 29.0518235), (41.0029372, 29.0450058), (40.9794534, 29.0921575), (40.9963307, 29.046744), (40.995681, 29.044205), (40.9874446, 29.0581344), (40.9597345, 29.094298), (40.9629788, 29.0898139), (40.9660829, 29.087387), (40.9616658, 29.1050172), (40.9606959, 29.1037009), (40.9576523, 29.1053959), (40.9585885, 29.1078163), (40.9597304, 29.1076202), (40.9658314, 29.1031739), (40.9715118, 29.0833729), (40.9837186, 29.0614638), (40.9814697, 29.0908318), (40.9747311, 29.0842898), (40.9812416, 29.0517564), (40.9579531, 29.0925061), (40.9709061, 29.0820057), (40.9939724, 29.0318017), (40.9698475, 29.078356), (40.9654045, 29.1003824), (40.9924343, 29.032476), (40.9676598, 29.0595031), (40.9764646, 29.0511586), (40.9874177, 29.021539), (40.9648501, 29.1007323), (40.970247, 29.071031), (40.9677102, 29.0836261), (40.9999816, 29.0313675), (40.9649529, 29.0980566), (40.9676742, 29.0898132), (40.9620749, 29.1062992), (40.9751647, 29.0792032), (40.9596975, 29.0961362), (40.9875049, 29.0653835), (40.9827749, 29.078256), (40.9577756, 29.1023897), (40.9575213, 29.0810586), (40.9668727, 29.0857825), (40.9760186, 29.0427638), (40.9685024, 29.0873269), (40.9731011, 29.0410283), (40.9921139, 29.0391372), (40.9762004, 29.0555338), (40.9545607, 29.0966793), (40.9571738, 29.094669), (40.9988424, 29.0502643), (40.9830148, 29.056061), (40.9781965, 29.0728274), (40.9942872, 29.0379297), (40.9713562, 29.0978204), (40.9638581, 29.0978986), (40.9831843, 29.0568501), (40.9664857, 29.0791607), (40.969817, 29.100101), (40.980973, 29.0660388), (40.970148, 29.0864974), (40.9852406, 29.0714248), (40.977581, 29.0659155), (40.981056, 29.086303), (40.983635, 29.083127), (40.9720931, 29.0708406), (40.9662094, 29.0908207), (40.9836513, 29.0444478), (40.9714884, 29.0707658), (40.9760497, 29.0752193), (40.9638122, 29.0802411), (40.9655526, 29.0772831), (40.9775196, 29.0462485), (40.9658564, 29.0781604), (40.9646309, 29.0782536), (40.9698639, 29.089215), (40.9680131, 29.0939548), (40.9683254, 29.0972113), (40.975982, 29.044486), (40.977615, 29.0790194), (40.965666, 29.08969), (40.9680688, 29.0743708), (40.966597, 29.0934468), (40.9729944, 29.0675765), (40.982983, 29.062974), (40.97954, 29.0416779), (40.986902, 29.0639907), (40.96012, 29.0755972), (40.9812024, 29.0781805), (40.977281, 29.047881), (40.9800653, 29.0392456), (40.9782148, 29.0478349), (40.9729873, 29.0821396), (40.9771033, 29.0735078), (40.9736467, 29.0488101), (40.9872802, 29.0403819), (41.0027792, 29.0289318), (40.9658074, 29.0805368), (40.9894382, 29.0554518), (40.978885, 29.069129), (40.9680086, 29.0832203), (40.9754406, 29.0409436), (40.9811971, 29.0449927), (40.976485, 29.0905647), (40.9668397, 29.0969217), (40.9670603, 29.0906395), (40.9977655, 29.0471327), (40.9747751, 29.0810591), (40.9822228, 29.0578427), (40.9723367, 29.0473961), (40.9950222, 29.0336035), (40.9656647, 29.0949973), (41.0016498, 29.0359117), (40.9716512, 29.0520931), (40.9761626, 29.0617872), (40.9677661, 29.0798625), (40.9650472, 29.0979052), (40.9728659, 29.0761899), (40.9639335, 29.1034061), (40.963235, 29.1000691), (40.979127, 29.075433)]
MASTER_IFS = [(40.9935974, 29.0379968), (40.9743889, 29.0905937), (41.0009544, 29.032716), (40.9534609, 29.0953113), (40.9804911, 29.0260378), (40.9847034, 29.0552783), (40.9832167, 29.0458334), (40.9689129, 29.0913704), (40.9588217, 29.0724344)]
MASTER_DEPOT = (40.9750726987988, 29.06994947117117)

def generate_dataset(num_customers):
    print(f"--- PREPARING DATASET FOR {num_customers} CUSTOMERS... ---")
    n = num_customers
    m, K, H, T, o, s = 2, 1, 2, 2, 4, 1

    depot_coord = MASTER_DEPOT
    if_coords = MASTER_IFS[:s]
    
    random.seed(42)
    anchor_point = MASTER_COORDINATES[50]
    
    def euclidean(p1, p2): return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    distances_to_anchor = [(coord, euclidean(coord, anchor_point)) for coord in MASTER_COORDINATES]
    
    sorted_customers = sorted(distances_to_anchor, key=lambda x: x[1])
    
    customer_coords = [coord for coord, dist in sorted_customers[:n]]

    depot_node = 0
    customer_nodes = list(range(1, n + 1))
    if_nodes = list(range(n + 1, n + s + 1))
    all_nodes = [depot_node] + customer_nodes + if_nodes
    
    coordinates = {depot_node: depot_coord}
    for i, coord in enumerate(customer_coords):
        coordinates[customer_nodes[i]] = coord
    for i, coord in enumerate(if_coords):
        coordinates[if_nodes[i]] = coord

    d_ij = { (i, j): euclidean(coordinates[i], coordinates[j]) for i in all_nodes for j in all_nodes }

    params = {
        'n': n, 'm': m, 'K': K, 'H': H, 'T': T, 'o': o, 's': s,
        'ah': [1, 2],
        'Qj': {0: 20, 1: 30}, 'uj': {0: 2, 1: 3},
        'CPj': {0: 10000, 1: 15000}, 
        'CRj': {0: 200, 1: 300}, 
        'CTj': {0: 100, 1: 100},
        'Mj': {0: 100, 1: 100}, 'Wjk': {(0, 0): 1, (1, 0): 1},
        'vehicle_capacity': 500, 
        'D': 10000,
        'qi': {i: random.randint(5, 10) for i in customer_nodes},
        'Ui': {i: 20 for i in customer_nodes},
        'si': {i: 3 for i in customer_nodes},
        'pij': {(i, j): 0 for i in customer_nodes for j in range(m)},
        'Ci': {i: list(range(H)) for i in customer_nodes},
        'lambda_rh': [[1, 0], [0, 1]],
        'art': [[1, 0], [0, 1]],
        'depot_node': depot_node, 'customer_nodes': customer_nodes, 'if_nodes': if_nodes, 'all_nodes': all_nodes,
        'd_ij': d_ij, 'coordinates': coordinates
    }
    return params


def save_results_to_json(solution, params, duration, routing_cost, bin_cost, xijlt):
    print("Writing results to JSON file...")
    
    detailed_routes = reconstruct_routes_from_solution(solution, xijlt, params)
    
    n_cust = params['n']
    customer_coords = [params['coordinates'][i] for i in params['customer_nodes']]
    depot_coord = params['coordinates'][params['depot_node']]
    if_coords = [params['coordinates'][i] for i in params['if_nodes']]
    
    # Create mapping for node indices (for GUI compatibility)
    old_to_new_map = {params['depot_node']: n_cust}
    for i, old_idx in enumerate(params['customer_nodes']):
        old_to_new_map[old_idx] = i
    for i, old_idx in enumerate(params['if_nodes']):
        old_to_new_map[old_idx] = n_cust + 1 + i

    # Apply mapping to routes
    for route in detailed_routes:
        try:
            route['nodes'] = [old_to_new_map[node_idx] for node_idx in route['nodes']]
        except KeyError as e:
            print(f"ERROR: Route {route['route_id']} index mapping error. Unknown node: {e}")
            continue

    final_analysis_data = {
        "analysis_results": {
            "total_cost": solution.objective_value,
            "total_execution_time_sec": duration,
            "milp_solution": {
                "solution_time_sec": duration,
                "total_container_cost": bin_cost.solution_value if hasattr(bin_cost, 'solution_value') else solution.get_value(bin_cost)
            },
            "vns_routing": {
                "total_routing_cost": routing_cost.solution_value if hasattr(routing_cost, 'solution_value') else solution.get_value(routing_cost),
                "total_route_count": len(detailed_routes),
                "routes": detailed_routes
            }
        },
        "coordinates": customer_coords,
        "depot": depot_coord,
        "IFs": if_coords
    }

    try:
        with open("im_results.json", "w", encoding="utf-8") as f:
            json.dump(final_analysis_data, f, ensure_ascii=False, indent=4)
        print("Analysis results saved to 'im_results.json' file.")
    except Exception as e:
        print(f"Error: Results couldn't be written to 'im_results.json' file. Error: {e}")

def run_integrated_model(params):
    print("\n🚚 Running FINAL INTEGRATED MODEL (IM) from Section 6.2...")
    start_time = time.time()
    
    n, m, o, s, T, H, K = params['n'], params['m'], params['o'], params['s'], params['T'], params['H'], params['K']
    depot_node, customer_nodes, if_nodes, all_nodes = params['depot_node'], params['customer_nodes'], params['if_nodes'], params['all_nodes']
    Cij, CPj, CRj, CTj, Qj, uj, Mj = params['d_ij'], params['CPj'], params['CRj'], params['CTj'], params['Qj'], params['uj'], params['Mj']
    qi, ah, Ui, D, Q, si, pij = params['qi'], params['ah'], params['Ui'], params['D'], params['vehicle_capacity'], params['si'], params['pij']
    Wjk, Ci, lambda_rh, art = params['Wjk'], params['Ci'], params['lambda_rh'], params['art']

    mdl = Model(name="WBARP_Integrated_Corrected")

    xijlt = mdl.binary_var_dict(((i,j,l,t) for i in all_nodes for j in all_nodes for l in range(o) for t in range(T) if i != j), name="x")
    vijlt = mdl.continuous_var_dict(((i,j,l,t) for i in all_nodes for j in all_nodes for l in range(o) for t in range(T) if i != j), name="v")
    yir = mdl.binary_var_dict(((i,r) for i in customer_nodes for r in Ci[i]), name="y")
    fih = mdl.binary_var_dict(((i,h) for i in customer_nodes for h in range(H)), name="f")
    xijk = mdl.integer_var_dict(((i,j,k) for i in customer_nodes for j in range(m) for k in range(K)), name="bin_alloc")
    z_plus = mdl.continuous_var_dict(((i,j) for i in customer_nodes for j in range(m)), name="z_plus")
    z_minus = mdl.continuous_var_dict(((i,j) for i in customer_nodes for j in range(m)), name="z_minus")
    wj = mdl.continuous_var_dict(range(m), name="wj")
    
    # FIXED: Better balanced routing vs bin costs - make routing more attractive
    routing_scaling_factor = 1.0
    
    routing_cost = mdl.sum(Cij[i,j] * routing_scaling_factor * xijlt[i,j,l,t] for (i,j,l,t) in xijlt)
    bin_cost = mdl.sum(CPj[j] * wj[j] for j in range(m)) + \
               mdl.sum(CRj[j] * z_minus[i,j] for i,j in z_minus) + \
               mdl.sum(CTj[j] * z_plus[i,j] for i,j in z_plus)
    mdl.minimize(routing_cost + bin_cost)

    # Constraints
    # Customer assignment to routes
    for i in customer_nodes:
        mdl.add_constraint(mdl.sum(yir[i,r] for r in Ci[i]) == 1)
    
    # Frequency assignment
    for i in customer_nodes:
        for h in range(H):
            mdl.add_constraint(mdl.sum(yir[i,r] * lambda_rh[r][h] for r in Ci[i]) == fih[i,h])
    
    # Visit requirement based on route and time
    for i in customer_nodes:
        for t in range(T):
            is_visited = mdl.sum(xijlt.get((j,i,l,t), 0) for j in all_nodes for l in range(o) if j!=i)
            should_be_visited = mdl.sum(yir[i,r] * art[r][t] for r in Ci[i])
            mdl.add_constraint(is_visited == should_be_visited)
    
    # Flow conservation
    for h_node in all_nodes:
        for l in range(o):
            for t in range(T):
                in_flow = mdl.sum(xijlt.get((j,h_node,l,t), 0) for j in all_nodes if j!=h_node)
                out_flow = mdl.sum(xijlt.get((h_node,j,l,t), 0) for j in all_nodes if j!=h_node)
                mdl.add_constraint(in_flow == out_flow)
    
    # Vehicle usage constraint
    for l in range(o):
        for t in range(T):
            mdl.add_constraint(mdl.sum(xijlt.get((depot_node,j,l,t), 0) for j in customer_nodes) <= 1)
    
    # Time/distance constraints
    for l in range(o):
        for t in range(T):
            tour_distance = mdl.sum(Cij[i,j] * xijlt[i,j,l,t] for i, j in Cij if (i,j,l,t) in xijlt)
            tour_service_time = mdl.sum(si.get(i,0) * mdl.sum(xijlt[j,i,l,t] for j in all_nodes if j!=i) for i in customer_nodes)
            mdl.add_constraint(tour_distance + tour_service_time <= D)
    
    # Vehicle capacity constraints
    for i, j, l, t in vijlt:
        mdl.add_constraint(vijlt[i,j,l,t] <= Q * xijlt[i,j,l,t])
    
    # Load flow constraints
    for j_node in customer_nodes:
        for l in range(o):
            for t in range(T):
                in_load = mdl.sum(vijlt.get((i,j_node,l,t), 0) for i in all_nodes if i!=j_node)
                out_load = mdl.sum(vijlt.get((j_node,i,l,t), 0) for i in all_nodes if i!=j_node)
                is_visited = mdl.sum(xijlt.get((i,j_node,l,t), 0) for i in all_nodes if i!=j_node)
                mdl.add_constraint(out_load - in_load >= qi[j_node] - Q * (1 - is_visited))
    
    # IF constraints - no outgoing load
    for p in if_nodes:
        for l in range(o):
            for t in range(T):
                mdl.add_constraint(mdl.sum(vijlt.get((p,j,l,t), 0) for j in all_nodes if j!=p) == 0)
    
    # Depot outgoing load constraint
    for j in all_nodes:
        if j == depot_node: continue
        for l in range(o):
            for t in range(T):
                mdl.add_constraint(vijlt.get((depot_node,j,l,t), 0) == 0)
    
    """# No direct return to depot from customers (must visit IF first if needed)
    for i in customer_nodes:
        for l in range(o):
            for t in range(T):
                mdl.add_constraint(xijlt.get((i,depot_node,l,t), 0) == 0)
"""
    # Vehicle departure constraint - if vehicle is used, it must depart from depot
    for l in range(o):
        for t in range(T):
            vehicle_departs = mdl.sum(xijlt.get((depot_node, j, l, t), 0) for j in all_nodes if j != depot_node)
            for i in all_nodes:
                if i == depot_node: continue
                for j in all_nodes:
                    if i == j: continue
                    mdl.add_constraint(xijlt[i, j, l, t] <= vehicle_departs)

    # Bin allocation constraints
    for i in customer_nodes:
        max_waste = mdl.sum(fih[i,h] * ah[h] for h in range(H)) * qi[i]
        total_cap = mdl.sum(Qj[j] * xijk.get((i,j,k),0) for j in range(m) for k in range(K) if Wjk.get((j,k),0)==1)
        mdl.add_constraint(total_cap >= max_waste)
    
    for i in customer_nodes:
        mdl.add_constraint(mdl.sum(uj[j] * mdl.sum(xijk.get((i,j,k),0) for k in range(K)) for j in range(m)) <= Ui[i])
    
    for j in range(m):
        mdl.add_constraint(wj[j] <= Mj[j])
    
    # Purchase/removal constraints
    for i in customer_nodes:
        for j in range(m):
            final_bins = mdl.sum(xijk.get((i,j,k),0) for k in range(K))
            initial_bins = pij.get((i,j),0)
            mdl.add_constraint(z_plus[i,j] >= final_bins - initial_bins)
            mdl.add_constraint(z_minus[i,j] >= initial_bins - final_bins)
            mdl.add_constraint(final_bins == initial_bins + z_plus[i,j] - z_minus[i,j])
    
    for j in range(m):
        mdl.add_constraint(wj[j] >= mdl.sum(z_plus.get((i,j),0) - z_minus.get((i,j),0) for i in customer_nodes))
    
    # Each customer must have exactly one frequency
    for i in customer_nodes:
        mdl.add_constraint(mdl.sum(fih[i,h] for h in range(H)) == 1)

    print("Integrated Model (IM) is solving...")
    # Set time limit and optimality gap to ensure we get a solution
    mdl.parameters.timelimit = 300  # 5 minutes
    mdl.parameters.mip.tolerances.mipgap = 0.05  # 5% optimality gap
    
    solution = mdl.solve(log_output=True)
    
    duration = time.time() - start_time
    print("\n--- IM RESULTS ---")
    if solution:
        print(f"Total Time: {duration:.2f} seconds")
        rc_val = solution.get_value(routing_cost)
        bc_val = solution.get_value(bin_cost)
        print(f"Total COST: {solution.objective_value:.2f} (Container: {bc_val:.2f}, Routing: {rc_val:.2f})")
        print(f"Solve Details: {solution.solve_details.status}")
        return solution, params, duration, routing_cost, bin_cost, xijlt
    else:
        print(f"Solution couldn't be found. Time: {duration:.2f} seconds")
        print(f"Solve Details: {mdl.solve_details.status if hasattr(mdl, 'solve_details') else 'Unknown'}")
        return None, None, None, None, None, None

def reconstruct_routes_from_solution(solution, xijlt_vars, params):

    routes = []
    depot = params['depot_node']
    T, o = params['T'], params['o']
    d_ij = params['d_ij']
    
    route_id_counter = 1
    
    # Debug: Print all active arcs
    print("\n=== Active Arcs in Solution ===")
    active_arcs = []
    for (i, j, l_var, t_var), var in xijlt_vars.items():
        if solution.get_value(var) > 0.5:
            active_arcs.append((i, j, l_var, t_var))
            print(f"Arc: {i} -> {j}, Vehicle: {l_var}, Day: {t_var}")
    
    if not active_arcs:
        print("WARNING: No active arcs found in the solution!")
        return routes
    
    for t in range(T):
        for l in range(o):
            # Get all arcs for this vehicle and day
            day_vehicle_arcs = [(i, j) for (i, j, l_var, t_var) in active_arcs 
                                if l_var == l and t_var == t]
            
            if not day_vehicle_arcs:
                continue
                
            print(f"\nDay {t+1}, Vehicle {l+1} active arcs: {day_vehicle_arcs}")
            
            # Build adjacency dictionary
            arc_dict = {}
            for i, j in day_vehicle_arcs:
                if i not in arc_dict:
                    arc_dict[i] = []
                arc_dict[i].append(j)
            
            # Find complete routes starting and ending at depot
            if depot in arc_dict:
                # For each outgoing edge from depot, try to build a complete route
                used_arcs = set()
                
                for next_node in arc_dict[depot]:
                    if (depot, next_node) in used_arcs:
                        continue
                        
                    path = [depot]
                    current_node = depot
                    visited_customers = set()
                    
                    # Follow the path until we return to depot or get stuck
                    while True:
                        if current_node in arc_dict:
                            # Find next unvisited node
                            next_candidates = []
                            for candidate in arc_dict[current_node]:
                                if (current_node, candidate) not in used_arcs:
                                    next_candidates.append(candidate)
                            
                            if not next_candidates:
                                break
                                
                            # Choose next node (prefer depot if available and we've visited customers)
                            next_node_choice = None
                            if depot in next_candidates and visited_customers:
                                next_node_choice = depot
                            else:
                                # Take first available customer
                                for candidate in next_candidates:
                                    if candidate != depot:
                                        next_node_choice = candidate
                                        break
                                if next_node_choice is None and next_candidates:
                                    next_node_choice = next_candidates[0]
                            
                            if next_node_choice is None:
                                break
                                
                            # Mark arc as used
                            used_arcs.add((current_node, next_node_choice))
                            path.append(next_node_choice)
                            
                            if next_node_choice != depot:
                                visited_customers.add(next_node_choice)
                            
                            current_node = next_node_choice
                            
                            # If we're back at depot, we have a complete route
                            if current_node == depot and len(path) > 2:
                                break
                        else:
                            break
                    
                    # Only add valid complete routes
                    if len(path) >= 3 and path[0] == depot and path[-1] == depot:
                        cost_of_route = sum(d_ij.get((path[i], path[i+1]), 0) for i in range(len(path) - 1))
                        
                        route_info = {
                            "route_id": route_id_counter,
                            "day": t + 1,
                            "waste_type": 0,
                            "nodes": path,
                            "optimized_cost": cost_of_route,
                        }
                        routes.append(route_info)
                        route_id_counter += 1
                        print(f"Route found: {path}, Cost: {cost_of_route:.2f}")
    
    print(f"\nTotal {len(routes)} routes found.")
    return routes

def plot_solution(params, solution, xijlt_vars):

    print("Visualizing the solution...")
    coordinates = params['coordinates']
    depot_node = params['depot_node']
    if_nodes = params['if_nodes']
    customer_nodes = params['customer_nodes']
    T = params['T']
    o = params['o']
    
    # First reconstruct the routes properly
    routes = reconstruct_routes_from_solution(solution, xijlt_vars, params)
    
    route_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6']

    for t in range(T):
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot customers
        cust_x = [coordinates.get(i, (None, None))[1] for i in customer_nodes if coordinates.get(i)]
        cust_y = [coordinates.get(i, (None, None))[0] for i in customer_nodes if coordinates.get(i)]
        ax.scatter(cust_x, cust_y, c='blue', label='Customer', s=80, alpha=0.7, zorder=3)

        # Add customer labels
        for i in customer_nodes:
            coord = coordinates.get(i)
            if coord:
                ax.annotate(f'C{i}', (coord[1], coord[0]), xytext=(5, 5), 
                            textcoords='offset points', fontsize=8, ha='left')

        # Plot depot
        depot_coord = coordinates.get(depot_node)
        if depot_coord:
            ax.scatter(depot_coord[1], depot_coord[0], c='green', marker='*', 
                       s=400, label='Depot', zorder=5, edgecolors='black', linewidth=1)
            ax.annotate('DEPOT', (depot_coord[1], depot_coord[0]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10, fontweight='bold')

        # Plot intermediate facilities
        if_x = [coordinates.get(i, (None, None))[1] for i in if_nodes if coordinates.get(i)]
        if_y = [coordinates.get(i, (None, None))[0] for i in if_nodes if coordinates.get(i)]
        ax.scatter(if_x, if_y, c='red', marker='^', s=200, label='Intermediate Facility (IF)', 
                   zorder=4, edgecolors='black', linewidth=1)
        
        for i in if_nodes:
            coord = coordinates.get(i)
            if coord:
                ax.annotate(f'IF{i}', (coord[1], coord[0]), xytext=(5, 5), 
                            textcoords='offset points', fontsize=9, fontweight='bold')

        # Plot routes for this day
        day_routes = [route for route in routes if route['day'] == t + 1]
        
        for idx, route in enumerate(day_routes):
            color = route_colors[idx % len(route_colors)]
            path = route['nodes']
            
            # Draw the route
            for i in range(len(path) - 1):
                coord_i = coordinates.get(path[i])
                coord_j = coordinates.get(path[i + 1])
                if coord_i and coord_j:
                    ax.plot([coord_i[1], coord_j[1]], [coord_i[0], coord_j[0]], 
                            color=color, linewidth=3, alpha=0.8, zorder=2)
                    
                    # Add arrow to show direction
                    if i < len(path) - 2:
                        mid_x = (coord_i[1] + coord_j[1]) / 2
                        mid_y = (coord_i[0] + coord_j[0]) / 2
                        dx = coord_j[1] - coord_i[1]
                        dy = coord_j[0] - coord_i[0]
                        ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), 
                                    xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
            
            # Add route label
            if len(path) > 2:
                mid_idx = len(path) // 2
                mid_coord = coordinates.get(path[mid_idx])
                if mid_coord:
                    ax.annotate(f'R{route["route_id"]}', (mid_coord[1], mid_coord[0]), 
                                xytext=(0, -15), textcoords='offset points', 
                                fontsize=9, fontweight='bold', color=color,
                                ha='center', bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', alpha=0.8))

        ax.set_title(f'Integrated Model Solution - Day {t+1}\n{len(day_routes)} Routes', fontsize=16)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio and tight layout
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()

        filename = f"solution_day_{t+1}_fixed.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Solution plot saved as '{filename}'.")
        plt.close()

def plot_routes_separately(params, solution, xijlt_vars):

    print("Visualizing routes separately...")
    coordinates = params['coordinates']
    depot_node = params['depot_node']
    if_nodes = params['if_nodes']
    customer_nodes = params['customer_nodes']
    T = params['T']
    
    routes = reconstruct_routes_from_solution(solution, xijlt_vars, params)
    
    if not routes:
        print("There are no routes to visualize")
        return
    
    # Create subplots for each day
    fig, axes = plt.subplots(1, T, figsize=(6*T, 6))
    if T == 1:
        axes = [axes]
    
    route_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8']
    
    for t in range(T):
        ax = axes[t]
        
        # Plot all nodes first
        cust_x = [coordinates.get(i, (None, None))[1] for i in customer_nodes if coordinates.get(i)]
        cust_y = [coordinates.get(i, (None, None))[0] for i in customer_nodes if coordinates.get(i)]
        ax.scatter(cust_x, cust_y, c='lightblue', s=60, alpha=0.6, zorder=2)
        
        depot_coord = coordinates.get(depot_node)
        if depot_coord:
            ax.scatter(depot_coord[1], depot_coord[0], c='green', marker='*', s=300, zorder=5)
        
        if_x = [coordinates.get(i, (None, None))[1] for i in if_nodes if coordinates.get(i)]
        if_y = [coordinates.get(i, (None, None))[0] for i in if_nodes if coordinates.get(i)]
        ax.scatter(if_x, if_y, c='red', marker='^', s=150, zorder=4)
        
        # Plot routes for this day
        day_routes = [route for route in routes if route['day'] == t + 1]
        
        for idx, route in enumerate(day_routes):
            color = route_colors[idx % len(route_colors)]
            path = route['nodes']
            
            # Highlight visited customers
            for node in path[1:-1]:
                coord = coordinates.get(node)
                if coord and node in customer_nodes:
                    ax.scatter(coord[1], coord[0], c=color, s=100, zorder=3, alpha=0.8)
            
            # Draw route
            for i in range(len(path) - 1):
                coord_i = coordinates.get(path[i])
                coord_j = coordinates.get(path[i + 1])
                if coord_i and coord_j:
                    ax.plot([coord_i[1], coord_j[1]], [coord_i[0], coord_j[0]], 
                            color=color, linewidth=4, alpha=0.9, zorder=3)
                    ax.scatter([coord_i[1], coord_j[1]], [coord_i[0], coord_j[0]], 
                               c=color, s=30, zorder=4)
        
        ax.set_title(f'Day {t+1} - {len(day_routes)} Routes')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle('Daily Route Distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig('daily_routes_comparison.png', dpi=300, bbox_inches='tight')
    print("Daily route comparison saved as 'daily_routes_comparison.png'.")
    plt.close()

if __name__ == "__main__":
    print("==========================================================")
    print("           Integrated Model (IM) Test")
    print("==========================================================")
    
    test_params = generate_dataset(num_customers=40)

    solution, params, duration, routing_cost, bin_cost, xijlt = run_integrated_model(test_params)

    if solution:
        save_results_to_json(solution, params, duration, routing_cost, bin_cost, xijlt)
        plot_solution(params, solution, xijlt)      # Plotting the solution
        plot_routes_separately(params, solution, xijlt) # Bonus: Separate visualization

    print("\nAnalysis completed.")