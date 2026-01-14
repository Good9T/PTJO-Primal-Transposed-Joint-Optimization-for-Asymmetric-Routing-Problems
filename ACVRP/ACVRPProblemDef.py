import torch

def get_random_problems(batch_size, node_num, problem_gen_params):

    scaler = problem_gen_params['scaler']
    capacity = problem_gen_params['capacity']
    demand_min = problem_gen_params['demand_min']
    demand_max = problem_gen_params['demand_max']
    perturb_min = problem_gen_params.get('perturb_min', 0.1)
    perturb_max = problem_gen_params.get('perturb_max', 0.5)
    asym_bias = problem_gen_params.get('asym_bias', 0.2)

    node_coords = torch.rand(batch_size, node_num, 2)
    coord_diff = node_coords.unsqueeze(2) - node_coords.unsqueeze(1)
    straight_dist = torch.norm(coord_diff, p=2, dim=-1)
    # L2

    perturb_ratio = torch.rand(batch_size, node_num, node_num) * (perturb_max - perturb_min) + perturb_min
    # perturb

    asym_bias_matrix = torch.rand(batch_size, node_num, node_num) * asym_bias
    asym_bias_matrix = asym_bias_matrix - asym_bias_matrix.transpose(1, 2) / 2
    # forced asymmetric

    problems_distance = straight_dist * (1 + perturb_ratio + asym_bias_matrix)
    problems_distance = torch.max(problems_distance, straight_dist)
    # force perturb distance >= straight distance
    problems_distance[:, torch.arange(node_num), torch.arange(node_num)] = 0.0
    # force self-distance = 0

    while True:
        old_problems_distance = problems_distance.clone()
        problems_distance, _ = (problems_distance[:, :, None, :] + problems_distance[:, None, :, :].transpose(2, 3)).min(dim=3)
        if (problems_distance == old_problems_distance).all():
            break

    scaled_problems = problems_distance.float() / scaler

    node_demand = torch.randint(low=demand_min, high=demand_max, size=(batch_size, node_num)) / float(capacity)
    node_demand[:, 0] = 0

    problems_demand = node_demand.unsqueeze(1).expand(batch_size, node_num, node_num)

    problems = torch.stack((scaled_problems, problems_demand), dim=3)
    # shape: (batch, node, node, 2)
    trans_problems = torch.transpose(problems, 1, 2)
    return problems, node_demand, trans_problems, node_coords

def load_single_problem_from_file(filename, node_num, scaler):
    problem = torch.empty(size=(node_num, node_num), dtype=torch.long)
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as err:
        print(err)
    line_num = 0
    for line in lines:
        linedata = line.split()
        if linedata[0].startswith('TYPE', 'DIMENSION', 'EDGE_WEIGHT_TYPE', 'EDGE_WEIGHT_FORMAT', 'EDGE_WEIGHT_SECTION','EOF'):
            continue
        int_map = map(int, linedata)
        int_list = list(int_map)
    problem[line_num] = torch.tensor(int_list, dtype=torch.long)
    line_num += 1
    problem[torch.arange(node_num), torch.arange(node_num)] = 0
    scaled_problems = problem.float() / scaler
    return scaled_problems