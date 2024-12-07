import numpy as np
import time
from mip import Model, xsum, minimize, INTEGER
from dijkstra import Graph
from dijkstra import DijkstraSPF
from sklearn.linear_model import LinearRegression

np.random.seed(9824)


### create functions
def get_routing_table(cost_matrix, ports):
    for n in range(len(ports)):
        globals()[f'port_{n}'] = nodes = list(range(len(ports)))
    graph = Graph()
    for n in range(len(ports)):
        for m in range(len(ports)):
            if n != m:
                graph.add_edge(nodes[n], nodes[m], cost_matrix[n][m])

    routing_table = np.full((len(ports), len(ports), len(ports)), np.inf)
    for n in range(len(ports)):
        dijkstra = DijkstraSPF(graph, n)
        print('for port', n, "\n%s %s" % ("destination", "distance"))
        for u in nodes:
            print("%s %d" % (u, dijkstra.get_distance(u)))
        for u in nodes:
            temp_path_lst = dijkstra.get_path(u)
            print('dijkstra path lst for origin', n, 'destination', u, '\n', temp_path_lst)
            if len(temp_path_lst) <= 2:
                temp_path_lst = np.full((len(ports)), np.inf)
            else:
                temp_path_lst.pop(0)
                temp_path_lst.pop(-1)
                while len(temp_path_lst) < len(ports):
                    temp_path_lst = np.append(temp_path_lst, np.inf)
            temp_path_lst = np.reshape(temp_path_lst, (1, 1, len(ports)))
            routing_table[n, u, :] = temp_path_lst
    return routing_table


def get_cost_matrix(moving_average_matrix, max_travel_time_matrix, ship_models, ports, moving_average_length):
    cost_matrix = np.zeros((len(ports), len(ports)))
    for n in range(len(ports)):
        for m in range(len(ports)):
            if n != m:
                cost_matrix[n][m] = np.ceil(
                    max_travel_time_matrix[n][m][len(ship_models) - 1] + ship_models[len(ship_models) - 1][1] / (
                                moving_average_matrix[n][m] + 1 / moving_average_length))
    return cost_matrix


def get_travel_time_matrix(distance_matrix, ship_models, ports):
    max_travel_time_matrix = np.zeros((len(ports), len(ports), len(ship_models)))
    min_travel_time_matrix = np.zeros((len(ports), len(ports), len(ship_models)))
    for n in range(len(ports)):
        for m in range(len(ports)):
            if n != m:
                for l in range(len(ship_models)):
                    min_travel_time_matrix[n][m][l] = np.ceil(distance_matrix[n][m] / ship_models[l][4])
                    max_travel_time_matrix[n][m][l] = np.ceil(distance_matrix[n][m] / ship_models[l][3])
    return min_travel_time_matrix, max_travel_time_matrix


def get_cargo_spawn_pattern(ports, cargo_spawn_pattern_history):
    cargo_spawn_pattern_history = cargo_spawn_pattern_history[1:, :, :]
    cargo_spawn_pattern = np.zeros((len(ports), len(ports)))
    for n in range(len(ports)):
        for m in range(len(ports)):
            if n != m:
                cargo_spawn_pattern[n][m] = np.random.randint(0, 11)
    temp_lst = np.reshape(cargo_spawn_pattern, (1, len(ports), len(ports)))
    cargo_spawn_pattern_history = np.append(cargo_spawn_pattern_history, temp_lst, axis=0)
    return cargo_spawn_pattern, cargo_spawn_pattern_history


def get_cargo_spawn(time, cargo_spawn_pattern, num_of_cargo_generated):
    cargo_spawn_pattern = np.copy(cargo_spawn_pattern)
    num_of_spawned_cargos = int(np.sum(cargo_spawn_pattern))
    cargo_spawn_pattern_direct = np.zeros((len(ports), len(ports)))
    cargo_spawn_pattern_routed = np.zeros((len(ports), len(ports)))
    spawned_cargos = np.zeros((num_of_spawned_cargos, 12))
    cargo_counter = 0
    while np.amax(cargo_spawn_pattern) >= 1:
        for i in range(len(cargo_spawn_pattern)):
            for j in range(len(cargo_spawn_pattern)):
                if i != j:
                    while cargo_spawn_pattern[i][j] >= 1:
                        spawned_cargos[cargo_counter][0] = num_of_cargo_generated
                        spawned_cargos[cargo_counter][1] = i
                        spawned_cargos[cargo_counter][2] = j
                        spawned_cargos[cargo_counter][3] = time
                        spawned_cargos[cargo_counter][4] = 1
                        spawned_cargos[cargo_counter][5] = np.random.choice(2, p=[1 - container_routing_ratio, container_routing_ratio])
                        if spawned_cargos[cargo_counter][5] == 0:
                            cargo_spawn_pattern_direct[i][j] += 1
                        else:
                            cargo_spawn_pattern_routed[i][j] += 1

                        spawned_cargos[cargo_counter][6] = time
                        spawned_cargos[cargo_counter][7] = np.inf
                        spawned_cargos[cargo_counter][8] = spawned_cargos[cargo_counter][1]
                        spawned_cargos[cargo_counter][9] = np.inf
                        cargo_spawn_pattern[i][j] -= 1
                        cargo_counter += 1
                        num_of_cargo_generated += 1
    return spawned_cargos, num_of_cargo_generated, cargo_spawn_pattern_direct, cargo_spawn_pattern_routed


def get_theoretical_unit_cost_lower_bound(ship_models):
    constant_coefficient = 1 / 110000
    fuel_efficiency = np.full((1, len(ship_models)), np.inf)
    for n in range(len(ship_models)):
        fuel_efficiency[0][n] = constant_coefficient * ship_models[n][3] ** 3 * (
                    ship_models[n][2] + ship_models[n][1] * 1 / 3) ** (2 / 3) / (
                                            ship_models[n][1] * ship_models[n][3] * 24)
    unit_cost_lower_bound = np.min(fuel_efficiency)
    return unit_cost_lower_bound


def efficiency_benchmark(ship_models, ships, cargos, distance_matrix, time, unit_cost_lower_bound,
                         actual_cumulative_fuel_cost, theoretical_cumulative_fuel_cost, num_of_cargo_delivered,
                         throughput, shipping_time_span, unit_fuel_cost, average_shipping_time_span,
                         unit_shipping_time_span, average_fuel_cost):
    constant_coefficient = 1 / 110000
    number_of_usable_ships = np.zeros((len(ship_models), len(ports)))
    global_queue_size = np.zeros((len(ports), len(ports)))
    for n in range(len(cargos)):
        cargo_id = int(cargos[n][0])
        origin = int(cargos[n][1])
        destination = int(cargos[n][2])
        for m in range(len(globals()[f'cargo_event_log_{cargo_id}'])):
            if globals()[f'cargo_event_log_{cargo_id}'].size != 0:
                if eval(f'cargo_event_log_{cargo_id}[{m}][0]') == time and cargos[n][2] == eval(
                        f'cargo_event_log_{cargo_id}[{m}][2]') and (
                        eval(f'cargo_event_log_{cargo_id}[{m}][1]') == 1 or eval(
                        f'cargo_event_log_{cargo_id}[{m}][1]') == 3):
                    print('cargo found for benchmark', globals()[f'cargo_event_log_{cargo_id}'])
                    num_of_cargo_delivered += 1
                    throughput += distance_matrix[origin][destination]
                    shipping_time_span += time - cargos[n][3]
                    theoretical_cumulative_fuel_cost += distance_matrix[origin][destination] * unit_cost_lower_bound
    try:
        average_shipping_time_span = shipping_time_span / num_of_cargo_delivered
    except ZeroDivisionError:
        print('error detected in calculating average_shipping_time_span')
    try:
        unit_shipping_time_span = shipping_time_span / throughput
    except ZeroDivisionError:
        print('error detected in calculating unit_shipping_time_span')

    for i in range(len(ships)):
        ship_id = int(ships[i][0])
        ship_model = int(ships[i][1])
        origin = int(ships[i][3])
        destination = int(ships[i][4])
        for j in range(len(globals()[f'ship_event_log_{ship_id}'])):
            if globals()[f'ship_event_log_{ship_id}'].size != 0:
                if eval(f'ship_event_log_{ship_id}[{j}][0]') == time and (
                        eval(f'ship_event_log_{ship_id}[{j}][1]') == 1 or eval(
                        f'ship_event_log_{ship_id}[{j}][1]') == 3):
                    actual_cumulative_fuel_cost += constant_coefficient * max(ship_models[ship_model][3], (
                                distance_matrix[origin][destination] / (
                                    eval(f'ship_event_log_{ship_id}[{j}][0]') - eval(
                                f'ship_event_log_{ship_id}[{j - 1}][0]')))) ** 3 * (ship_models[ship_model][2] + eval(
                        f'ship_event_log_{ship_id}[{j}][3]') * 1 / 3) ** (2 / 3) / 24 / max(ship_models[ship_model][3],
                                                                                            (distance_matrix[origin][
                                                                                                 destination] / (eval(
                                                                                                f'ship_event_log_{ship_id}[{j}][0]') - eval(
                                                                                                f'ship_event_log_{ship_id}[{j - 1}][0]')))) * \
                                                   distance_matrix[origin][destination]
    try:
        average_fuel_cost = actual_cumulative_fuel_cost / num_of_cargo_delivered
    except ZeroDivisionError:
        print('error detected in calculating average_fuel_cost')
    try:
        unit_fuel_cost = actual_cumulative_fuel_cost / throughput
    except ZeroDivisionError:
        print('error detected in calculating unit_fuel_cost')
    for x in range(len(ships)):
        if ships[x][3] == ships[x][4] and ships[x][2] == 1:
            ship_model = int(ships[x][1])
            location = int(ships[x][3])
            number_of_usable_ships[ship_model][location] += 1
    for y in range(len(cargos)):
        if cargos[y][4] == 1 and cargos[y][9] != np.inf:
            origin = int(cargos[y][8])
            destination = int(cargos[y][9])
            global_queue_size[origin][destination] += 1
    return num_of_cargo_delivered, throughput, theoretical_cumulative_fuel_cost, actual_cumulative_fuel_cost, average_fuel_cost, unit_fuel_cost, shipping_time_span, average_shipping_time_span, unit_shipping_time_span, number_of_usable_ships, global_queue_size


def ship_and_cargo_status_update(ships, cargos, time):
    for n in range(len(ships)):
        for m in range(len(globals()[f'ship_event_log_{n}'])):
            if eval(f'ship_event_log_{n}[{m}][0]') == time and (
                    eval(f'ship_event_log_{n}[{m}][1]') == 1 or eval(f'ship_event_log_{n}[{m}][1]') == 3):
                ship_id = int(ships[n][0])
                ships[n][2] = 1
                ships[n][3] = ships[n][4]
                ships[n][5] = 0
                ships[n][6] = np.inf
                ships[n][7] = 0
                print('ship', ship_id, 'arrived. showing ship\'s updated information and event log\n', ships[n], '\n',
                      globals()[f'ship_event_log_{ship_id}'])
                break
    for i in range(len(cargos)):
        cargo_id = int(cargos[i][0])
        for j in range(len(globals()[f'cargo_event_log_{cargo_id}'])):
            if globals()[f'cargo_event_log_{cargo_id}'].size != 0:
                if eval(f'cargo_event_log_{cargo_id}[{j}][0]') == time and (
                        eval(f'cargo_event_log_{cargo_id}[{j}][1]') == 1 or eval(
                        f'cargo_event_log_{cargo_id}[{j}][1]') == 3):
                    if cargos[i][9] == cargos[i][2]:
                        cargos[i][4] = 3
                        cargos[i][8] = cargos[i][9]
                    else:
                        cargos[i][4] = 1
                        cargos[i][8] = cargos[i][9]
                        cargos[i][9] = np.inf
                        cargos[i][6] = time
                        cargos[i][7] = np.inf
                    cargos[i][10] = 0
                    cargos[i][11] = np.inf
                    print('cargo', cargo_id, 'arrived. showing cargo\'s updated information and event log',
                          cargos[i], '\n', globals()[f'cargo_event_log_{cargo_id}'])
                    break
    return ships, cargos


def get_cargo_event_log(num_of_cargo_generated_by_last_epoch, num_of_cargo_generated):
    for n in range(num_of_cargo_generated_by_last_epoch, num_of_cargo_generated):
        globals()[f'cargo_event_log_{n}'] = np.zeros((0, 4))


def get_num_of_ship(ships, ship_models, location):
    num_of_ship = np.zeros((1, len(ship_models)))
    for n in range(len(ships)):
        if ships[n][2] == 1 and ships[n][3] == ships[n][4] == location:
            num_of_ship[0][int(ships[n][1])] += 1
    return num_of_ship


# def get_theoretical_moving_average(cargo_spawn_pattern_history, origin, destination, time, moving_average_length):
#     if time >= moving_average_length - 1:
#         denominator = moving_average_length
#     elif time < moving_average_length - 1:
#         denominator = t + 1
#     moving_average = np.sum(cargo_spawn_pattern_history[:,origin,destination]) / denominator
#     return moving_average

def get_moving_average_matrix(cargo_spawn_pattern_history, ports, moving_average_length, time):
    moving_average_matrix = np.zeros((len(ports), len(ports)))
    for n in range(len(ports)):
        for m in range(len(ports)):
            if n != m:
                if time >= moving_average_length - 1:
                    denominator = moving_average_length
                elif time < moving_average_length - 1:
                    denominator = time + 1
                moving_average_matrix[n][m] = np.sum(cargo_spawn_pattern_history[:, n, m]) / denominator
    return moving_average_matrix


def get_demand_forecasting(cargo_spawn_pattern_history, ports, moving_average_length):
    demand_forecasting_matrix = np.zeros((len(ports), len(ports)))
    for n in range(len(ports)):
        for m in range(len(ports)):
            if n != m:
                x = np.array(list(range(moving_average_length))).reshape((-1, 1))
                y = cargo_spawn_pattern_history[:, n, m]
                model = LinearRegression().fit(x, y)
                temp_demand_forecasting = model.predict([[moving_average_length]])
                ### change demand to 0 of predicted to be negative
                if temp_demand_forecasting < 0:
                    temp_demand_forecasting = 0
                demand_forecasting_matrix[n, m] = temp_demand_forecasting
    return demand_forecasting_matrix


def get_seperated_cargo_spawn_pattern_history(cargo_spawn_pattern_direct_history, cargo_spawn_pattern_routed_history,
                                              cargo_spawn_pattern_direct, cargo_spawn_pattern_routed, ports):
    cargo_spawn_pattern_direct_history = cargo_spawn_pattern_direct_history[1:, :, :]
    cargo_spawn_pattern_routed_history = cargo_spawn_pattern_routed_history[1:, :, :]
    cargo_spawn_pattern_direct = np.reshape(cargo_spawn_pattern_direct, (1, len(ports), len(ports)))
    cargo_spawn_pattern_routed = np.reshape(cargo_spawn_pattern_routed, (1, len(ports), len(ports)))
    cargo_spawn_pattern_direct_history = np.append(cargo_spawn_pattern_direct_history, cargo_spawn_pattern_direct,
                                                   axis=0)
    cargo_spawn_pattern_routed_history = np.append(cargo_spawn_pattern_routed_history, cargo_spawn_pattern_routed,
                                                   axis=0)
    return cargo_spawn_pattern_direct_history, cargo_spawn_pattern_routed_history


def cargoflow_index_converter(index, ports):
    num_of_port = len(ports)
    origin = int(np.ceil((1 + index) / (num_of_port - 1) - 1))
    if index % (num_of_port - 1) < origin:
        destination = index % (num_of_port - 1)
    else:
        destination = index % (num_of_port - 1) + 1
    return origin, destination


# def get_delivery_time_span(origin, destination, moving_average, max_travel_time_matrix, ship_models):
#     if moving_average != 0:
#         delivery_time_span = np.ceil(max_travel_time_matrix[origin][destination][len(ship_models)-1] + ship_models[len(ship_models)-1][1] / moving_average)
#     else:
#         delivery_time_span = np.inf
#     return delivery_time_span

# def get_cargo_deadline(cargos, delivery_time_span, origin, destination):
#     for n in range(len(cargos)):
#         if cargos[n][8] == origin and cargos[n][9] == destination and cargos[n][7] == np.inf:
#             cargos[n][7] = cargos[n][6] + delivery_time_span
#     return cargos

def get_cargo_deadline(cargos, cost_matrix_actual, time):
    for n in range(len(cargos)):
        if cargos[n][7] == np.inf:
            origin = int(cargos[n][8])
            destination = int(cargos[n][9])
            cargos[n][7] = cost_matrix_actual[origin][destination] + time
    return cargos


def get_archived_cargo(cargos, archived_cargos):
    if cargos.size != 0:
        last_item_not_checked = True
        while last_item_not_checked == True:
            for n in range(len(cargos)):
                if cargos[n][8] == cargos[n][9] == cargos[n][2]:
                    temp_lst = cargos[n].reshape(1, 12)
                    archived_cargos = np.append(archived_cargos, temp_lst, axis=0)
                    cargos = np.delete(cargos, n, 0)
                    if n == len(cargos):
                        last_item_not_checked = False
                    break
                if n == len(cargos) - 1:
                    last_item_not_checked = False
    else:
        return cargos, archived_cargos
    return cargos, archived_cargos


def get_cargo_routing_and_cargo_spawn_pattern_actual(ports, cargos, routing_table):
    cargo_spawn_pattern_actual = np.zeros((len(ports), len(ports)))
    for n in range(len(cargos)):
        if cargos[n][5] == 0 and cargos[n][9] == np.inf:
            origin = int(cargos[n][1])
            destination = int(cargos[n][2])
            cargos[n][9] = cargos[n][2]
            cargo_spawn_pattern_actual[origin][destination] += 1
        elif cargos[n][5] == 1 and cargos[n][9] == np.inf:
            origin = int(cargos[n][8])
            destination = int(cargos[n][2])
            sector_destination = routing_table[origin][destination][0]
            if sector_destination != np.inf:
                cargos[n][9] = sector_destination
            else:
                cargos[n][9] = cargos[n][2]
            cargo_spawn_pattern_actual[origin][int(cargos[n][9])] += 1
    return cargos, cargo_spawn_pattern_actual


def get_cargo_routing_and_cargo_spawn_pattern_actual_history(ports, cargo_spawn_pattern_actual,
                                                             cargo_spawn_pattern_actual_history):
    cargo_spawn_pattern_actual_history = cargo_spawn_pattern_actual_history[1:, :, :]
    cargo_spawn_pattern_actual = np.reshape(cargo_spawn_pattern_actual, (1, len(ports), len(ports)))
    cargo_spawn_pattern_actual_history = np.append(cargo_spawn_pattern_actual_history, cargo_spawn_pattern_actual,
                                                   axis=0)
    return cargo_spawn_pattern_actual_history


# def get_queue_size(cargos, origin, destination):
#     queue_size = 0
#     if cargos.size != 0:
#         for n in range(len(cargos)):
#             if cargos[n][4] == 1 and cargos[n][8] == origin and cargos[n][9] == destination:
#                 queue_size += 1
#     return queue_size

# def get_queue_deadline(cargos, origin, destination):
#     queue_deadline = np.inf
#     for n in range(len(cargos)):
#         if cargos[n][4] == 1 and cargos[n][8] == origin and cargos[n][9] == destination and cargos[n][7] < queue_deadline:
#             queue_deadline = cargos[n][7]
#     return queue_deadline

# def get_unit_cost(origin, destination, queue_size, queue_deadline, distance_matrix):
#     constant_coefficient = 1/110000
#     unit_cost = np.zeros((1,3))
#     if queue_size != 0:
#         for n in range(len(ship_models)):
#             if queue_deadline - t >= max_travel_time_matrix[origin][destination][n]:
#                 unit_cost[0][n] = constant_coefficient * ship_models[n][3] ** 3 * (
#                             ship_models[n][2] + min(queue_size, ship_models[n][1]) * (1 / 3)) ** (2 / 3) / min(
#                     queue_size, ship_models[n][1]) / ship_models[n][3] / 24
#             elif max_travel_time_matrix[origin][destination][n] > queue_deadline - t >= min_travel_time_matrix[origin][destination][n]:
#                 unit_cost[0][n] = constant_coefficient * (distance_matrix[origin][destination] / (queue_deadline - t)) ** 3 * (
#                             ship_models[n][2] + min(queue_size, ship_models[n][1]) * (1 / 3)) ** (2 / 3) / min(
#                     queue_size, ship_models[n][1]) / (distance_matrix[origin][destination] / (queue_deadline - t)) / 24
#             else:
#                 unit_cost[0][n] = constant_coefficient * ship_models[n][4] ** 3 * (
#                         ship_models[n][2] + min(queue_size, ship_models[n][1]) * (1 / 3)) ** (2 / 3) / min(
#                     queue_size, ship_models[n][1]) / ship_models[n][4] / 24
#     else:
#         unit_cost[:,:] = np.inf
#     return unit_cost

# def get_future_unit_cost(origin, destination, queue_size, queue_deadline, moving_average, distance_matrix, time, unit_cost,delivery_time_limit):
#     constant_coefficient = 1 / 110000
#     future_unit_cost = unit_cost
#     queue_deadline = min(queue_deadline, time + delivery_time_limit + 1)
#     for n in range(len(ship_models)):
#         counter = 0
#         future_time = time
#         while queue_deadline - future_time > min_travel_time_matrix[origin][destination][n]:
#             counter += 1
#             future_time += 1
#             future_queue_size = queue_size + moving_average * counter
#             if future_queue_size == 0:
#                 temp = np.inf
#                 break
#             elif queue_deadline - future_time >= max_travel_time_matrix[origin][destination][n]:
#                 temp = constant_coefficient * ship_models[n][3] ** 3 * (
#                             ship_models[n][2] + min(ship_models[n][1], future_queue_size) * (1 / 3)) ** (2 / 3) / min(
#                     ship_models[n][1], future_queue_size) / ship_models[n][3] / 24
#             else:
#                 temp = constant_coefficient * (distance_matrix[origin][destination] / (queue_deadline - future_time)) ** 3 * (
#                             ship_models[n][2] + min(ship_models[n][1], future_queue_size) * (1 / 3)) ** (2 / 3) / min(ship_models[n][1],
#                                                                                                   future_queue_size) / (distance_matrix[origin][destination] / (queue_deadline - future_time)) / 24
#             if future_unit_cost.shape[0] == counter:
#                 future_unit_cost = np.append(future_unit_cost,[[np.inf, np.inf, np.inf]], axis = 0)
#             future_unit_cost[counter][n] = temp
#     return future_unit_cost

def get_model_of_departure_ship_and_time_to_arrive(unit_cost_enumeration_matrix, num_of_ship, origin, destination,
                                                   max_travel_time_matrix):
    unit_cost = unit_cost_enumeration_matrix[:, 0, :]
    ship_selection = False
    while ship_selection == False:
        candidate_ship_model = np.where(unit_cost == np.amin(unit_cost))[0][0]
        print('candidate ship model', candidate_ship_model)
        if num_of_ship[0][candidate_ship_model] >= 1:
            model_of_departure_ship = candidate_ship_model
            speed_level = np.where(unit_cost == np.amin(unit_cost))[1][0]
            time_to_arrive = max_travel_time_matrix[origin][destination][model_of_departure_ship] - speed_level
            ship_selection = True
        else:
            print('ship model', candidate_ship_model, 'is not available. checking other options')
            unit_cost[candidate_ship_model] = np.inf
    return model_of_departure_ship, time_to_arrive


def departure_scheduling(cargos, ships, ship_models, num_of_ship, model_of_departure_ship, time_to_arrive, origin,
                         destination, time, cargoflow, counter):
    print('start searching candidate ship of model', model_of_departure_ship)
    for n in range(len(ships)):
        if ships[n][1] == model_of_departure_ship and ships[n][2] == 1 and ships[n][3] == ships[n][4] == origin:
            ship_id = int(ships[n][0])
            print('ship', ship_id, 'will be used. updating information and event log.')
            ships[n][2] = 2
            ships[n][4] = destination
            ships[n][6] = time + time_to_arrive
            ship_temp_lst_1 = [[time, 0, origin, 0]]
            ship_temp_lst_2 = [[time + time_to_arrive, 1, destination, 0]]
            num_of_ship[0][model_of_departure_ship] -= 1
            available_capacity = ship_models[model_of_departure_ship][1]
            break
    print('start searching cargos to be transported on this sector')
    cargos = cargos[cargos[:, 7].argsort()]
    num_of_cargo_searched = 0
    while available_capacity >= 1 and num_of_cargo_searched < len(cargos):
        print(available_capacity,
              'unit of capacity still remain available, and potential cargo exists. looking for next suitable cargo')
        for m in range(num_of_cargo_searched, len(cargos)):
            num_of_cargo_searched = m + 1
            if cargos[m][4] == 1 and cargos[m][8] == origin and cargos[m][9] == destination:
                cargo_id = int(cargos[m][0])
                print(cargo_id, 'will be transported. updating information and event log')
                cargos[m][4] = 2
                cargos[m][11] = time_to_arrive
                cargo_temp_lst_1 = [[time, 0, origin, ship_id]]
                cargo_temp_lst_2 = [[time + time_to_arrive, 1, destination, ship_id]]
                globals()[f'cargo_event_log_{cargo_id}'] = np.append(globals()[f'cargo_event_log_{cargo_id}'],
                                                                     cargo_temp_lst_1, axis=0)
                globals()[f'cargo_event_log_{cargo_id}'] = np.append(globals()[f'cargo_event_log_{cargo_id}'],
                                                                     cargo_temp_lst_2, axis=0)
                available_capacity -= 1
                if available_capacity == 0 and num_of_cargo_searched < len(cargos):
                    print('ship capacity used up, but more suitable cargos might exist')
                elif available_capacity >= 1 and num_of_cargo_searched == len(cargos):
                    print('no potential suitable cargos, but ship capacity remains available')
                elif available_capacity == 0 and num_of_cargo_searched == len(cargos):
                    print('perfect. last cargo in the list occupied the last unit of ship capacity')
                break
            if available_capacity == 0 and num_of_cargo_searched < len(cargos):
                print('ship capacity used up, but more suitable cargos might exist')
            elif available_capacity >= 1 and num_of_cargo_searched == len(cargos):
                print('no potential suitable cargos, but ship capacity remains available')
            elif available_capacity == 0 and num_of_cargo_searched == len(cargos):
                print('perfect. last cargo in the list occupied the last unit of ship capacity')
    ship_temp_lst_1[0][3] = ship_models[model_of_departure_ship][1] - available_capacity
    ship_temp_lst_2[0][3] = ship_models[model_of_departure_ship][1] - available_capacity
    globals()[f'ship_event_log_{ship_id}'] = np.append(globals()[f'ship_event_log_{ship_id}'],
                                                       ship_temp_lst_1, axis=0)
    globals()[f'ship_event_log_{ship_id}'] = np.append(globals()[f'ship_event_log_{ship_id}'],
                                                       ship_temp_lst_2, axis=0)
    return cargos, ships, num_of_ship


# def get_port_moving_average(cargos, archived_cargos, time):
#     counter_1 = np.zeros((1,len(ports)))
#     counter_2 = np.zeros((1, len(ports)))
#     for n in range(len(cargos)):
#         origin = int(cargos[n][1])
#         if cargos[n][3] > time - 24 * 7:
#             counter_1[0][origin] += 1
#     for m in range(len(archived_cargos)):
#         origin = int(archived_cargos[m][1])
#         if archived_cargos[m][5] == 0 and archived_cargos[m][3] > time - 24 * 7:
#             counter_2[0][origin] += 1
#     if time >= 24 * 7:
#         port_moving_average = (counter_2 + counter_1) / (24 * 7)
#     else:
#         port_moving_average = (counter_2 + counter_1) / (time + 1)
#     return port_moving_average

def get_port_moving_average(moving_average_actual_matrix):
    port_moving_average = np.zeros((1, len(moving_average_actual_matrix)))
    for n in range(len(moving_average_actual_matrix)):
        port_moving_average[0][n] = np.sum(moving_average_actual_matrix[n])
    return port_moving_average


def get_ship_buffer_and_liquidity(ships, ports, target_ship_model):
    ship_buffer_and_liquidity = np.zeros((2, len(ports)))
    for n in range(len(ships)):
        if ships[n][1] == target_ship_model:
            destination = int(ships[n][4])
            ship_buffer_and_liquidity[0][destination] += 1
            if ships[n][3] == ships[n][4] and ships[n][2] == 1:
                ship_buffer_and_liquidity[1][destination] += 1
    return ship_buffer_and_liquidity


def get_target_buffer_size(ship_buffer_and_liquidity, port_moving_average, ports):
    target_buffer_size = np.zeros((1, len(ports)))
    for n in range(len(ports)):
        target_buffer_size[0][n] = np.sum(ship_buffer_and_liquidity[0]) * port_moving_average[0][n] / np.sum(
            port_moving_average)
    return target_buffer_size


def get_ship_supply_and_demand(ship_buffer_and_liquidity, target_buffer_size, ports):
    ship_supply_and_demand = np.zeros((1, len(ports)))
    donor_availability_lst = np.ones((1, len(ports)))
    acceptor_availability_lst = np.ones((1, len(ports)))
    critical_level = np.zeros((1, len(ports)))
    while np.sum(donor_availability_lst) >= 1 and np.sum(acceptor_availability_lst) >= 1:
        for n in range(len(ports)):
            if donor_availability_lst[0][n] >= 1 or acceptor_availability_lst[0][n] >= 1:
                critical_level[0][n] = (target_buffer_size[0][n] - ship_buffer_and_liquidity[0][n]) / \
                                       target_buffer_size[0][n]
                if critical_level[0][n] > 0 and target_buffer_size[0][n] - ship_buffer_and_liquidity[0][n] >= 1:
                    donor_availability_lst[0][n] = 0
                elif critical_level[0][n] < 0 and ship_buffer_and_liquidity[0][n] - target_buffer_size[0][n] >= 1 and \
                        ship_buffer_and_liquidity[1][n] >= 2:
                    acceptor_availability_lst[0][n] = 0
                else:
                    donor_availability_lst[0][n] = 0
                    acceptor_availability_lst[0][n] = 0
                    critical_level[0][n] = 0
            else:
                critical_level[0][n] = 0
        if np.amax(critical_level) > 0 and np.amin(critical_level) < 0:
            acceptor_index = np.where(critical_level == np.amax(critical_level))
            donor_index = np.where(critical_level == np.amin(critical_level))
            ship_supply_and_demand[0][acceptor_index[1][0]] += 1
            ship_buffer_and_liquidity[0][acceptor_index[1][0]] += 1
            ship_supply_and_demand[0][donor_index[1][0]] -= 1
            ship_buffer_and_liquidity[0][donor_index[1][0]] -= 1
            ship_buffer_and_liquidity[1][donor_index[1][0]] -= 1
    return ship_supply_and_demand


def excute_reposition_plan(reposition_plan, target_ship_model, ship_models, ports, ships, cargos,
                           max_travel_time_matrix, time):
    print('global reposition starts')
    while np.amax(reposition_plan) >= 1:
        print('demand exist, start looking for next ship for reposition')
        for i in range(len(ports)):
            for j in range(len(ports)):
                if i != j:
                    while reposition_plan[i][j] >= 1:
                        print('-1')
                        for k in range(len(ships)):
                            if ships[k][1] == target_ship_model and ships[k][2] == 1 and ships[k][3] == ships[k][
                                4] == i:
                                ship_id = int(ships[k][0])
                                print('ship', ship_id,
                                      'selected to be used for reposition. updating information and event log')
                                ships[k][2] = 2
                                ships[k][4] = j
                                ships[k][6] = max_travel_time_matrix[i][j][target_ship_model]
                                ship_temp_lst_1 = [[time, 2, i, 0]]
                                ship_temp_lst_2 = [[time + max_travel_time_matrix[i][j][target_ship_model], 3, j, 0]]
                                available_capacity = ship_models[target_ship_model][1]
                                reposition_plan[i][j] -= 1
                                print('looking for suitable cargos along with repositioned ship')
                                cargos = cargos[cargos[:, 7].argsort()]
                                if shipping_during_rebalance == True:
                                    cargo_search = True
                                else:
                                    cargo_search = False
                                while cargo_search == True:
                                    print('0')
                                    for l in range(len(cargos)):
                                        print('1')
                                        if available_capacity >= 1 and cargos[l][8] == i and cargos[l][9] == j and \
                                                cargos[l][4] == 1 and cargos[l][7] >= time + \
                                                max_travel_time_matrix[i][j][target_ship_model]:
                                            print('2')
                                            cargo_id = int(cargos[l][0])
                                            print('cargo', cargo_id, 'selected to be transported by repositioned ship',
                                                  ship_id, 'updating cargo information and event log')
                                            cargos[l][4] = 2
                                            cargos[l][11] = max_travel_time_matrix[i][j][target_ship_model]
                                            available_capacity -= 1
                                            cargo_temp_lst_1 = [[time, 2, i, ship_id]]
                                            cargo_temp_lst_2 = [
                                                [time + max_travel_time_matrix[i][j][target_ship_model], 3, j, ship_id]]
                                            globals()[f'cargo_event_log_{cargo_id}'] = np.append(
                                                globals()[f'cargo_event_log_{cargo_id}'], cargo_temp_lst_1, axis=0)
                                            globals()[f'cargo_event_log_{cargo_id}'] = np.append(
                                                globals()[f'cargo_event_log_{cargo_id}'], cargo_temp_lst_2, axis=0)
                                        elif available_capacity == 0:
                                            print('3')
                                            print('capacity of ship', ship_id, 'used up. stopping cargo search')
                                            cargo_search = False
                                            break
                                    if l == len(cargos) - 1:
                                        print('4')
                                        print('all cargos checked. no potential suitable cargo exists')
                                        cargo_search = False
                                ship_temp_lst_1[0][3] = ship_models[target_ship_model][1] - available_capacity
                                ship_temp_lst_2[0][3] = ship_models[target_ship_model][1] - available_capacity
                                globals()[f'ship_event_log_{ship_id}'] = np.append(
                                    globals()[f'ship_event_log_{ship_id}'], ship_temp_lst_1, axis=0)
                                globals()[f'ship_event_log_{ship_id}'] = np.append(
                                    globals()[f'ship_event_log_{ship_id}'], ship_temp_lst_2, axis=0)
                                break
    print('global reposition ends')
    return ships, cargos


def get_donor_and_acceptor(ship_buffer_and_liquidity, target_buffer_size, ports):
    critical_level = np.zeros((1, len(ports)))
    acceptor_index = None
    donor_index = None
    for n in range(len(ports)):
        critical_level[0][n] = (target_buffer_size[0][n] - ship_buffer_and_liquidity[0][n]) / target_buffer_size[0][n]
    if np.amax(critical_level) >= 0.5:
        acceptor_index = np.where(critical_level == np.amax(critical_level))[1][0]
        while np.amin(critical_level) < 0:
            potential_donor = np.where(critical_level == np.amin(critical_level))[1][0]
            if critical_level[0][potential_donor] < 0 and ship_buffer_and_liquidity[0][potential_donor] - \
                    target_buffer_size[0][potential_donor] >= 1 and ship_buffer_and_liquidity[1][potential_donor] >= 2:
                donor_index = potential_donor
                print('acceptor', acceptor_index, 'and donor', donor_index, 'found')
                return acceptor_index, donor_index
            else:
                critical_level[0][potential_donor] = 0
        if np.amin(critical_level) >= 0:
            print('acceptor', acceptor_index, 'found, but no available donor')
            return acceptor_index, donor_index
    else:
        print('no acceptor or donor found')
        return acceptor_index, donor_index


def execute_paired_reposition(counter, acceptor_index, donor_index, cargos, ships, ship_models, target_ship_model,
                              max_travel_time_matrix, time):
    print('paired reposition starts')
    print('searching for a ship for reposition')
    for n in range(len(ships)):
        if ships[n][1] == target_ship_model and ships[n][2] == 1 and ships[n][3] == ships[n][4] == donor_index:
            ship_id = int(ships[n][0])
            print('ship', ship_id, 'selected to be repositioned. updating ship information and event log')
            ships[n][2] = 2
            ships[n][4] = acceptor_index
            ships[n][6] = max_travel_time_matrix[donor_index][acceptor_index][target_ship_model]
            available_capacity = ship_models[target_ship_model][1]
            ship_temp_lst_1 = [[time, 2, donor_index, 0]]
            ship_temp_lst_2 = [
                [time + max_travel_time_matrix[donor_index][acceptor_index][target_ship_model], 3, acceptor_index, 0]]
            print('cargo search starts')
            cargos = cargos[cargos[:, 7].argsort()]
            if shipping_during_rebalance == True:
                cargo_search = True
            else:
                cargo_search = False
            while cargo_search == True:
                for m in range(len(cargos)):
                    if available_capacity >= 1 and cargos[m][8] == donor_index and cargos[m][9] == acceptor_index and \
                            cargos[m][4] == 1 and cargos[m][7] >= time + \
                            max_travel_time_matrix[donor_index][acceptor_index][target_ship_model]:
                        cargo_id = int(cargos[m][0])
                        print('cargo', cargo_id, 'selected to be transported by repositioned ship', ship_id,
                              'updating cargo information and event log')
                        cargos[m][4] = 2
                        cargos[m][11] = max_travel_time_matrix[donor_index][acceptor_index][target_ship_model]
                        available_capacity -= 1
                        cargo_temp_lst_1 = [[time, 2, donor_index, ship_id]]
                        cargo_temp_lst_2 = [
                            [time + max_travel_time_matrix[donor_index][acceptor_index][target_ship_model], 3,
                             acceptor_index, ship_id]]
                        globals()[f'cargo_event_log_{cargo_id}'] = np.append(globals()[f'cargo_event_log_{cargo_id}'],
                                                                             cargo_temp_lst_1, axis=0)
                        globals()[f'cargo_event_log_{cargo_id}'] = np.append(globals()[f'cargo_event_log_{cargo_id}'],
                                                                             cargo_temp_lst_2, axis=0)
                    elif available_capacity == 0:
                        print('capacity of ship', ship_id, 'used up. stopping cargo search')
                        cargo_search = False
                        break
                if m == len(cargos) - 1:
                    print('all cargos checked. no potential suitable cargo exists')
                    cargo_search = False
            ship_temp_lst_1[0][3] = ship_models[target_ship_model][1] - available_capacity
            ship_temp_lst_2[0][3] = ship_models[target_ship_model][1] - available_capacity
            globals()[f'ship_event_log_{ship_id}'] = np.append(
                globals()[f'ship_event_log_{ship_id}'], ship_temp_lst_1, axis=0)
            globals()[f'ship_event_log_{ship_id}'] = np.append(
                globals()[f'ship_event_log_{ship_id}'], ship_temp_lst_2, axis=0)
            break
    print('paired reposition ends')
    return ships, cargos


def get_cargo_in_queue(cargos, origin, destination):
    cargo_in_queue = np.zeros((0, 12))
    for n in range(len(cargos)):
        if cargos[n][8] == origin and cargos[n][9] == destination and cargos[n][4] == 1:
            temp_lst = cargos[n].reshape(1, 12)
            cargo_in_queue = np.append(cargo_in_queue, temp_lst, axis=0)
    queue_size = len(cargo_in_queue)
    return queue_size, cargo_in_queue


def get_cargo_queue_alpha(cargos, ship_models, time, delivery_time_limit, moving_average):
    max_capacity = np.amax(ship_models, axis=0)[1]
    row_counter = len(cargos)
    counter = 0
    queue_size = len(cargos)

    dummy_lst = np.zeros((1, 12))
    if len(cargos) <= max_capacity:
        for m in range(int(max_capacity) - len(cargos)):
            cargos = np.append(cargos, dummy_lst, axis=0)

    for n in range(row_counter, int(max_capacity)):
        counter += 1
        cargos[n][0] = np.inf
        if moving_average != 0:
            if (n + 1 - queue_size) % moving_average >= 0.5:
                next_epoch_indicator = 1
            else:
                next_epoch_indicator = 0
            cargos[n][7] = time + delivery_time_limit + (np.floor(counter / moving_average) + next_epoch_indicator)
        else:
            cargos[n][7] = np.inf
    cargo_queue_alpha = cargos
    return cargo_queue_alpha


def get_unit_cost_enumeration_matrix(queue_size, moving_average, cargos, time, distance_matrix, ship_models,
                                     min_travel_time_matrix, max_travel_time_matrix, origin, destination):
    constant_coefficient = 1 / 110000
    max_capacity = np.amax(ship_models, axis=0)[1]
    print('max capacity', max_capacity)
    first_dimension_size = len(ship_models)
    distance = distance_matrix[origin][destination]
    cargo_weight = 1 / 3
    penalty = 100
    cargos = cargos[np.argsort(cargos[:, 7])]
    print('sorted cargos', cargos)
    if max_capacity >= queue_size:
        if moving_average != 0:
            second_dimension_size = np.ceil((max_capacity - queue_size) / moving_average) + 1
        else:
            unit_cost_enumeration_matrix = np.full(
                (int(first_dimension_size), 1, 1), np.inf)
            return unit_cost_enumeration_matrix
    else:
        second_dimension_size = 1
    travel_duration_range = np.zeros((len(ship_models)))
    for n in range(len(ship_models)):
        travel_duration_range[n] = int(
            max_travel_time_matrix[origin][destination][n] - min_travel_time_matrix[origin][destination][n] + 1)
    third_dimension_size = np.amax(travel_duration_range, axis=0)
    print('max travel duration range', third_dimension_size)
    unit_cost_enumeration_matrix = np.full(
        (int(first_dimension_size), int(second_dimension_size), int(third_dimension_size)), np.inf)
    for i in range(int(first_dimension_size)):
        exhaustive_t = False
        for j in range(int(second_dimension_size)):
            if exhaustive_t == True:
                break
            elif ship_models[i][1] <= moving_average * j + queue_size:
                exhaustive_t = True
            for k in range(int(third_dimension_size)):
                if k > travel_duration_range[i]:
                    break
                num_of_cargo_loaded = min(ship_models[i][1], queue_size + moving_average * j)
                quantity_of_lateness = 0
                time_until_delivery = j + (max_travel_time_matrix[origin][destination][i] - k)
                time_of_delivery = time + time_until_delivery
                for l in range(len(cargos)):
                    if time_of_delivery >= cargos[l][7]:
                        quantity_of_lateness += time_of_delivery - cargos[l][7]
                if num_of_cargo_loaded != 0:
                    unit_cost_enumeration_matrix[i][j][k] = constant_coefficient * (
                                (distance / (max_travel_time_matrix[origin][destination][i] - k)) ** 3 * (
                                    num_of_cargo_loaded * cargo_weight + ship_models[i][2]) ** (2 / 3)) / (
                                                                        num_of_cargo_loaded * (distance / (
                                                                            max_travel_time_matrix[origin][destination][
                                                                                i] - k)) * 24) + (
                                                                        quantity_of_lateness * penalty) / (
                                                                        num_of_cargo_loaded * distance)
                else:
                    unit_cost_enumeration_matrix[i][j][k] = np.inf
    return unit_cost_enumeration_matrix


# def get_cargo_routing_plan(cost_matrix, moving_average_direct, moving_average_routed, max_travel_time_matrix, ship_models, ports):
#     total_routed_cargos = np.sum(moving_average_routed)
#     temp_cost_matrix = np.full((len(ports),len(ports)),np.inf)
#     for n in range(len(ports)):
#         for m in range(len(ports)):
#             temp_cost_matrix[n][m] = max_travel_time_matrix[n][m][len(ship_models)-1] + np.ceil(ship_models[len(ship_models)-1][1] / (moving_average_direct[n][m] + total_routed_cargos))
#     temp_route_matrix = np.full((len(ports),len(ports),len(ports),1))
#     route_matrix = np.full((len(ports),len(ports),len(ports),1))
#     for i in range(len(ports)):
#         for j in range(len(ports)):
#             cost = 0
#             search = True
#             current_location = i
#             candidate = list(range(len(ports)))
#             while search == True:
#                 for k in candidate:
#                     cost += temp_cost_matrix[current_location][k]
#                     if cost < cost_matrix[current_location][j] and k != j:
#                         current_location = k
#     return

### get user inputs
num_of_ports = int(input('how many ports to generate?(input \'5\' to be safe): '))
num_of_ship_models = int(input('input number of ship models(input \'3\' to be safe,maximum is 9): '))
num_of_ships = int(input('input number of ships: '))
reposition_interval = int(input('what is interval between two repositions?: '))
moving_average_length = int(input('what is moving average length?: '))
shipping_during_rebalance = bool(int(input('if loading cargos during reposition? 0 for no, 1 for yes: ')))
demand_forecasting_model = bool(int(input('which model to use for demand forecasting? 0 for moving average, 1 for linear regression: ')))
container_routing_ratio = float(input('what is the ratio of containers that accept container routing?: '))

### create entities
## generate ports(ports, attributes)[id, x cordinate, y coordinate]
ports = np.zeros((num_of_ports, 3))
for n in range(num_of_ports):
    ports[n][0] = n
    ports[n][1] = np.random.randint(0, 1001)
    ports[n][2] = np.random.randint(0, 1001)

## generate distance matrix (origin,destination)
distance_matrix = np.zeros((num_of_ports, num_of_ports))
for n in range(num_of_ports):
    for m in range(num_of_ports):
        if n != m:
            distance_matrix[n][m] = np.sqrt((ports[n][1] - ports[m][1]) ** 2 + (ports[n][2] - ports[m][2]) ** 2)
            if distance_matrix[n][m] == 0:
                print('Err: ports with identical coordination detected')
                time.sleep(5)

## generate ship models(ship models, attributes)[id, capacity, lightweight, min speed, max speed]
ship_models = np.zeros((num_of_ship_models, 5))
for n in range(num_of_ship_models):
    ship_models[n][0] = n
    ship_models[n][1] = 15 * (n + 1)
    ship_models[n][2] = 1 / 3 * 15 * (n + 1)
    ship_models[n][3] = 10 - 2 / (n + 1)
    ship_models[n][4] = 25 - 5 / (n + 1)

## generate ships[id, model, status[0 out of service, 1 standby, 2 in service], sector origin, sector destination, sector finished, sector total, occupied capacity, total capacity]
ships = np.zeros((num_of_ships, 9))
for n in range(num_of_ships):
    ships[n][0] = n
    ships[n][1] = np.random.randint(0, num_of_ship_models)
    ships[n][2] = 1
    ships[n][3] = np.random.randint(0, num_of_ports)
    ships[n][4] = ships[n][3]
    ships[n][6] = np.inf
    ships[n][8] = ship_models[int(ships[n][1])][1]

## generate ship event log[time, event type, event target, occupied capacity during travel]
for n in range(num_of_ships):
    globals()[f'ship_event_log_{n}'] = np.zeros((0, 4))

## generate cargos[id, origin, destination, spawn time, status, routing[0 if not applicable, 1 routable], sector initial time, sector deadline, sector origin, sector destination, sector finished, sector total]
cargos = np.zeros((0, 12))

## generate cargo archive
archived_cargos = np.zeros((0, 12))

## benchmark dataset
num_of_cargo_delivered_data = np.zeros((0, 2))
throughput_data = np.zeros((0, 2))
fuel_efficiency_benchmark_data = np.zeros((0, 3))
average_fuel_cost_data = np.zeros((0, 2))
unit_fuel_cost_data = np.zeros((0, 2))
shipping_time_span_data = np.zeros((0, 2))
average_shipping_time_span_data = np.zeros((0, 2))
unit_shipping_time_span_data = np.zeros((0, 2))
num_of_usable_ships_data = np.zeros((0, len(ship_models) + 1))
global_queue_size_data = np.zeros((0, len(ports) + 1))

## calculations
min_travel_time_matrix, max_travel_time_matrix = get_travel_time_matrix(distance_matrix=distance_matrix,
                                                                        ship_models=ship_models,
                                                                        ports=ports)  # (port i, port j, ship model)
unit_cost_lower_bound = get_theoretical_unit_cost_lower_bound(ship_models=ship_models)

## initial conditions
t = -1
go_simulation = True
num_of_cargo_generated = 0

## benchmark initial conditions
num_of_cargo_delivered = 0
throughput = 0
theoretical_cumulative_fuel_cost = 0
actual_cumulative_fuel_cost = 0
average_fuel_cost = 0
unit_fuel_cost = 0
shipping_time_span = 0
average_shipping_time_span = 0
unit_shipping_time_span = 0

# cost_matrix = np.zeros((len(ports),len(ports)))
routing_table = np.full((len(ports), len(ports), len(ports)), np.inf)
cargo_spawn_pattern_history = np.zeros((moving_average_length, len(ports), len(ports)))
cargo_spawn_pattern_direct_history = np.zeros((moving_average_length, len(ports), len(ports)))
cargo_spawn_pattern_routed_history = np.zeros((moving_average_length, len(ports), len(ports)))
cargo_spawn_pattern_actual_history = np.zeros((moving_average_length, len(ports), len(ports)))

### simulation
while go_simulation == True:
    t += 1
    print('### epoch', t, 'starts')
    print('start system performance benchmarking')
    num_of_cargo_delivered, throughput, theoretical_cumulative_fuel_cost, actual_cumulative_fuel_cost, average_fuel_cost, unit_fuel_cost, shipping_time_span, average_shipping_time_span, unit_shipping_time_span, num_of_usable_ships, global_queue_size = efficiency_benchmark(
        ship_models=ship_models, ships=ships, cargos=cargos, time=t, distance_matrix=distance_matrix,
        actual_cumulative_fuel_cost=actual_cumulative_fuel_cost,
        theoretical_cumulative_fuel_cost=theoretical_cumulative_fuel_cost, unit_cost_lower_bound=unit_cost_lower_bound,
        num_of_cargo_delivered=num_of_cargo_delivered, throughput=throughput, shipping_time_span=shipping_time_span,
        unit_fuel_cost=unit_fuel_cost, average_shipping_time_span=average_shipping_time_span,
        unit_shipping_time_span=unit_shipping_time_span, average_fuel_cost=average_fuel_cost)
    print('start updating status of ships and cargos in travel')
    ships, cargos = ship_and_cargo_status_update(ships=ships, cargos=cargos, time=t)
    cargos, archived_cargos = get_archived_cargo(cargos=cargos, archived_cargos=archived_cargos)
    print('inactive cargo information are archived')
    ## spawn new cargos; create event log for each; cargo routing
    num_of_cargo_generated_by_last_epoch = num_of_cargo_generated  # record for editing cargo list and creating event logs
    print(num_of_cargo_generated_by_last_epoch,
          'cargos had been generated by the end of last epoch. start spawning new cargos for this epoch')
    cargo_spawn_pattern, cargo_spawn_pattern_history = get_cargo_spawn_pattern(ports=ports,
                                                                               cargo_spawn_pattern_history=cargo_spawn_pattern_history)
    print('cargo spawn pattern history\n', cargo_spawn_pattern_history)
    spawned_cargos, num_of_cargo_generated, cargo_spawn_pattern_direct, cargo_spawn_pattern_routed = get_cargo_spawn(
        time=t, cargo_spawn_pattern=cargo_spawn_pattern, num_of_cargo_generated=num_of_cargo_generated)
    cargos = np.append(cargos, spawned_cargos, axis=0)
    print('new cargos spawned and added to cargo dataset. spawned cargo list\n', spawned_cargos)
    get_cargo_event_log(num_of_cargo_generated_by_last_epoch=num_of_cargo_generated_by_last_epoch,
                        num_of_cargo_generated=num_of_cargo_generated)
    print('cargo event log created for spawned cargos')
    print(num_of_cargo_generated, 'cargos has been generated until this epoch')
    print(
        'start routing spawned cargos. Dijkstra shortest path algorithm used for deciding next stop if cargo routing is requested on a cargo')
    cargos, cargo_spawn_pattern_actual = get_cargo_routing_and_cargo_spawn_pattern_actual(ports=ports, cargos=cargos,
                                                                                          routing_table=routing_table)
    cargo_spawn_pattern_actual_history = get_cargo_routing_and_cargo_spawn_pattern_actual_history(ports=ports,
                                                                                                  cargo_spawn_pattern_actual_history=cargo_spawn_pattern_actual_history,
                                                                                                  cargo_spawn_pattern_actual=cargo_spawn_pattern_actual)
    print('cargo spawn pattern actual history\n', cargo_spawn_pattern_actual_history)
    cargo_spawn_pattern_direct_history, cargo_spawn_pattern_routed_history = get_seperated_cargo_spawn_pattern_history(
        ports=ports, cargo_spawn_pattern_direct_history=cargo_spawn_pattern_direct_history,
        cargo_spawn_pattern_routed_history=cargo_spawn_pattern_routed_history,
        cargo_spawn_pattern_direct=cargo_spawn_pattern_direct, cargo_spawn_pattern_routed=cargo_spawn_pattern_routed)
    print('cargo spawn pattern directed history\n', cargo_spawn_pattern_direct_history,
          '\ncargo spawn pattern routed history\n', cargo_spawn_pattern_routed_history)
    print('all cargo spawn patterns\ncargo spawn pattern\n', cargo_spawn_pattern, '\ncargo spawn pattern_actual\n',
          cargo_spawn_pattern_actual, '\ncargo spawn pattern for direct shipping\n', cargo_spawn_pattern_direct,
          '\ncargo spawn pattern for routed shipping\n', cargo_spawn_pattern_routed)
    moving_average_matrix = get_moving_average_matrix(cargo_spawn_pattern_history=cargo_spawn_pattern_history,
                                                      ports=ports, moving_average_length=moving_average_length, time=t)
    demand_forecasting_matrix = get_demand_forecasting(cargo_spawn_pattern_history=cargo_spawn_pattern_history,
                                                       ports=ports, moving_average_length=moving_average_length)
    moving_average_actual_matrix = get_moving_average_matrix(
        cargo_spawn_pattern_history=cargo_spawn_pattern_actual_history, ports=ports,
        moving_average_length=moving_average_length, time=t)
    demand_forecasting_actual_matrix = get_demand_forecasting(
        cargo_spawn_pattern_history=cargo_spawn_pattern_actual_history,
        ports=ports, moving_average_length=moving_average_length)
    moving_average_direct_matrix = get_moving_average_matrix(
        cargo_spawn_pattern_history=cargo_spawn_pattern_direct_history, ports=ports,
        moving_average_length=moving_average_length, time=t)
    demand_forecasting_direct_matrix = get_demand_forecasting(
        cargo_spawn_pattern_history=cargo_spawn_pattern_direct_history,
        ports=ports, moving_average_length=moving_average_length)
    moving_average_routed_matrix = get_moving_average_matrix(
        cargo_spawn_pattern_history=cargo_spawn_pattern_routed_history, ports=ports,
        moving_average_length=moving_average_length, time=t)
    demand_forecasting_routed_matrix = get_demand_forecasting(
        cargo_spawn_pattern_history=cargo_spawn_pattern_routed_history,
        ports=ports, moving_average_length=moving_average_length)
    print('all types of moving average retrieved\nmoving average\n', moving_average_matrix, '\nmoving average actual\n',
          moving_average_actual_matrix, '\nmoving average direct\n', moving_average_direct_matrix,
          '\nmoving average routed\n', moving_average_routed_matrix)

    # switch demand forecasting model if needed
    if demand_forecasting_model == 0:
        demand_forecasting_matrix = moving_average_matrix
        demand_forecasting_actual_matrix = moving_average_actual_matrix
        demand_forecasting_direct_matrix = moving_average_direct_matrix
        demand_forecasting_routed_matrix = moving_average_routed_matrix

    cost_matrix = get_cost_matrix(moving_average_length=moving_average_length,
                                  moving_average_matrix=demand_forecasting_matrix, ship_models=ship_models, ports=ports,
                                  max_travel_time_matrix=max_travel_time_matrix)
    cost_matrix_actual = get_cost_matrix(moving_average_length=moving_average_length,
                                         moving_average_matrix=demand_forecasting_actual_matrix,
                                         ship_models=ship_models, ports=ports,
                                         max_travel_time_matrix=max_travel_time_matrix)
    cost_matrix_direct = get_cost_matrix(moving_average_length=moving_average_length,
                                         moving_average_matrix=demand_forecasting_direct_matrix,
                                         ship_models=ship_models, ports=ports,
                                         max_travel_time_matrix=max_travel_time_matrix)
    cost_matrix_routed = get_cost_matrix(moving_average_length=moving_average_length,
                                         moving_average_matrix=demand_forecasting_routed_matrix,
                                         ship_models=ship_models, ports=ports,
                                         max_travel_time_matrix=max_travel_time_matrix)
    print('cost matrix\n', cost_matrix, '\ncost matrix actual\n', cost_matrix_actual, '\ncost matrix direct\n',
          cost_matrix_direct, '\ncost matrix routed\n', cost_matrix_routed)
    cargos = get_cargo_deadline(cargos=cargos, cost_matrix_actual=cost_matrix_actual, time=t)
    print('deadlines have been assigned to cargos on every shipping leg')
    print('start solving for routing table base on cost on every shipping leg(cost is calculated in time)')
    routing_table = get_routing_table(cost_matrix=cost_matrix, ports=ports)
    print('routing table as below\n', routing_table)

    ## bechmarke dataset updates
    # record cumulative fuel cost data
    temp_fuel_efficiency_lst = np.zeros((1, 3))
    temp_fuel_efficiency_lst[0][0] = t
    temp_fuel_efficiency_lst[0][1] = theoretical_cumulative_fuel_cost
    temp_fuel_efficiency_lst[0][2] = actual_cumulative_fuel_cost
    fuel_efficiency_benchmark_data = np.append(fuel_efficiency_benchmark_data, temp_fuel_efficiency_lst, axis=0)
    # record unit cost data
    temp_unit_fuel_cost_lst = np.zeros((1, 2))
    temp_unit_fuel_cost_lst[0][0] = t
    temp_unit_fuel_cost_lst[0][1] = unit_fuel_cost
    unit_fuel_cost_data = np.append(unit_fuel_cost_data, temp_unit_fuel_cost_lst, axis=0)
    # record num of cargo delivered
    temp_num_of_cargo_delivered_lst = np.zeros((1, 2))
    temp_num_of_cargo_delivered_lst[0][0] = t
    temp_num_of_cargo_delivered_lst[0][1] = num_of_cargo_delivered
    num_of_cargo_delivered_data = np.append(num_of_cargo_delivered_data, temp_num_of_cargo_delivered_lst, axis=0)
    # record throughput
    temp_throughput_lst = np.zeros((1, 2))
    temp_throughput_lst[0][0] = t
    temp_throughput_lst[0][1] = throughput
    throughput_data = np.append(throughput_data, temp_throughput_lst, axis=0)
    # recoed shipping time span
    temp_shipping_time_span_lst = np.zeros((1, 2))
    temp_shipping_time_span_lst[0][0] = t
    temp_shipping_time_span_lst[0][1] = shipping_time_span
    shipping_time_span_data = np.append(shipping_time_span_data, temp_shipping_time_span_lst, axis=0)
    # recoed average shipping time span
    temp_average_shipping_time_span_lst = np.zeros((1, 2))
    temp_average_shipping_time_span_lst[0][0] = t
    temp_average_shipping_time_span_lst[0][1] = average_shipping_time_span
    average_shipping_time_span_data = np.append(average_shipping_time_span_data, temp_average_shipping_time_span_lst,
                                                axis=0)
    # record unit shipping time span
    temp_unit_shipping_time_span_lst = np.zeros((1, 2))
    temp_unit_shipping_time_span_lst[0][0] = t
    temp_unit_shipping_time_span_lst[0][1] = unit_shipping_time_span
    unit_shipping_time_span_data = np.append(unit_shipping_time_span_data, temp_unit_shipping_time_span_lst, axis=0)
    # record average cost data
    temp_average_fuel_cost_lst = np.zeros((1, 2))
    temp_average_fuel_cost_lst[0][0] = t
    temp_average_fuel_cost_lst[0][1] = average_fuel_cost
    average_fuel_cost_data = np.append(average_fuel_cost_data, temp_average_fuel_cost_lst, axis=0)
    # record num of usable ships
    temp_num_of_usable_ships_lst = np.zeros((1, len(ship_models) + 1))
    temp_num_of_usable_ships_lst[0][0] = t
    print('num of usable ships are', num_of_usable_ships)
    for n in range(len(ship_models)):
        temp_num_of_usable_ships_lst[0][n + 1] = np.sum(num_of_usable_ships[n, :])
    print('num of usable ships of each model are', temp_num_of_usable_ships_lst)
    num_of_usable_ships_data = np.append(num_of_usable_ships_data, temp_num_of_usable_ships_lst, axis=0)
    # record global queue size
    temp_global_queue_size_lst = np.zeros((1, len(ports) + 1))
    temp_global_queue_size_lst[0][0] = t
    print('global queue size are', global_queue_size)
    for n in range(len(ports)):
        temp_global_queue_size_lst[0][n + 1] = np.sum(global_queue_size[n, :])
    print('global queue size at each port are', temp_global_queue_size_lst)
    global_queue_size_data = np.append(global_queue_size_data, temp_global_queue_size_lst, axis=0)
    # benchmark dataset output
    if t == 24 * 7:
        print("fuel_efficiency_benchmark_data", fuel_efficiency_benchmark_data)
        np.savetxt('theoretical and actual fuel consumption 0.csv', fuel_efficiency_benchmark_data, delimiter=',')
        print("unit_fuel_cost_data", unit_fuel_cost_data)
        np.savetxt('unit cost 0.csv', unit_fuel_cost_data, delimiter=',')
        print("average_fuel_cost_data", average_fuel_cost_data)
        np.savetxt('average cost 0.csv', average_fuel_cost_data, delimiter=',')
        print("num_of_cargo_delivered_data", num_of_cargo_delivered_data)
        np.savetxt('num of cargo delivered 0.csv', num_of_cargo_delivered_data, delimiter=',')
        print("throughput_data", throughput_data)
        np.savetxt('throughput 0.csv', throughput_data, delimiter=',')
        print("shipping_time_span_data", shipping_time_span_data)
        np.savetxt('shipping time span 0.csv', shipping_time_span_data, delimiter=',')
        print("average_shipping_time_span_data", average_shipping_time_span_data)
        np.savetxt('average shipping time span 0.csv', average_shipping_time_span_data, delimiter=',')
        print("unit_shipping_time_span_data", unit_shipping_time_span_data)
        np.savetxt('unit shipping time span 0.csv', unit_shipping_time_span_data, delimiter=',')
        print("num_of_usable_ships_data", num_of_usable_ships_data)
        np.savetxt('num of usable ships 0.csv', num_of_usable_ships_data, delimiter=',')
        print("queue_size_data", global_queue_size_data)
        np.savetxt('queue size 0.csv', global_queue_size_data, delimiter=',')
    ## arrange departure and cargo assignment for each cargo flow
    print('start departure scheduling')
    for n in range(len(ports) * (len(ports) - 1)):
        print('start scheduling for cargo flow', n)
        # calculations
        cargoflow_index = n
        origin, destination = cargoflow_index_converter(index=cargoflow_index, ports=ports)
        print('origin', origin, 'destination', destination)
        num_of_ship = get_num_of_ship(ships=ships, ship_models=ship_models, location=origin)
        print('the number of usable ships at port', origin, 'is', num_of_ship, 'for each ship model respectively')
        # moving_average = get_theoretical_moving_average(time=t, cargos = cargos, origin = origin, destination = destination, moving_average_length=moving_average_length)
        # moving_average = get_theoretical_moving_average(cargo_spawn_pattern_history=cargo_spawn_pattern_history,origin=origin,destination=destination,time=t,moving_average_length=moving_average_length)
        moving_average = demand_forecasting_actual_matrix[origin][destination]
        print('the moving average of rate of incoming cargo of this cargo flow is', moving_average)
        # delivery_time_span = get_delivery_time_span(origin = origin, destination = destination, moving_average = moving_average, max_travel_time_matrix = max_travel_time_matrix, ship_models = ship_models)
        delivery_time_span = cost_matrix_actual[origin][destination]
        print('delivery time span for shipping leg', cargoflow_index, 'is', delivery_time_span)
        print('guaranteed number of epoch(s) for newly spawned cargos to be delivered is', delivery_time_span)
        # set deadline for cargos in target cargo queue
        # cargos = get_cargo_deadline(cargos = cargos, delivery_time_span = delivery_time_span, origin = origin, destination = destination)
        # print('deadline updated for spawned cargos for next shipment sector')
        # initialize departure check
        departure_check = True
        print('start departure check for cargo flow', cargoflow_index)
        counter = 0
        while departure_check == True:
            counter += 1
            print('departure check iteration #', counter, 'for shipping leg', cargoflow_index, 'in epoch', t)
            queue_size, cargo_in_queue = get_cargo_in_queue(cargos=cargos, origin=origin, destination=destination)
            print('current queue size is', queue_size, '\ncargo in queue\n', cargo_in_queue)
            cargo_queue_alpha = get_cargo_queue_alpha(cargos=cargo_in_queue, ship_models=ship_models, time=t,
                                                      delivery_time_limit=delivery_time_span,
                                                      moving_average=moving_average)
            print('\ntheoretical cargo queue for unit cost enumeration\n', cargo_queue_alpha)
            unit_cost_enumeration_matrix = get_unit_cost_enumeration_matrix(queue_size=queue_size,
                                                                            moving_average=moving_average,
                                                                            cargos=cargo_queue_alpha, time=t,
                                                                            distance_matrix=distance_matrix,
                                                                            ship_models=ship_models,
                                                                            min_travel_time_matrix=min_travel_time_matrix,
                                                                            max_travel_time_matrix=max_travel_time_matrix,
                                                                            origin=origin, destination=destination)
            index_of_min_cost_from_enumeration = np.where(
                unit_cost_enumeration_matrix == np.amin(unit_cost_enumeration_matrix))
            print('unit_cost_enumeration_matrix', unit_cost_enumeration_matrix)
            print(
                'unit_cost_enumeration_matrix indicates best comnination for cost efficiency (ship model, departure time, speed level )',
                index_of_min_cost_from_enumeration[0][0], index_of_min_cost_from_enumeration[1][0],
                index_of_min_cost_from_enumeration[2][0])

            # optimal timing? ship availability? schedule departure if feasible
            print('start checking if best timing and if ship available')
            if np.sum(num_of_ship) >= 1 and index_of_min_cost_from_enumeration[1][0] == 0 and np.min(
                    unit_cost_enumeration_matrix) != np.inf:
                print('theoretical optimal departure timing right now, and ship available(!! model not guaranteed !!)')
                model_of_departure_ship, time_to_arrive = get_model_of_departure_ship_and_time_to_arrive(
                    unit_cost_enumeration_matrix=unit_cost_enumeration_matrix, num_of_ship=num_of_ship, origin=origin,
                    destination=destination, max_travel_time_matrix=max_travel_time_matrix)
                print('ship selected for this departure is ship model', model_of_departure_ship,
                      'this departure ship will arrive in', time_to_arrive, 'epoch(s)')
                print('departure scheduling starts')
                cargos, ships, num_of_ship = departure_scheduling(cargoflow=cargoflow_index, counter=counter,
                                                                  cargos=cargos, ships=ships, ship_models=ship_models,
                                                                  num_of_ship=num_of_ship,
                                                                  model_of_departure_ship=model_of_departure_ship,
                                                                  time_to_arrive=time_to_arrive, origin=origin,
                                                                  destination=destination, time=t)
            elif index_of_min_cost_from_enumeration[1][0] == 0 and np.min(unit_cost_enumeration_matrix) != np.inf:
                print('theoretical optimal departure timing right now, but no ship available')
                departure_check = False
            else:
                print('theoretical optimal departure timing is not reached yet. ship availability unknown')
                departure_check = False
    ## reposition
    print('ship reposition and reposition optimization')
    # port_moving_average = get_port_moving_average(cargos = cargos, archived_cargos = archived_cargos, time = t)
    # print('moving averages of rate of incoming cargos at each port is', port_moving_average)
    port_moving_average = get_port_moving_average(moving_average_actual_matrix=demand_forecasting_actual_matrix)
    print('moving averages of rate of incoming cargos at each port is', port_moving_average)
    for m in range(len(ship_models)):
        # auditing for ship repositioning decisions
        print('checking reposition for ship model', m)
        ship_buffer_and_liquidity = get_ship_buffer_and_liquidity(ships=ships, ports=ports, target_ship_model=m)
        print('pre-reposition ship buffer and liquidity check result for ship model', m, '\n',
              ship_buffer_and_liquidity)
        target_buffer_size = get_target_buffer_size(ship_buffer_and_liquidity=ship_buffer_and_liquidity,
                                                    port_moving_average=port_moving_average, ports=ports)
        print('pre-target buffer sizes at each port for ship model', m, '\n', target_buffer_size)
        ship_supply_and_demand = get_ship_supply_and_demand(
            ship_buffer_and_liquidity=np.copy(ship_buffer_and_liquidity),
            target_buffer_size=target_buffer_size, ports=ports)
        print('pre-supply and demand of ship model', m, 'among all ports\n', ship_supply_and_demand)
        # periodic global repositioning
        print('checking if global reposition scheduled for this epoch')
        if t % reposition_interval == 0 and t != 0:
            print('scheduling periodic global reposition')
            ship_buffer_and_liquidity = get_ship_buffer_and_liquidity(ships=ships, ports=ports, target_ship_model=m)
            print('reposition ship buffer and liquidity check result for ship model', m, '\n',
                  ship_buffer_and_liquidity)
            target_buffer_size = get_target_buffer_size(ship_buffer_and_liquidity=ship_buffer_and_liquidity,
                                                        port_moving_average=port_moving_average, ports=ports)
            print('target buffer sizes at each port for ship model', m, '\n', target_buffer_size)
            ship_supply_and_demand = get_ship_supply_and_demand(
                ship_buffer_and_liquidity=np.copy(ship_buffer_and_liquidity),
                target_buffer_size=target_buffer_size, ports=ports)
            print('demand of ship model', m, 'among all ports\n', ship_supply_and_demand)
            # ILP Model and solution using MIP package
            print('creating ILP model')
            N = -1 * ship_supply_and_demand
            d = distance_matrix
            V = set(range(len(d)))
            model = Model()
            x = [[model.add_var(var_type=INTEGER, lb=-10000) for j in V] for i in V]
            y = [[model.add_var(var_type=INTEGER, lb=-10000) for j in V] for i in V]
            z = [[model.add_var(var_type=INTEGER, lb=-10000) for j in V] for i in V]
            model.objective = minimize(xsum(d[i][j] * (y[i][j] + z[i][j]) for i in V for j in V if j >= i))
            for i in V:
                for j in V:
                    model += x[i][j] + x[j][i] == 0
            for i in V:
                for j in V:
                    model += y[i][j] - z[i][j] == x[i][j]
            for i in V:
                model += xsum(y[i][j] - z[i][j] for j in V) == N[0][i]
            model.optimize()
            reposition_plan = np.zeros((len(ports), len(ports)))
            for i in range(len(ports)):
                for j in range(len(ports)):
                    reposition_plan[i][j] = x[i][j].x
            print('optimal reposition plan for min(total travel distance) is\n', reposition_plan)
            print('start executing reposition plan')
            ships, cargos = excute_reposition_plan(reposition_plan=reposition_plan, target_ship_model=m,
                                                   ship_models=ship_models, ports=ports, ships=ships, cargos=cargos,
                                                   max_travel_time_matrix=max_travel_time_matrix, time=t)
        print('checking if buffer size critical level tripwire is triggered for ship model', m)
        print('checking ship buffer critical level tripwire')
        critical_level_tripwire = True
        counter = 0
        while critical_level_tripwire == True:
            counter += 1
            print('ship buffer critical level check #', counter)
            ship_buffer_and_liquidity = get_ship_buffer_and_liquidity(ships=ships, ports=ports, target_ship_model=m)
            print('reposition ship buffer and liquidity check result for ship model', m, '\n',
                  ship_buffer_and_liquidity)
            target_buffer_size = get_target_buffer_size(ship_buffer_and_liquidity=ship_buffer_and_liquidity,
                                                        port_moving_average=port_moving_average, ports=ports)
            print('target buffer sizes at each port for ship model', m, '\n', target_buffer_size)
            ship_supply_and_demand = get_ship_supply_and_demand(
                ship_buffer_and_liquidity=np.copy(ship_buffer_and_liquidity),
                target_buffer_size=target_buffer_size, ports=ports)
            print('supply and demand of ship model', m, 'among all ports\n', ship_supply_and_demand)
            print('checking if suitable ship donor and acceptor exist')
            acceptor_index, donor_index = get_donor_and_acceptor(ship_buffer_and_liquidity=ship_buffer_and_liquidity,
                                                                 target_buffer_size=target_buffer_size, ports=ports)
            if acceptor_index == None or donor_index == None:
                print('suitable donor or acceptor, or both do not exist. tripwire not triggered and search cancelled')
                critical_level_tripwire = False
            elif acceptor_index != None and donor_index != None:
                print('both donor', donor_index, 'and acceptor', acceptor_index,
                      'found. start scheduling paired reposition')
                ships, cargos = execute_paired_reposition(counter=counter, acceptor_index=acceptor_index,
                                                          donor_index=donor_index, cargos=cargos, ships=ships,
                                                          ship_models=ship_models, target_ship_model=m,
                                                          max_travel_time_matrix=max_travel_time_matrix, time=t)
    print('###epoch', t, 'ends')
