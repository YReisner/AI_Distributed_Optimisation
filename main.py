import numpy as np
import matplotlib.pyplot as plt
import copy

class Agent():
    '''
    A single agent. Because the assumption is that each agent holds only one variable, the code is written
    as if the agent is the variable.
    '''
    def __init__(self,domain_init,num_agents):
        '''
        :param domain_init: size of the domain for the variables
        :param num_agents: number of agents in the problem
        '''
        self.state = np.random.randint(10)
        self.constraints = np.zeros((domain_init,domain_init,num_agents))
        self.neighbors = np.zeros(num_agents)
        self.agent_view = np.ones(num_agents)*20
        self.agent_view_r = np.zeros(num_agents)
        self.pair = None
        self.in_mailbox = []
        self.requests = []
        self.current_R = 0
        self.r_view = []
        self.potential_state = None
        self.main = False


    def set_constraint_table(self,other_agent,costs):
        '''
        Used in the creation of the problem, simulating the constraints
        :param other_agent: Int, index of the agent a constraint was created with
        :param costs: Int matrix, the costs of the constraints
        '''
        self.constraints[:,:,other_agent] = costs

    def get_neighbors(self):
        '''
        After all constraints are made, we simply give an indication of who our neighbors are.
        Because we use indices for everything, it's easier to keep a binary vector.
        '''
        for i in range(0,self.constraints.shape[2]):
            if sum(sum(self.constraints[:,:,i])) > 0:
                self.neighbors[i] = 1

    def find_best(self):
        '''
        Tries to find the best state for the agent. If a better state was found, the agent has a 0.7 chance to
        move to that state. This is a DSA-C style solution.
        '''
        if min(self.agent_view) < 11: # This simply makes sure we already received at least one message (not first iteration)
            cost_list = np.zeros(self.constraints.shape[0])
            for neighbor_index in range(len(self.neighbors)):
                if self.neighbors[neighbor_index] == 1:
                    for i in range(len(cost_list)):

                        cost_list[i] += self.constraints[i,int(self.agent_view[neighbor_index]),neighbor_index]
            best = np.where(cost_list == cost_list.min())
            best = best[0]
            if len(best) == 1:
                best = best[0]
                if best != self.state:
                    if np.random.random() <= 0.7:
                        self.state = best
            else:
                if np.random.random() <= 0.7:
                    choice = np.random.randint(len(best))
                    chosen = best[choice]
                    while chosen == self.state:
                        choice = np.random.randint(len(best))
                        chosen = best[choice]
                    self.state = chosen





    def get_cost(self):
        '''
        :return: The current cost of the agent's state
        '''
        if min(self.agent_view) < 11: # This simply makes sure we already received at least one message (not first iteration)
            cost = 0
            for neighbor_index in range(len(self.neighbors)):
                if self.neighbors[neighbor_index] == 1:
                    cost += self.constraints[self.state,int(self.agent_view[neighbor_index]),neighbor_index]
            return cost
        else:
            return None

    def alone_best(self,my_index):
        '''
        Agent didn't pair with anyone - finds the best option for himself and calculates the R for it. This
        is for MGM2
        '''
        if min(self.agent_view) < 11:  # This simply makes sure we already received at least one message (not first iteration)
            cost_list = np.zeros(self.constraints.shape[0])
            for neighbor_index in range(len(self.neighbors)):
                if self.neighbors[neighbor_index] == 1:
                    for i in range(len(cost_list)):
                        cost_list[i] += self.constraints[i, int(self.agent_view[neighbor_index]), neighbor_index]
            best = np.argmin(cost_list)
            R = cost_list[self.state] - cost_list[best]
            self.current_R = R
            self.potential_state = best
            return R, my_index,best

    def pair_best(self,other,my_index,other_index):
        '''
        Finds the best states/actions for a pair of agents
        :param other: paired agent
        :param my_index: my index in the agent list
        :param other_index: pair index in the agent list
        :return: R, the indices, the agents and the potential states
        '''
        if min(self.agent_view) < 11:  # This simply makes sure we already received at least one message (not first iteration)
            costs = np.zeros((10,10))
            for i in range(costs.shape[0]):
                for j in range(costs.shape[1]):
                    for neighbor_index in range(len(self.neighbors)):
                        if self.neighbors[neighbor_index] == 1:
                            if neighbor_index == other_index:
                                costs[i,j] += self.constraints[i,j,neighbor_index]
                            else:
                                costs[i, j] += self.constraints[i, int(self.agent_view[neighbor_index]), neighbor_index]
                    for neighbor_index in range(len(other.neighbors)):
                        if other.neighbors[neighbor_index] == 1 and neighbor_index != my_index:
                            costs[i,j] += self.constraints[j,int(other.agent_view[neighbor_index]),neighbor_index]
            best = np.unravel_index(costs.argmin(), costs.shape)
            best_cost = costs[best[0],best[1]]
            R = costs[self.state,other.state] - best_cost
            self.current_R = R
            self.potential_state = best[0]
            other.potential_state = best[1]
            return R,my_index,other_index, self,other, best


def create_problem(num_agents,domain_size,p1,p2):
    '''
    :param num_agents: Int, number of agents in the simulation
    :param domain_size: Int, the domain size for the vaiables
    :param p1: Float between 0 and 1, the chance two agents are neighbors (constrained together)
    :param p2: Float between 0 and 1, the chance two neighbors variable combination cost is higher than 0
    :param seed: Int, the seed for all the randomization
    :return: a list of agents
    '''
    agents = []
    for i in range(0,num_agents):
        new_agent = Agent(domain_size,num_agents)
        agents.append(new_agent)

    for i in range(0,num_agents):
        agent1 = agents[i]
        for g in range(i+1,num_agents):
            agent2 = agents[g]
            if np.random.random() <= p1:
                constraints_loc = np.random.random((domain_size,domain_size))
                constraints_loc = constraints_loc <= p2
                unfiltered_costs = np.random.randint(10,size = (domain_size,domain_size))
                costs = np.zeros((domain_size,domain_size))
                costs[constraints_loc] = unfiltered_costs[constraints_loc]
                transposed_costs = np.transpose(costs)

                agent1.set_constraint_table(g,costs)
                agent2.set_constraint_table(i,transposed_costs)
        agent1.get_neighbors()
    return agents

def copy_problem(agents):
    '''
    Deep Copies the agents to use for a second algorithm, for the same problem
    :param agents: list of agents
    :return: deep copied list of agents
    '''
    new_agents = []
    for agent in agents:
        new_agent = copy.deepcopy(agent)
        new_agents.append(new_agent)
    return new_agents


def DSA_C(agents,iterations):
    '''
    Solves a distributed optimisation problem with DSA-C algorithm
    :param agents: list of agents
    :param iterations: number of iterations before stopping
    :return: iteration indexes and prices on those iterations
    '''
    locs = []
    costs = []
    out_mailbox = []

    # send messages before iterations start
    for agent_index in range(0, 30):
        agent = agents[agent_index]
        for neighbor in range(len(agent.neighbors)):
            if agent.neighbors[neighbor] == 1:
                out_mailbox.append((neighbor, agent_index, agent.state))

    for iteration in range(0, iterations):
        # Give away the messages
        for message in out_mailbox:
            for_agent = message[0]
            agents[for_agent].in_mailbox.append(message)
        out_mailbox = []

        # Step 3, each agent accepts message and sends messages
        for agent_index in range(0, 30):
            agent = agents[agent_index]
            for message in agent.in_mailbox:
                agent.agent_view[message[1]] = message[2]
            agent.in_mailbox = []
            for neighbor in range(len(agent.neighbors)):
                if agent.neighbors[neighbor] == 1:
                    out_mailbox.append((neighbor, agent_index, agent.state))

        #cost on first iteration, before changes are made
        if iteration == 0:
            current_cost = 0
            for agent in agents:
                cost = agent.get_cost()
                if cost is not None:
                    current_cost += cost
            locs.append(0)
            costs.append(current_cost)

        # Step 4 and 5, each agent finds best alternative and move to it by chance
        for agent in agents:
            agent.find_best()


        # save sum of costs every 10 iterations
        if iteration % 10 == 0:
            current_cost = 0
            for agent in agents:
                cost = agent.get_cost()
                if cost is not None:
                    current_cost += cost
            if iteration != 0:
                locs.append(iteration)
                costs.append(current_cost)
        #if iteration % 50 == 0:
        #    print("in iteration %d the total cost is %d" % (iteration, current_cost))

    return locs, costs



def mgm2(agents_MGM, iterations, pair_chance):
    '''
    Uses MGM2 algorithm to solve a distributed optimisation problem
    :param agents_MGM: list of agents
    :param iterations: amount of iterations to finish
    :param pair_chance: chance to send a friend request
    :return:  iteration indexes and prices on those iterations
    '''
    locs = []
    costs = []
    out_mailbox = []
    alone = np.ones(len(agents_MGM))
    offered = np.zeros(len(agents_MGM))
    r_box = []
    # Start by sending message before the first iteration
    for agent_index in range(len(agents_MGM)):
        agent = agents_MGM[agent_index]
        for neighbor in range(len(agent.neighbors)):
            if agent.neighbors[neighbor] == 1:
                out_mailbox.append((neighbor, agent_index, agent.state))


    for iteration in range(0, iterations):
        # distribute message to agent's mailboxes
        for message in out_mailbox:
            for_agent = message[0]
            agents_MGM[for_agent].in_mailbox.append(message)
        out_mailbox = []

        # Each agent "reads" the messages in his mailbox and updates his agent_view
        for agent_index in range(len(agents_MGM)):
            agent = agents_MGM[agent_index]
            for message in agent.in_mailbox:
                agent.agent_view[message[1]] = message[2]
            agent.in_mailbox = []

        # Sum of costs on first iteration (before first change)
        if iteration == 0:
            current_cost = 0
            for agent in agents_MGM:
                cost = agent.get_cost()
                if cost is not None:
                    current_cost += cost
            locs.append(0)
            costs.append(current_cost)


        # step 1, friend requests
        for agent_index in range(len(agents_MGM)):
            agent = agents_MGM[agent_index]
            if sum(agent.neighbors) > 0:
                if np.random.random() <= pair_chance:
                    choice = np.random.choice(np.where(agent.neighbors == 1)[0])

                    details = (choice, agent, agent_index)
                    agents_MGM[choice].requests.append(details)
                    offered[agent_index] = 1

        # step 2, pairs and alternatives
        for agent_index in range(len(agents_MGM)):
            agent = agents_MGM[agent_index]
            if offered[agent_index] == 0:
                if len(agent.requests) > 0:  # Did anyone offer?
                    agent.main = True
                    chosen = np.random.randint(len(agent.requests))
                    alone[agent.requests[chosen][2]] = 0
                    agents_MGM[agent.requests[chosen][2]].pair = agent
                    agent.pair = agents_MGM[agent.requests[chosen][2]]
                    message = agent.pair_best(agent.requests[chosen][1], agent_index, agent.requests[chosen][2])
                    for neighbor in range(len(agents_MGM)):
                        if agent.neighbors[neighbor] == 1 or message[4].neighbors[neighbor] == 1:
                            for_message = (neighbor, message)
                            r_box.append(for_message)
                else:  # No one offered
                    message = agent.alone_best(agent_index)
                    for neighbor in range(len(agents_MGM)):
                        if agent.neighbors[neighbor] == 1:
                            for_message = (neighbor, message)
                            r_box.append(for_message)
            agent.requests = []

        # Taking care of all the ones that offered and no one wanted to be friends with them :-(
        for agent_index in range(len(agents_MGM)):
            agent = agents_MGM[agent_index]
            if offered[agent_index] == 1 and alone[agent_index] == 1:
                message = agent.alone_best(agent_index)
                for neighbor in range(len(agents_MGM)):
                    if agent.neighbors[neighbor] == 1:
                        for_message = (neighbor, message)
                        r_box.append(for_message)

        offered = np.zeros(len(agents_MGM))
        alone = np.ones(len(agents_MGM))

        # Distributing r messages
        for message in r_box:
            for_agent = message[0]
            if agents_MGM[for_agent].pair is not None and agents_MGM[for_agent].main is False:
                agents_MGM[for_agent].pair.r_view.append(message[1][0])
            else:
                agents_MGM[for_agent].r_view.append(message[1][0])

        r_box = []

        # Choosing who changes state
        for agent_index in range(len(agents_MGM)):
            agent = agents_MGM[agent_index]
            if agent.pair is not None and agent.main is True:
                if agent.current_R > 0 and agent.current_R > np.max(agent.r_view):
                    agent.state = agent.potential_state
                    agent.pair.state = agent.pair.potential_state
            if agent.pair is None:
                if agent.current_R > 0 and agent.current_R > np.max(agent.r_view):
                    agent.state = agent.potential_state
            agent.r_view = []
            agent.pair = None
            agent.current_R = 0
            agent.potential_state = None
            agent.main = False

        # Give away the messages
        for agent_index in range(len(agents_MGM)):
            agent = agents_MGM[agent_index]
            for neighbor in range(len(agent.neighbors)):
                if agent.neighbors[neighbor] == 1:
                    out_mailbox.append((neighbor, agent_index, agent.state))

        # save sum of costs every 10 iterations
        if iteration % 10 == 0:
            current_cost = 0
            for agent in agents_MGM:
                cost = agent.get_cost()
                if cost is not None:
                    current_cost += cost
            if iteration != 0:
                locs.append(iteration)
                costs.append(current_cost)
        #if iteration % 50 == 0:
        #    print("in iteration %d the total cost is %d" % (iteration, current_cost))
    return locs, costs

def first_graph(p1,p2_array,times):
    '''
    produces error bar graphs for the final result (cost at last iteration) for each p2 option, keeping a p1 value
    constant.
    :param p1: p1 value (float between 0 and 1)
    :param p2_array: values to check for p2
    :param times: amount of runs for each p2 value
    '''
    dsa_results = np.zeros((times,len(p2_array)))
    mgm_results = np.zeros((times,len(p2_array)))
    for p_index in range(len(p2_array)):
        print('Getting %d runs for p2 = %2.1f' %(times,p2_array[p_index]))
        for time in range(times):
            agents1 = create_problem(30, 10, p1, p2_array[p_index])
            agents2 = copy_problem(agents1)
            locs_dsa, costs_dsa = DSA_C(agents1, 1000)
            locs_mgm, costs_mgm = mgm2(agents2, 1000, 0.6)
            dsa_results[time,p_index] = costs_dsa[-1]
            mgm_results[time, p_index] = costs_mgm[-1]
    means_dsa = np.mean(dsa_results,axis=0)
    std_dsa = np.std(dsa_results,axis=0)
    means_mgm = np.mean(mgm_results, axis=0)
    std_mgm = np.std(mgm_results, axis=0)

    plt.errorbar(p2_array,means_dsa,yerr=std_dsa,capsize=1,color='b',label='DSA')
    plt.errorbar(p2_array, means_mgm, yerr=std_mgm,capsize=1, color='g', label='MGM2')
    plt.legend()
    plt.title('DSA and MGM2 comparison for p1 = %2.1f per p2' %p1)
    plt.xlabel('p2 value')
    plt.ylabel('Sum of costs for %d runs' %times)
    plt.show()

def second_graph(p1,times):
    '''
    produces error bar plot for the results of both algorithms on 10 runs for p1 = 0.2 and p2 = 0.5 p2 always equals 1.
    :param p1: value of p1
    :param times: how many runs per p1
    '''
    dsa_results = np.zeros((times, len(np.arange(0,1010,10))))
    mgm_results = np.zeros((times, len(np.arange(0,1010,10))))
    for time in range(times):
        print("going through run number %d" %(time+1))
        agents1 = create_problem(30, 10, p1, 1)
        agents2 = copy_problem(agents1)
        locs_dsa, costs_dsa = DSA_C(agents1, 1001)
        locs_mgm, costs_mgm = mgm2(agents2, 1001, 0.6)
        dsa_results[time,:] = costs_dsa
        mgm_results[time, :] = costs_mgm
    means_dsa = np.mean(dsa_results,axis=0)
    std_dsa = np.std(dsa_results,axis=0)
    means_mgm = np.mean(mgm_results, axis=0)
    std_mgm = np.std(mgm_results, axis=0)

    plt.errorbar(np.arange(0,1010,10),means_dsa,yerr=std_dsa,capsize=1,color='b',label='DSA',alpha=0.5)
    plt.errorbar(np.arange(0,1010,10), means_mgm, yerr=std_mgm,capsize=1, color='g', label='MGM2',alpha=0.5)
    plt.legend()
    plt.title('DSA and MGM2 comparison for p1 = %2.1f by iteration' %p1)
    plt.xlabel('iteration')
    plt.ylabel('Sum of costs for %d runs' %times)
    plt.show()

def mgm_pair_chance_graph(times):
    '''
    Testing the effect of pair-chance on algorithm success
    :param times: amount of runs for each parameter (0.1 to 0.9)
    graphs the results
    '''
    mgm = np.zeros((times, 9))
    f = np.arange(0.1, 1, 0.1)
    for i in range(len(f)):
        print('now doing %2.1f' % f[i])
        for j in range(times):
            print('run number %d' % j)
            agents2 = create_problem(30, 10, 0.5, 0.5)


            locs_mgm, costs_mgm = mgm2(agents2, 1001, f[i])
            mgm[j, i] = costs_mgm[-1]
    means_mgm = np.mean(mgm, axis=0)
    std_mgm = np.std(mgm, axis=0)
    plt.errorbar(f, means_mgm, yerr=std_mgm, capsize=1)
    plt.title('MGM final cost per friend request chance, p1 = 0.5, p2 = 0.5')
    plt.xlabel('friend request value')
    plt.ylabel('Sum of costs for 10 runs')
    plt.show()

def one_run(p1,p2):
    '''
    Return the result (sum of costs) for each algorithm over the iterations, over the same problem. Currently two seperate graphs.
    :param p1: chances to be neighbors
    :param p2: chances to have a constraint higher than 10
    :return:
    '''
    agents1 = create_problem(30, 10, p1, p2)
    agents2 = copy_problem(agents1)
    locs, costs = DSA_C(agents1, 1000)
    locs_mgm, costs_mgm = mgm2(agents2, 1000, 0.7)

    plt.plot(locs_mgm, costs_mgm)
    plt.title('MGM2 tracking plot for one run')
    plt.ylabel('Current overall cost for all agents')
    plt.xlabel('Iteration number')
    plt.show()

    plt.plot(locs, costs)
    plt.title('DSA-C tracking plot for one run')
    plt.ylabel('Current overall cost for all agents')
    plt.xlabel('Iteration number')
    plt.show()



np.random.seed(42)

#first_graph(0.5,np.arange(0.1,1,0.1),10)

#second_graph(0.2,10)

#mgm_pair_chance_graph(10)

#one_run()