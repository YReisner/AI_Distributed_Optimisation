import numpy as np
import matplotlib.pyplot as plt

class Agent:
    '''
    A single agent. Because the assumption is that each agent holds only one variable, the code is written
    as if the agent is the variable.
    '''
    def __init__(self,domain_init,num_agents):
        '''
        :param domain_init: size of the domain for the variables
        :param num_agents: number of agents in the probblem
        '''
        self.state = np.random.randint(10)
        self.constraints = np.zeros((domain_init,domain_init,num_agents))
        self.neighbors = np.zeros(num_agents)
        self.agent_view = np.ones(num_agents)*20
        self.in_mailbox = []


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
            best = np.argmin(cost_list)
            if best != self.state:
                if np.random.random() <= 0.7:
                    self.state = best

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



def create_problem(num_agents,domain_size,p1,p2,seed):
    '''
    :param num_agents: Int, number of agents in the simulation
    :param domain_size: Int, the domain size for the vaiables
    :param p1: Float between 0 and 1, the chance two agents are neighbors (constrained together)
    :param p2: Float between 0 and 1, the chance two neighbors variable combination cost is higher than 0
    :param seed: Int, the seed for all the randomization
    :return: a list of agents
    '''
    agents = []
    np.random.seed(seed)
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

agents = create_problem(30,10,0.6,0.2,100)
locs = []
costs = []
out_mailbox = []
for iteration in range(0,200):
    # Give away the messages
    for message in out_mailbox:
        for_agent = message[0]
        agents[for_agent].in_mailbox.append(message)
    out_mailbox = []

    #Step 3, each agent accepts message and sends messages
    for agent_index in range(0,30):
        agent = agents[agent_index]
        for message in agent.in_mailbox:
            agent.agent_view[message[1]] = message[2]
        agent.in_mailbox =[]
        for neighbor in range(len(agent.neighbors)):
            if agent.neighbors[neighbor] == 1:
                out_mailbox.append((neighbor,agent_index,agent.state))

    # Step 4 and 5, each agent finds best alternative and move to it by chance
    for agent in agents:
        agent.find_best()

    if iteration%5 == 0:

        current_cost = 0
        for agent in agents:
            cost = agent.get_cost()
            if cost is not None:
                current_cost += cost
        if iteration != 0:
            print('now in iteration %d cost is %d' % (iteration,current_cost))
            locs.append(iteration)
            costs.append(current_cost)


plt.plot(locs,costs)
plt.title('DSA-C tracking plot for one run')
plt.ylabel('Current overall cost for all agents')
plt.xlabel('Iteration number')
plt.show()