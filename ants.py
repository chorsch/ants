import numpy as np
from gym import spaces
import pettingzoo
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gym.utils import seeding

class env(AECEnv):
    def __init__(self,
                num_food = 25,
                num_hazards = 5,
                hazard_punishment = 1,
                N = 10,
                x = 10,
                y = 10,
                food_value = 5):

        self.seed()

        #number of agents in the colony
        self.N = N
        #x dimension of the board
        self.x = x
        #y dimension of the board
        self.y = y

        self.num_food = num_food

        self.food_value = food_value

        self.num_hazards = num_hazards

        #the amount an agents energy is decreased for passing over a hazard
        self.hazard_punishment = hazard_punishment

        #the list of agents
        self.agents = [str(i) for i in range(N)]

        self.possible_agents = [str(r) for r in range(N)]

        #all agent locations on the board.
        #(0,0) is the colony
        self.agent_locations = {agent:(0,0) for agent in self.agents}

        #the amount of individual energy each agent has.
        #agents start with 10 (this can be changed)
        self.agent_energies = {agent: 10 for agent in self.agents}

        #number of steps taken on the current day
        self.steps = 0
        self.total_turns = 0

        #randomizes a board with dimensions x*y
        #the board is random 0s and 1s, and in the lower right corner area
        #defined by the hazard dimensions, it is random 0s, 1s, and 2s
        self.initialize_board()

        #the possible actions an agent can take
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.agents}

        #observation has agent location and value of surrounding tiles
        #0 boundary
        #1 is empty
        #2 is food
        #3 is hazard
        #4 is another agent
        #state[2] is upper
        #then move clockwise
        low = np.array([0] * 10)
        high = np.array([self.x, self.y] + [4] * 8)
        obs_space = spaces.Box(low=low, high=high, dtype=np.int64)
        self.observation_spaces = {agent:obs_space for agent in self.agents}

        self.dones = {i: False for i in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def reset(self):
        self.total_turns = 0
        self.rewards = {i: 0 for i in self.agents}
        self.done = False
        self.dones = {i: False for i in self.agents}
        self.agent_energies = {agent: 10 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        self.steps = 0

        self.initialize_board()

        #set the agent iterator
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        state = np.array([0,0,0,0,self.board[0,1],self.board[1,1], self.board[1,0],0,0,0])

    def step(self, action):
        #print("action ", action)
        self.steps = self.total_turns // self.N
        a = action
        self._cumulative_rewards[self.agent_selection] = 0
        self.rewards[self.agent_selection] = 0

        #only perform an action if the agent is alive
        if not self.dones[self.agent_selection]:

            #check which action was taken
            if a == 0:
                self.move_left()
            elif a == 1:
                self.move_up()
            elif a == 2:
                self.move_right()
            elif a == 3:
                self.move_down()

        #if all agents die, the sim is over
        if self.num_alive_agents() == 0:
            self.done = True

        x,y = self.agent_locations[self.agent_selection]
        state = [x, y]
        state += [self.board[x-1, y]] if x-1 >= 0 else [0]
        state += [self.board[x-1, y+1]] if y+1 < self.y and x-1 >= 0 else [0]
        state += [self.board[x, y+1]] if y+1 < self.y else [0]
        state += [self.board[x+1, y+1]] if y+1 < self.y  and x+1 < self.x else [0]
        state += [self.board[x+1, y]] if x+1 < self.x else [0]
        state += [self.board[x+1, y-1]] if x+1 < self.x and y-1 >= 0 else [0]
        state += [self.board[x, y-1]] if y-1 >= 0 else [0]
        state += [self.board[x-1, y-1]] if x-1 >= 0 and y-1 >= 0 else [0]
        state = np.array(state)
        #self.observations[self.agent_selection] = state

        reward = self.rewards[self.agent_selection]
        done = self.dones[self.agent_selection]

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        self.total_turns += 1

        self._accumulate_rewards()

    def render(self, mode = 'human'):
        print(self.board)

    def close(self):
        pass

    def seed(self, seed = None):
        #print(seed)
        randomizer, s = seeding.np_random(seed)
        self.randomizer = randomizer

    def observe(self, agent):
        x, y = self.agent_locations[agent]
        state = [x, y]
        state += [self.board[x-1, y]] if x-1 >= 0 else [0]
        state += [self.board[x-1, y+1]] if y+1 < self.y and x-1 >= 0 else [0]
        state += [self.board[x, y+1]] if y+1 < self.y else [0]
        state += [self.board[x+1, y+1]] if y+1 < self.y  and x+1 < self.x else [0]
        state += [self.board[x+1, y]] if x+1 < self.x else [0]
        state += [self.board[x+1, y-1]] if x+1 < self.x and y-1 >= 0 else [0]
        state += [self.board[x, y-1]] if y-1 >= 0 else [0]
        state += [self.board[x-1, y-1]] if x-1 >= 0 and y-1 >= 0 else [0]
        #print("agent ", agent)
        #print(state)
        return np.array(state)

    def initialize_board(self):
         self.board = np.array([[1] * self.y for _ in range(self.x)])

        # #set food
         for i in range(self.num_food):
             x = self.randomizer.randint(0,self.x)
             y = self.randomizer.randint(0,self.y)
             while (x,y) == (0,0) or self.board[x,y] == 2:
                 x = self.randomizer.randint(0,self.x)
                 y = self.randomizer.randint(0,self.y)
             self.board[x,y] = 2

         #set traps
         for i in range(self.num_hazards):
             x = self.randomizer.randint(0,self.x)
             y = self.randomizer.randint(0,self.y)
             while (x,y) == (0,0) or self.board[x,y] == 2 or self.board[x,y] == 3:
                 x = self.randomizer.randint(0,self.x)
                 y = self.randomizer.randint(0,self.y)
             self.board[x,y] = 3

         self.board[0,0] = 4
         #print(self.board)

    # Handles picking up food and dropping off food at the colony
    def pickup_food(self):
        (x,y) = self.agent_locations[self.agent_selection]
        # Picking up food
        if self.board[x,y] == 2:
            self.board[x,y] = 1
            #one food gives N energy
            self.rewards[self.agent_selection] += self.food_value + 1
            self.agent_energies[self.agent_selection] += self.food_value

    def move_left(self):
        agent = self.agent_selection
        (x,y) = self.agent_locations[agent]
        self.rewards[agent] -= 1
        self.agent_energies[agent]-= 1
        if self.agent_energies[agent]> 0 and y > 0:
            if self.num_agents_on(x,y) > 1:
                self.board[x,y] = 4
            else:
                self.board[x,y] = 1
            self.agent_locations[agent] = (x, y - 1)
            self.pickup_food()
            self.check_location()
            self.board[x,y-1] = 4

        self.check_energy()

    def move_right(self):
        agent = self.agent_selection
        (x,y) = self.agent_locations[agent]
        self.agent_energies[agent]-= 1
        self.rewards[agent] -= 1
        if self.agent_energies[agent] > 0 and y < self.y-1:
            if self.num_agents_on(x,y) > 1:
                self.board[x,y] = 4
            else:
                self.board[x,y] = 1
            self.agent_locations[agent] = (x, y + 1)
            self.pickup_food()
            self.check_location()
            self.board[x, y+1] = 4

        self.check_energy()

    def move_down(self):
        agent = self.agent_selection
        (x,y) = self.agent_locations[agent]
        self.agent_energies[agent]-= 1
        self.rewards[agent] -= 1
        if self.agent_energies[agent] > 0 and x < self.x - 1:
            if self.num_agents_on(x,y) > 1:
                self.board[x,y] = 4
            else:
                self.board[x,y] = 1
            self.agent_locations[agent] = (x + 1, y)
            self.pickup_food()
            self.check_location()
            self.board[x+1, y] = 4

        self.check_energy()

    def move_up(self):
        agent = self.agent_selection
        (x,y) = self.agent_locations[agent]
        self.agent_energies[agent]-= 1
        self.rewards[agent] -= 1
        if self.agent_energies[agent]> 0 and x > 0:
            if self.num_agents_on(x,y) > 1:
                self.board[x,y] = 4
            else:
                self.board[x,y] = 1
            self.agent_locations[agent] = (x - 1, y)
            self.pickup_food()
            self.check_location()
            self.board[x-1, y] = 4

        self.check_energy()

    #Gives the current agent an energy reduction punishment if it is standing in a hazard
    def check_location(self):
        (x,y) = self.agent_locations[self.agent_selection]
        if self.board[x, y] == 3:
            self.rewards[self.agent_selection] -= 1

    #Kills the current agent if it's energy has dipped below 0
    def check_energy(self):
        if self.agent_energies[self.agent_selection] < 0:
            self.dones[self.agent_selection]= True
            # self.rewards[self.agent_selection] -= 5

    #returns the number of agents that haven't dies yet
    def num_alive_agents(self):
        count = 0
        for agent in self.agents:
            if not self.dones[agent]:
                count += 1
        return count

    def num_agents_on(self, x, y):
        count = 0
        for agent in self.agents:
            if self.agent_locations[agent] == (x,y):
                count += 1
        return count

    def print_board(self):
        print("Agent " + str(self.agent_selection) + "s turn")
        print("locations: ", self.agent_locations)
        print("energies: ", self.agent_energies)
        print("Steps: " + str(self.steps))
        print(self.board)
