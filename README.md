# ants
A custom pettingzoo environment that simulates a very simple ant colony.

This environment is a basic ant colony simulation. There are by default 10 ants.

**The Board**

The environment is a basic grid world with each cell having 4 possible options:
- 0: Boundary (can't enter these cells)
- 1: Empty
- 2: Food (food is collected if this cell is entered)
- 3: Hazard (ants lose 1 energy and get a negative reward for entering this cell)
- 4: Ant (one or more ants are in this cell)

The board is randomly generated after every reset, given the specific number of
food pieces and hazards specified.

**Actions**

Each time step, ants are able to move one cell up, down, left, or right.
Moving takes 1 energy.
Actions:
- 0: move left
- 1: move up
- 2: move right
- 3: move down

**Observations**

An observation for an ant is an array representing the contents of the cells
directly around them in the order starting with directly above the agent, then
moving clockwise around the agent for a total of 8 values.

Ants use 1 energy to move and collect energy (specified by the food_value parameter)
every time they move into a food cell. If an ant runs out of energy at any point,
they die. The simulation ends when all ants have died.  
