# District-Heating-Genetic-Algorithm
This is genetic algorithm (GA) for optimizing city district heating energy production. GA is a computational method for solving complex optimization problems and search problems. GA is suitable for optimizing energy systems and hundreds of scientific papers have reported of using it in problems related to energy field.

Code is programmed with Python. Python was chosen as a programming language, because it has a lot of interesting libraries for genetic algorithms and decent plotting abilities. Python is commonly used in data sciences.

Idea to this work has been developed based on criteria in Helsinki Energy Challenge. Helsinki Energy Challenge is an innovation competition organized by Helsinki city in 2020–2021. Main objective of the competition is to find solution how to replace coal power plants in Helsinki district heating system. Roughly 56 % of the energy is produced with coal. 

Advantage of GA is that method could applied to other cities by changing starting parameters. This way algorithm produces different result which are more suitable for local conditions. Method could be useful for solving many other optimization problems than district heating production.

Algortihm finds the optimized or "good enough" combination of power plants but also the ideal number of power plant units using GA. It includes simplified model to calculate fitness of different solutions. Only emissions, cost, and powerplant capacity of solution was considered.

Genetic Algorithm flow
1. Population
2. Fitness Calculation
3. Mating pool
4. Parents selection
5. Mating -> Crossover and mutation
6. Offspring
7. Return to step 1 or stop algorithm

Program prints results of the algorithm and plots development of fitness.

![image](https://user-images.githubusercontent.com/55585889/121179815-cdcf4900-c868-11eb-8f41-dbc87df05a80.png)

Printed result in terminal output

![image](https://user-images.githubusercontent.com/55585889/121180168-37e7ee00-c869-11eb-83c3-5e593cd86390.png)

Best number of power plants

![image](https://user-images.githubusercontent.com/55585889/121180199-40402900-c869-11eb-9bd8-d10aef8bb951.png)

Evolution of fitness in best combination of power plants

Principle of algorithm works very well in this code. However, simplified model of the energy system coded in it, is not complex enough that it would require GA to be solved. Powerplants with smallest sum of emission and cost parameters proved to be most successful in evolution. For future development more parameters should be included, so that advantages of GA would step in.
