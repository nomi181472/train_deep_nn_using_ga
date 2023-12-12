import gym
import random
import numpy as np
from numpy import *
import math
from statistics import median
import matplotlib.pyplot as plt
import matplotlib

from scipy import signal

env_space = gym.make('LunarLander-v2')
env_space.reset()

i = 0




class DeepNeuralNetwork():
    def __init__(self, inp_nodes, hidden_nodes_1, hidden_nodes_2, out_nodes):
        self.i_n = inp_nodes
        self.h_1 = hidden_nodes_1
        self.h_2 = hidden_nodes_2
        self.o_n = out_nodes
        self.func = [False, False, False, False]


    def set_activation_func(self, function):
        if function == 'sigmoid':
            self.func = [True, False, False, False]
        elif function == 'relu':
            self.func = [False, True, False, False]
        elif function == 'lrelu':
            self.func == [False, False, True, False]
        elif function == 'softmax':
            self.func = [False, False, False, True]
        else:
            print('Error: Mentioned', function,
                  'is not present in function list. Kindly choose among \'sigmoid\' for Sigmoid output, \'relu\' for ReLU outptu, \'lrelu\' for Leaky ReLU or \'softmax\' for Softmax function output')

    def fit(self, inp):
        self.i_l = np.asarray(inp).reshape(-1, 1)
        weight_ = self.i_w
        out = np.dot(weight_, inp)

        self.h_l_1 = self.Act(out).reshape(-1, 1)
        weight_ = self.h_1
        out = np.dot(weight_, self.h_l_1)

        self.h_l_2 = self.Act(out).reshape(-1, 1)
        weight_ = self.h_2
        out = np.dot(weight_, self.h_l_2)

        self.func = [False, False, False, True]
        self.o_l = self.Act(out).reshape(-1, 1)
        return np.argmax(self.o_l)


    def Act(self, _):
        if self.func[0] and not self.func[1] and not self.func[2] and not self.func[3]:
            print(self.func[3])
            return 1 / (1 + np.exp(-_))
        elif not self.func[0] and self.func[1] and not self.func[2] and not self.func[3]:
            return np.maximum(0, _)
        elif not self.func[0] and not self.func[1] and self.func[2] and not self.func[3]:
            return _ * self.alpha
        elif self.func[0] and self.func[1] and not self.func[2] and not self.func[3]:
            return np.clip(_, -1, 1)
        else:
            return np.exp(_ - np.max(_)) / np.sum(np.exp(_ - np.max(_)))

    def SetAlpha(self, alpha):
        try:
            if self.func == [False, False, True, False]:
                self.alpha = alpha
        except AttributeError:
            print('Activation function not defined. Set Alpha value after defining activation function')

    def SetWeights(self, i_w, h_1, h_2):
        self.i_w = i_w
        self.h_1 = h_1
        self.h_2 = h_2






class GeneticAlgorithm:

    def __init__(self, input_layer, h_n_1, h_n_2, out_layer):
        self.input_layer = input_layer
        self.h_n_1 = h_n_1
        self.h_n_2 = h_n_2
        self.out_layer = out_layer

    def to_weights(self):

        # tHe weight Initialization
        input_weight = np.random.randn(self.h_n_1, self.input_layer) * np.sqrt(2 / self.input_layer)
        hidden_layer_1 = np.random.randn(self.h_n_2, self.h_n_1) * np.sqrt(2 / self.h_n_1)
        hidden_layer_2 = np.random.randn(self.out_layer, self.h_n_2) * np.sqrt(2 / self.h_n_2)
        threshold = np.random.rand()
        return input_weight, hidden_layer_1, hidden_layer_2, threshold

    def mutate(self, child):
        k = random.randint(0, child.shape[1])
        des = random.randint(0, 10)
        if des <= 5:
            for i in range(k):
                limit = random.randint(0, child.shape[1])
                mutation = random.randint(-300, 300)
                child[0, limit] += mutation

        return child

    # def crossover(self, par_1, par_2):
    def crossover(self, parents,population,parent_pop):
        new_population = []
        for _ in range(population - parent_pop):
            child = []
            n1 = random.randint(0, len(parents))
            parent_1 = parents[n1]
            # del parents[n]
            n2 = random.randint(0, len(parents))
            while n2 == n1:
                n2 = random.randint(0, len(parents))
            parent_2 = parents[n2]
            # del parents[n]

            for i in range(parent_1.shape[1]):
                if i % 2 == 0:
                    child = np.append(child, parent_1[0, i])
                else:
                    child = np.append(child, parent_2[0, i])
            mutated_child = self.mutate(child.reshape(1, -1))
            new_population.append(mutated_child)

        return parents + new_population

    def to_evolve(self, score, gene,population,parent_pop,):
        n = random.randint(0, len(score))
        parents = []
        parent_pop = random.randint(10, 70)
        for i in range(parent_pop):
            loc_1 = score.index(max(score))
            score.remove(max(score))
            parents.append(gene[loc_1].reshape(1, -1))
            del gene[loc_1]

        return self.crossover(parents,population=population,parent_pop=parent_pop)

    def init_population(self, population, iteration_time,is_model_loaded=False,load_model_weights=(1,1,1)):
        population_award = []
        population_gene_pool = []
        for _ in range(population):
            input_weight, hidden_layer_1, hidden_layer_2, threshold = self.to_weights()
            if is_model_loaded:
                input_weight, hidden_layer_1, hidden_layer_2 = load_model_weights
            observation,_ = env_space.reset()
            award = 0
            for __ in range(iteration_time):
                # env_space.render()
                model = DeepNeuralNetwork(4, 4, 2, 1)
                model.SetWeights(input_weight, hidden_layer_1, hidden_layer_2)
                model.set_activation_func('relu')
                action = model.fit(observation)
                observation, reward, done, info ,_= env_space.step(action)
                award += reward
                #print(f"----------------------i={__}--Reward:{reward} isDone:{done} action: {action}-------------------")
                if done:
                    break
            population_award.append(award)
            chromosome = np.concatenate((input_weight.flatten(), hidden_layer_1.flatten(), hidden_layer_2.flatten()))
            population_gene_pool.append(chromosome)
        return population_award, population_gene_pool



class Agent():
    def __init__(self,parent_pop = 0,
        generations = 70,
        population = 80,
        iteration_time = 700,
        i_layer = 8,
        h_l_1 = 36,
        h_l_2 = 36,
        o_layer = 4,

                 ):
        self.parent_pop = parent_pop
        self.generations = generations
        self.population = population
        self.iteration_time = iteration_time
        self.i_layer = i_layer
        self.h_l_1 = h_l_1
        self.h_l_2 = h_l_2
        self.o_layer = o_layer
        self.ga_model = GeneticAlgorithm(i_layer, h_l_1, h_l_2, o_layer)
        self.model = DeepNeuralNetwork(4, 7, 4, 1)
        self.model.set_activation_func('relu')
        self.load_model=(0,0,0)
        self.best_awards_gen = []
        self.med_awards_gen = []
        self.avg_awards_gen = []
        self.it = []
        self.PID = []
        self.prev = 0
        self.iterations = []
        self.iterations_2 = []
        self.avg = -99
        self.current_award = -9999
        self.generation_awards = []
        self.i = 0
        self.diff = 0

    def to_model_weights(self,_):
        X = self.i_layer * self.h_l_1
        Y = X + self.h_l_2 * self.h_l_1
        Z = Y + self.h_l_2 * self.o_layer

        input_weight, hidden_layer_1, hidden_layer_2 = _[0, 0:X], _[0, X:Y], _[0, Y:Z]

        input_weight = input_weight.reshape(-1, self.i_layer)
        # input_weight = (input_weight - mean(input_weight))/std(input_weight)
        hidden_layer_1 = hidden_layer_1.reshape(-1, self.h_l_1)
        # hidden_layer_1 = (hidden_layer_1 - mean(hidden_layer_1))/std(hidden_layer_1)
        hidden_layer_2 = hidden_layer_2.reshape(-1, self.h_l_2)
        # hidden_layer_2 = (hidden_layer_2 - mean(hidden_layer_2))/std(hidden_layer_2)
        return input_weight, hidden_layer_1, hidden_layer_2

    def optimize_agent(self,weight_path=""):
        is_model_available=len(weight_path)>0
        if  is_model_available:
            weights = np.load(weight_path, allow_pickle=True)
            self.load_model= self.to_model_weights(weights.reshape(1, -1))

        self.pop_award, self.pop_gene = self.ga_model.init_population(self.population, self.iteration_time, is_model_available,self.load_model)

        while True:
            new_population = self.ga_model.to_evolve(self.pop_award, self.pop_gene,population=self.population,parent_pop=self.parent_pop)
            self.pop_award = []
            self.pop_gene = []
            for _ in new_population:
                observation, extra = env_space.reset()
                input_weight, hidden_1, hidden_2 = self.to_model_weights(_)
                self.model.SetWeights(input_weight, hidden_1, hidden_2)

                award = 0
                for x in range(self.iteration_time):
                    # env_space.render()
                    self.model.set_activation_func('relu')
                    action = self.model.fit(observation)

                    observation, reward, done, info, _ = env_space.step(int(action))
                    award += reward
                    #print(
                    #    f"----------------------i={x}--Reward:{reward} isDone:{done} action: {int(action)}-------------------")
                    if done:
                        break
                self.PID.append(award)
                self.pop_award.append(award)

                chromosome = np.concatenate((input_weight.flatten(), hidden_1.flatten(), hidden_2.flatten()))
                self.pop_gene.append(chromosome)

            self.avg_awards_gen = np.append(self.avg_awards_gen, np.average(self.PID))
            self.best_awards_gen = np.append(self.best_awards_gen, np.amax(self.pop_award))  # Store Maximum of Each Generation
            self.med_awards_gen = np.append(self.med_awards_gen, np.median(self.PID))

            if np.median(self.PID) >= self.current_award:
                self.current_award = max(self.current_award, np.median(self.PID))
                np.save('Lunar_MED2', self.pop_gene[self.pop_award.index(max(self.pop_award))])
            if max(self.pop_award) > self.prev:
                np.save('Lunar_BEST2', self.pop_gene[self.pop_award.index(max(self.pop_award))])
                self.prev = max(self.pop_award)
            self.i += 1
            print('[Generation: %3d] [current record:%5d]  [Median Score:%5d] [Top Score: %3d] [History: %3d]' % (
                self.i, round(self.current_award, 2), round(np.median(self.PID), 2), np.amax(self.pop_award),self. prev))
            if self.current_award > 320 or self.i >self.generations:
                break
    def plot_results(self):
        t = np.linspace(0, agent.i, agent.i)
        plt.plot(t, self.best_awards_gen, 'r', label='Best Fitness Scores')
        plt.title('Rewards vs Generation Plot')
        plt.xlabel('Generations')
        plt.ylabel('Rewards')
        plt.grid()
        print('Average', np.mean(agent.best_awards_gen), 0)
        plt.legend(loc='lower right')
        plt.show()

    def test_agent(self,weight_path="",repeat_iteration=100,render_model="",):
        weights=np.load(weight_path,allow_pickle=True)

        #weights= self.to_model_weights(weights.reshape(1,-1))
        #self.pop_award, self.pop_gene = self.ga_model.init_population(self.population, self.iteration_time, True, weights)
        #new_population = self.ga_model.to_evolve(self.pop_award, self.pop_gene, population=self.population,parent_pop=self.parent_pop)

        max = 0
        for i in range(repeat_iteration):
            reward = 0
            input, hidden1, hidden2 = self.to_model_weights(weights.reshape(1,-1))

            self.model.SetWeights(input, hidden1, hidden2)

            env_space = gym.make('LunarLander-v2')
            if render_model == "human":

                env_space.close()
                env_space = gym.make('LunarLander-v2', render_mode="human")
            #best 24
            observation, extra = env_space.reset(seed=i)
            for _ in range(self.iteration_time):
                self.model.set_activation_func('relu')
                action = self.model.fit(observation)

                observation, current_reward, terminated, truncated, info = env_space.step(int(action))
                reward += current_reward
                # print(f"----------------------i={x}--Reward:{reward} isDone:{done} action: {int(action)}-------------------")
                if terminated or truncated:

                    break
            env_space.close()
            if reward> max:
                max=reward
                print(f"i:{i},max:{max}")
            print(f"{i}--------Reward={reward}")




# for gen in range(generations):
agent=Agent(generations=10)
path="Lunar_BEST2_330.npy"
agent.optimize_agent(path)
agent.plot_results()
#env_space.close()
#agent.test_agent(path,10,)
env_space.close()




