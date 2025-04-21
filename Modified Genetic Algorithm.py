import random
import joblib
import numpy as np
import pandas as pd

model1 = joblib.load("classify_ANN.pkl")
model2 = joblib.load("regress_ANN.pkl")

def f1(x):
    Y = [0, 0, 0, 0, 0]
    x_values = [x[0], x[1], x[2], x[3], x[4]]  # 仅对第一个函数对象调用
    Y[0] = (x_values[0] - 25) / 50
    Y[1] = (x_values[1] - 2) / 8
    Y[2] = (x_values[2] - 2) / 8
    Y[3] = (x_values[3] - 100) / 200
    Y[4] = (x_values[4] - 20) / 88
    return model1.predict(np.array(Y).reshape(1, -1))[0]

def f2(x):
    Y = [0, 0, 0, 0, 0]
    x_values = [x[0], x[1], x[2], x[3], x[4]]  # 仅对第一个函数对象调用
    Y[0] = (x_values[0] - 25) / 50
    Y[1] = (x_values[1] - 2) / 8
    Y[2] = (x_values[2] - 2) / 8
    Y[3] = (x_values[3] - 100) / 200
    Y[4] = (x_values[4] - 20) / 88
    return model2.predict(np.array(Y).reshape(1, -1))[0]


# 生成第一代样本
def generate_population(n):
    population = []
    for _ in range(n):
        attr1 = random.uniform(25, 50)
        attr2 = random.uniform(2, 10)
        attr3 = random.uniform(2, 10)
        attr4 = random.uniform(100, 300)
        attr5 = random.uniform(100, 100)
        individual = [attr1, attr2, attr3, attr4, attr5]
        population.append(individual)
    return population


# 基因均匀交叉
def crossover(parent1, parent2):
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        if random.random() < 0.5:  # 以0.5的概率选择父代1的特征
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2


# 基因变异
def mutate(individual, mutation_rate=0.2):

    # 定义每个基因的取值范围
    bounds = [(25, 75), (2, 10), (2, 10), (100, 300), (100, 100)]

    # 进行基因变异操作
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(bounds[i][0], bounds[i][1])
    return individual


# 使用交叉+变异函数生成n个子代样本
def generate_samples(population, sample_size):
    samples = []
    while len(samples) < sample_size-len(population):
        for i in range(0, len(population), 2):
            if i + 1 < len(population):
                child1, child2 = crossover(population[i], population[i+1])
                child1 = mutate(child1)
                child2 = mutate(child2)
                samples.append(child1)
                if len(samples) < sample_size-len(population):
                    samples.append(child2)
                if len(samples) >= sample_size-len(population):
                    break       # 确保循环正确停止
    combined_population = population + samples
    return combined_population







ds = 2    # 损伤程度
m = 100   # 迭代次数
n = 500   # 样本个数

likelihood_ds = []
best_individuals = []

for generation in range(m):

    if generation == 0:
        population = generate_population(n)
    else:
        population = generated_samples

    # 对个体进行评估
    selected_child = []
    Damage_state = [0] * n
    Peak_impact_force = [0] * n
    for i in range(n):

        Damage_state[i] = f1(population[i])

        if Damage_state[i] == ds:
            selected_child.append(population[i])

    likelihood_ds.append(Damage_state.count(2)/len(Damage_state))


    # 将两个列表中的元素合并成一个新的列表
    # 根据第六列（即每个子列表的最后一个元素）对合并后的列表进行排序
    # 只保留前50%的list
    combined_list = [0] * len(selected_child)

    for i in range(len(selected_child)):
        combined_list[i] = selected_child[i] + [f2(population[i])]

    sorted_list = sorted(combined_list, key=lambda x: x[5])
    final_list = [item[:5] for item in sorted_list[:int(len(selected_child)/2)]]

    # 将1~m-1代中最优个体写入best_individuals
    best_individuals.append(sorted_list[0])

    # 生成n-len(selected_child)个样本,并和selected_child组合成n个样本
    generated_samples = generate_samples(final_list, n)


    # 输出每代的最优解和进化过程
    if generation == m-1:
        combined_generated_samples = [0] * len(generated_samples)
        for i in range(len(generated_samples)):
            combined_generated_samples[i] = generated_samples[i] + [f1(generated_samples[i])] + [
                f2(generated_samples[i])]

        best_individuals_list = pd.DataFrame(best_individuals)
        best_individuals_list.to_excel('Modified_GA_best_individuals.xlsx', index=False)

        likelihood_ds_list = pd.DataFrame(likelihood_ds)
        likelihood_ds_list.to_excel('Modified_GA_likelihood_ds.xlsx', index=False)