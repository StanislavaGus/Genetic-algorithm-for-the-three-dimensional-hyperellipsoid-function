import numpy as np
import random
import matplotlib.pyplot as plt
import time

class GeneticAlgorithm:

    def __init__(self, interval=(-5.12, 5.12), population_size=100, mutation_rate=0.15,
                 crossover_rate=0.7, parameter_n=2):
        self.interval = interval  # Интервал поиска для всех осей
        self.population_size = population_size  # Размер популяции
        self.mutation_rate = mutation_rate  # Вероятность мутации
        self.crossover_rate = crossover_rate  # Вероятность кроссовера
        self.parameter_n = parameter_n  # Параметр для SBX кроссовера

        self.population = None
        self.population_fitness = None
        self.tmt_population = None

    def hyper_ellipsoid_function(self, x1, x2):
        """ Функция гиперэллипсоида для двух переменных """
        return 5 * (1) * (x1 ** 2) + 5 * (2) * (x2 ** 2)

    def _generate_population(self):
        """ Генерация популяции в двумерном пространстве (x1 и x2) """
        self.population = [(random.uniform(self.interval[0], self.interval[1]),
                            random.uniform(self.interval[0], self.interval[1])) for _ in range(self.population_size)]

    def _count_fitness_function(self):
        """ Вычисление значения фитнес-функции для популяции """
        self.population_fitness = [self.hyper_ellipsoid_function(ind[0], ind[1]) for ind in self.population]

    def _tournament_selection(self):
        """ Отбор родителей на основе минимальной фитнес-функции в случайной паре """
        idx1, idx2 = random.sample(range(self.population_size), 2)
        if self.population_fitness[idx1] < self.population_fitness[idx2]:
            return self.population[idx1]
        else:
            return self.population[idx2]

    def _select_parents(self):
        """ Добавляем выбранных родителей в тмт популяцию с использованием турнирного отбора """
        self.tmt_population = []
        for _ in range(self.population_size):
            selected_parent = self._tournament_selection()
            self.tmt_population.append(selected_parent)

    def sbx_crossover(self, parent1, parent2):
        """ Simulated Binary Crossover (SBX) """
        child1 = []
        child2 = []
        for i in range(len(parent1)):
            u = random.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (self.parameter_n + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (self.parameter_n + 1))

            child1_value = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            child2_value = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])

            child1.append(np.clip(child1_value, self.interval[0], self.interval[1]))
            child2.append(np.clip(child2_value, self.interval[0], self.interval[1]))

        return child1, child2

    def arithmetic_crossover(self, parent1, parent2):
        """ Arithmetic crossover """
        child1 = []
        child2 = []
        for i in range(len(parent1)):
            w = random.uniform(0, 1)
            child1_value = w * parent1[i] + (1 - w) * parent2[i]
            child2_value = w * parent2[i] + (1 - w) * parent1[i]

            child1.append(np.clip(child1_value, self.interval[0], self.interval[1]))
            child2.append(np.clip(child2_value, self.interval[0], self.interval[1]))

        return child1, child2

    def crossover(self, parent1, parent2):
        """ Выполняем кроссинговер с проверкой вероятности и выбором метода """
        if random.random() <= self.crossover_rate:
            if random.random() < 0.5:
                child1, child2 = self.sbx_crossover(parent1, parent2)
            else:
                child1, child2 = self.arithmetic_crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2

        return child1, child2

    # остальной код остается тем же


    def mutate_population(self, population):
        """ Мутация популяции """
        mutated_population = []
        for individual in population:
            mutated_individual = []
            for gene in individual:
                if random.random() <= self.mutation_rate:
                    mutated_gene = random.uniform(self.interval[0], self.interval[1])
                else:
                    mutated_gene = gene
                mutated_individual.append(mutated_gene)
            mutated_population.append(tuple(mutated_individual))
        return mutated_population

    def _create_new_generation(self):
        """ Формирование нового поколения """
        new_population = []
        self._select_parents()

        while len(new_population) < self.population_size:
            parent1 = random.choice(self.tmt_population)
            parent2 = random.choice(self.tmt_population)
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(child1)
            new_population.append(child2)

        new_population = new_population[:self.population_size]
        new_population = self.mutate_population(new_population)
        self.population = new_population
        self._count_fitness_function()

    def draw_3d_plot(self, best_solution=None, highlight_best=False, generation=None):
        """ Построение 3D графика с возможностью указания поколения """
        x1 = np.linspace(self.interval[0], self.interval[1], 100)
        x2 = np.linspace(self.interval[0], self.interval[1], 100)
        X1, X2 = np.meshgrid(x1, x2)
        Z = self.hyper_ellipsoid_function(X1, X2)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)

        # Отмечаем точки популяции
        population_x1 = [ind[0] for ind in self.population]
        population_x2 = [ind[1] for ind in self.population]
        population_fitness = self.population_fitness

        # Синие точки для популяции
        ax.scatter(population_x1, population_x2, population_fitness, color='blue', label='Population', s=40, zorder=1)

        if highlight_best and best_solution is not None:
            # Красная непрозрачная точка для лучшего решения (поверх остальных точек)
            best_fitness = self.hyper_ellipsoid_function(best_solution[0], best_solution[1])
            ax.scatter(best_solution[0], best_solution[1], best_fitness, color='red', label='Best Solution', s=100,
                       zorder=2)  # Увеличенный размер и непрозрачная

        # Добавляем номер поколения на график
        if generation is not None:
            ax.text2D(0.05, 0.95, f"Generation: {generation}", transform=ax.transAxes)

        ax.set_title('Hyper-Ellipsoid Function with Population Points')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('f(x1, x2)')
        ax.legend()
        plt.show()

    def run_algorithm(self, print_interval=10, print_best_interval=1, stop_when_same_results=True, max_generations=101,
                      show_plots=True):
        """ Основная функция выполнения генетического алгоритма """
        start_time = time.time()  # Записываем время начала выполнения

        self.num_generations = max_generations
        best_solutions = []  # Для хранения лучших решений каждого поколения

        # Генерация начальной популяции
        self._generate_population()
        self._count_fitness_function()

        # Печатаем и выводим график для начальной популяции
        print("Initial population:")
        if show_plots:
            self.draw_3d_plot(generation=0)  # График для начальной популяции (без лучшего решения)

        for generation in range(1, self.num_generations + 1):
            self._create_new_generation()

            # Находим лучшее решение в текущем поколении
            best_fitness = min(self.population_fitness)
            best_solution = self.population[self.population_fitness.index(best_fitness)]
            best_solutions.append(best_solution)

            # Печатаем лучшее решение для текущего поколения через заданное число поколений
            if generation % print_best_interval == 0:
                print(f"Generation {generation}, Best solution: {best_solution}, Fitness: {best_fitness}")

            # Печатаем и рисуем график на заданных интервалах, если включен вывод графиков
            if show_plots and generation % print_interval == 0:
                self.draw_3d_plot(best_solution, highlight_best=True, generation=generation)

            # Проверяем критерий остановки по совпадению лучших решений
            if stop_when_same_results and generation >= 5:  # Проверяем последние 5 поколения
                last_5_solutions = best_solutions[-5:]
                if all(last_5_solutions[0] == sol for sol in last_5_solutions):
                    print(f"Stopping early: Best solution has remained the same for the last 5 generations.")
                    break

        # Печать последнего поколения
        print(f"Final Generation {generation}, Best solution: {best_solution}, Fitness: {best_fitness}")
        if show_plots:
            self.draw_3d_plot(best_solution, highlight_best=True, generation=generation)

        # Печатаем время выполнения
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")