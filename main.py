from GeneticAlgorithm import GeneticAlgorithm

if __name__ == '__main__':

    # Вызов 1: Стандартные параметры, n=2 (по умолчанию) + вывод графиков раз в 20 поколений
    print("\nВызов 1: Стандартные параметры, n=2")
    print("Параметры: population_size=100, crossover_rate=0.7, mutation_rate=0.15, n=2")
    genAlg = GeneticAlgorithm(population_size=100, mutation_rate=0.15, crossover_rate=0.7, parameter_n=2)
    genAlg.run_algorithm(print_interval=20,print_best_interval=20, stop_when_same_results=True, max_generations=100, show_plots=True)

    # Вызов 2: Увеличенное количество особей в популяции, n=2
    print("\nВызов 2: Увеличенное количество особей в популяции, n=2")
    print("Параметры: population_size=200, crossover_rate=0.7, mutation_rate=0.15, n=2")
    genAlg = GeneticAlgorithm(population_size=200, mutation_rate=0.15, crossover_rate=0.7, parameter_n=2)
    genAlg.run_algorithm(print_interval=20, print_best_interval=20, stop_when_same_results=True, max_generations=100, show_plots=False)

    # Вызов 3: Увеличенная вероятность кроссинговера, n=2
    print("\nВызов 3: Увеличенная вероятность кроссинговера, n=2")
    print("Параметры: population_size=100, crossover_rate=0.9, mutation_rate=0.15, n=2")
    genAlg = GeneticAlgorithm(population_size=100, mutation_rate=0.15, crossover_rate=0.9, parameter_n=2)
    genAlg.run_algorithm(print_interval=20, print_best_interval=20, stop_when_same_results=True, max_generations=100, show_plots=False)

    # Вызов 4: Увеличенная вероятность мутации, n=2
    print("\nВызов 4: Увеличенная вероятность мутации, n=2")
    print("Параметры: population_size=100, crossover_rate=0.7, mutation_rate=0.3, n=2")
    genAlg = GeneticAlgorithm(population_size=100, mutation_rate=0.3, crossover_rate=0.7, parameter_n=2)
    genAlg.run_algorithm(print_interval=20, print_best_interval=20,stop_when_same_results=True, max_generations=100, show_plots=False)

    # Вызов 5: Уменьшенная вероятность кроссинговера и мутации, n=2
    print("\nВызов 5: Уменьшенная вероятность кроссинговера и мутации, n=2")
    print("Параметры: population_size=100, crossover_rate=0.5, mutation_rate=0.05, n=2")
    genAlg = GeneticAlgorithm(population_size=100, mutation_rate=0.05, crossover_rate=0.5, parameter_n=2)
    genAlg.run_algorithm(print_interval=20, print_best_interval=20,stop_when_same_results=True, max_generations=100, show_plots=False)

    # Вызов 6: Увеличенное количество особей и высокая вероятность мутации, n=2
    print("\nВызов 6: Увеличенное количество особей и высокая вероятность мутации, n=2")
    print("Параметры: population_size=200, crossover_rate=0.7, mutation_rate=0.3, n=2")
    genAlg = GeneticAlgorithm(population_size=200, mutation_rate=0.3, crossover_rate=0.7, parameter_n=2)
    genAlg.run_algorithm(print_interval=20, print_best_interval=20,stop_when_same_results=True, max_generations=100, show_plots=False)

    # --- Эксперименты с n = 3 ---
    
    # Вызов 7: Стандартные параметры, n=3
    print("\nВызов 7: Стандартные параметры, n=3")
    print("Параметры: population_size=100, crossover_rate=0.7, mutation_rate=0.15, n=3")
    genAlg = GeneticAlgorithm(population_size=100, mutation_rate=0.15, crossover_rate=0.7, parameter_n=3)
    genAlg.run_algorithm(print_interval=20, print_best_interval=20,stop_when_same_results=True, max_generations=100, show_plots=False)

    # Вызов 8: Увеличенное количество особей в популяции, n=3
    print("\nВызов 8: Увеличенное количество особей в популяции, n=3")
    print("Параметры: population_size=200, crossover_rate=0.7, mutation_rate=0.15, n=3")
    genAlg = GeneticAlgorithm(population_size=200, mutation_rate=0.15, crossover_rate=0.7, parameter_n=3)
    genAlg.run_algorithm(print_interval=20, print_best_interval=20, stop_when_same_results=True, max_generations=100, show_plots=False)

    # Вызов 9: Увеличенная вероятность кроссинговера, n=3
    print("\nВызов 9: Увеличенная вероятность кроссинговера, n=3")
    print("Параметры: population_size=100, crossover_rate=0.9, mutation_rate=0.15, n=3")
    genAlg = GeneticAlgorithm(population_size=100, mutation_rate=0.15, crossover_rate=0.9, parameter_n=3)
    genAlg.run_algorithm(print_interval=20, print_best_interval=20,stop_when_same_results=True, max_generations=100, show_plots=False)

    # Вызов 10: Увеличенная вероятность мутации, n=3
    print("\nВызов 10: Увеличенная вероятность мутации, n=3")
    print("Параметры: population_size=100, crossover_rate=0.7, mutation_rate=0.3, n=3")
    genAlg = GeneticAlgorithm(population_size=100, mutation_rate=0.3, crossover_rate=0.7, parameter_n=3)
    genAlg.run_algorithm(print_interval=20, print_best_interval=20,stop_when_same_results=True, max_generations=100, show_plots=False)

    # Вызов 11: Уменьшенная вероятность кроссинговера и мутации, n=3
    print("\nВызов 11: Уменьшенная вероятность кроссинговера и мутации, n=3")
    print("Параметры: population_size=100, crossover_rate=0.5, mutation_rate=0.05, n=3")
    genAlg = GeneticAlgorithm(population_size=100, mutation_rate=0.05, crossover_rate=0.5, parameter_n=3)
    genAlg.run_algorithm(print_interval=20, print_best_interval=20,stop_when_same_results=True, max_generations=100, show_plots=False)

    # Вызов 12: Увеличенное количество особей и высокая вероятность мутации, n=3
    print("\nВызов 12: Увеличенное количество особей и высокая вероятность мутации, n=3")
    print("Параметры: population_size=200, crossover_rate=0.7, mutation_rate=0.3, n=3")
    genAlg = GeneticAlgorithm(population_size=200, mutation_rate=0.3, crossover_rate=0.7, parameter_n=3)
    genAlg.run_algorithm(print_interval=20, print_best_interval=20,stop_when_same_results=True, max_generations=100, show_plots=False)
