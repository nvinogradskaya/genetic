import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem, QInputDialog, QMessageBox

# Функция для построения графика
def plot_function():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(-2, 3, 100)
    y = np.linspace(-2, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = (X - 2) ** 4 + (X - 2 * Y) ** 2

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.view_init(elev=5, azim=115)
    ax.dist = 7
    plt.show()

# Определение целевой функции
def function(x, y):
    return (x - 2) ** 4 + (x - 2 * y) ** 2

class GeneticAlgorithmApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генетический алгоритм")
        self.setGeometry(100, 100, 800, 500)

        # Количество хромосом
        self.label_chromosomes = QLabel("Количество хромосом:", self)
        self.label_chromosomes.setGeometry(20, 20, 200, 20)
        self.text_chromosomes = QLineEdit(self)
        self.text_chromosomes.setGeometry(250, 20, 150, 20)

        # Вероятность мутации
        self.label_mutation = QLabel("Вероятность мутации:", self)
        self.label_mutation.setGeometry(20, 60, 200, 20)
        self.text_mutation = QLineEdit(self)
        self.text_mutation.setGeometry(250, 60, 150, 20)

        # Вероятность кроссинговера
        self.label_crossover = QLabel("Вероятность кроссинговера:", self)
        self.label_crossover.setGeometry(20, 100, 200, 20)
        self.text_crossover = QLineEdit(self)
        self.text_crossover.setGeometry(250, 100, 150, 20)

        # Максимальное значение гена
        self.label_max_gene = QLabel("Максимальное значение гена:", self)
        self.label_max_gene.setGeometry(20, 140, 200, 20)
        self.text_max_gene = QLineEdit(self)
        self.text_max_gene.setGeometry(250, 140, 150, 20)

        # Минимальное значение гена
        self.label_min_gene = QLabel("Минимальное значение гена:", self)
        self.label_min_gene.setGeometry(20, 180, 200, 20)
        self.text_min_gene = QLineEdit(self)
        self.text_min_gene.setGeometry(250, 180, 150, 20)

        # Количество поколений
        self.label_generations = QLabel("Количество поколений:", self)
        self.label_generations.setGeometry(20, 220, 200, 20)
        self.text_generations = QLineEdit(self)
        self.text_generations.setGeometry(250, 220, 150, 20)

        # Кнопка "Старт"
        self.start_button = QPushButton("Старт", self)
        self.start_button.setGeometry(200, 260, 200, 30)
        self.start_button.clicked.connect(self.run_genetic_algorithm)

        # Таблица с результатами
        self.table = QTableWidget(self)
        self.table.setGeometry(420, 20, 560, 450)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Ген 1", "Ген 2", "Результат"])
        self.table.setColumnWidth(0, 180)
        self.table.setColumnWidth(1, 180)
        self.table.setColumnWidth(2, 180)

        # Текстовое поле с результатом
        self.result_text = QTextEdit(self)
        self.result_text.setGeometry(20, 310, 400, 160)
        self.result_text.setReadOnly(True)

        # Кнопка "Показать поколение"
        self.show_generation_button = QPushButton("Показать поколение", self)
        self.show_generation_button.setGeometry(20, 260, 160, 30)
        self.show_generation_button.clicked.connect(self.show_generation)

        # Инициализация списка истории популяции
        self.population_history = []

    def run_genetic_algorithm(self):
        num_chromosomes = int(self.text_chromosomes.text())
        mutation_rate = float(self.text_mutation.text())
        crossover_rate = float(self.text_crossover.text())
        max_gene_value = float(self.text_max_gene.text())
        min_gene_value = float(self.text_min_gene.text())
        num_generations = int(self.text_generations.text())

        # Генерация начальной популяции
        population = self.generate_population(num_chromosomes, max_gene_value, min_gene_value)
        # Отображение начальной популяции в таблице
        self.update_table(population)
        self.population_history.append(population)
        self.update_table(self.population_history[0])

        # Запуск генетического алгоритма
        best_solution, best_fitness = self.genetic_algorithm(population, mutation_rate, crossover_rate, num_generations, min_gene_value, max_gene_value)

        result = f"Лучшее решение: {best_solution}\nЛучшее значение функции: {best_fitness}"
        self.result_text.setPlainText(result)

    def generate_population(self, size, max_gene_value, min_gene_value):
        population = []
        for _ in range(size):
            # Генерация случайных значений для каждого гена особи
            population.append((random.uniform(min_gene_value, max_gene_value), random.uniform(min_gene_value, max_gene_value)))
        return population

    def fitness_function(self, x, y):
        return (x - 2) ** 4 + (x - 2 * y) ** 2

    def genetic_algorithm(self, population, mutation_rate, crossover_rate, generations, min_gene_value, max_gene_value):
        for _ in range(generations):
            fitness_scores = self.evaluate_population(population)

            # Находим лучшую особь в популяции
            best_fitness = min(fitness_scores)
            best_individual = population[fitness_scores.index(best_fitness)]

            # Выбираем родителей и генерируем потомство
            new_population = []
            for _ in range(len(population) // 2):
                parents = self.select_parents(population, fitness_scores)
                offspring = self.crossover(parents, crossover_rate)
                mutated_offspring = self.mutate(offspring, mutation_rate, min_gene_value, max_gene_value)
                new_population.extend([offspring, mutated_offspring])

            population = new_population
            self.population_history.append(population)

        # Находим лучшую особь после всех поколений
        fitness_scores = self.evaluate_population(population)
        best_fitness = min(fitness_scores)
        best_individual = population[fitness_scores.index(best_fitness)]

        return best_individual, best_fitness

    def update_table(self, population):
        self.table.clearContents()
        self.table.setRowCount(len(population))

        for row, individual in enumerate(population):
            gene1_item = QTableWidgetItem(str(individual[0]))
            gene2_item = QTableWidgetItem(str(individual[1]))
            result_item = QTableWidgetItem(str(self.fitness_function(*individual)))

            self.table.setItem(row, 0, gene1_item)
            self.table.setItem(row, 1, gene2_item)
            self.table.setItem(row, 2, result_item)

        self.table.resizeColumnsToContents()

    def show_generation(self):
        generation_number, ok = QInputDialog.getInt(self, "Введите номер поколения", "Номер поколения:")
        if ok:
            if generation_number >= 0 and generation_number < len(self.population_history):
                population = self.population_history[generation_number]
                self.update_table(population)
            else:
                QMessageBox.warning(self, "Неверный номер поколения", "Пожалуйста, введите корректный номер поколения.")

    def evaluate_population(self, population):
        fitness_scores = []
        for individual in population:
            fitness_scores.append(self.fitness_function(*individual))
        return fitness_scores

    def select_parents(self, population, fitness_scores):
        parents = []
        for _ in range(2):
            # Случайный выбор 5 особей для турнира
            tournament = random.choices(list(enumerate(population)), k=5)
            tournament_fitness = [fitness_scores[i] for i, _ in tournament]
            winner_index = tournament_fitness.index(min(tournament_fitness))
            winner = tournament[winner_index][1]
            parents.append(winner)
        return parents

    def crossover(self, parents, crossover_rate):
        offspring = []
        for i in range(len(parents[0])):
            if random.random() < crossover_rate:
                # Случайная точка разреза для скрещивания + Создание потомков путем комбинирования генов родителей
                offspring_value = (parents[0][i] + parents[1][i]) / 2.0
            else:
                offspring_value = parents[0][i]
            offspring.append(offspring_value)
        return tuple(offspring)

    def mutate(self, offspring, mutation_rate, min_gene_value, max_gene_value):
        mutated_offspring = []
        for gene in offspring:
            if random.random() < mutation_rate:
                # Выбор случайного гена для мутации + Мутация выбранного гена
                mutated_gene = random.uniform(min_gene_value, max_gene_value)
            else:
                mutated_gene = gene
            mutated_offspring.append(mutated_gene)
        return tuple(mutated_offspring)


if __name__ == "__main__":
    plot_function()
    app = QApplication([])
    window = GeneticAlgorithmApp()
    window.show()
    sys.exit(app.exec())
