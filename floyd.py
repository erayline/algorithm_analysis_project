import time
import copy
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

def floyd_warshall(dist_matrix):
    n = len(dist_matrix)
    for i in range(n):
        dist_matrix[i][i] = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist_matrix[i][k] + dist_matrix[k][j] < dist_matrix[i][j]:
                    dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]
    for i in range(n):
        if dist_matrix[i][i] < 0:
            raise ValueError("Negative cycle detected in graph!")
    return dist_matrix

def generate_random_graph(n, edge_prob=0.8, max_weight=10):
    matrix = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and random.random() < edge_prob:
                matrix[i][j] = random.randint(1, max_weight)
    for i in range(n):
        matrix[i][i] = 0
    return matrix

def print_matrix(matrix, label="Matrix"):
    print(f"\n{label}:")
    n = len(matrix)
    col_widths = [max(len(str(matrix[i][j])) for i in range(n)) for j in range(n)]
    for i in range(n):
        row_str = " ".join(f"{matrix[i][j]:>{col_widths[j]}}" for j in range(n))
        print(row_str)

def visualize_graph(dist_matrix, title="Graph Visualization", show_shortest=True):
    G = nx.DiGraph()
    n = len(dist_matrix)
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            if dist_matrix[i][j] < float('inf') and i != j:
                G.add_edge(i, j, weight=dist_matrix[i][j])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, arrows=True)
    if show_shortest:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.axis('off')
    plt.show()

def test_floyd_warshall():
    print("Floyd-Warshall Algorithm Tests with Visualization")
    print("=" * 50)
    
    input_matrix1 = [
        [0, 5, float('inf')],
        [float('inf'), 0, 3],
        [float('inf'), float('inf'), 0]
    ]
    expected_output1 = [
        [0, 5, 8],
        [float('inf'), 0, 3],
        [float('inf'), float('inf'), 0]
    ]
    
    print("\nTest Case 1: Small Manual Graph")
    print("Input:")
    print_matrix(input_matrix1, "Input Distance Matrix")
    visualize_graph(input_matrix1, "Input Graph (Test 1)")
    
    result1 = copy.deepcopy(input_matrix1)
    tracker = EmissionsTracker()
    tracker.start()
    start_time = time.time()
    result1 = floyd_warshall(result1)
    end_time = time.time()
    tracker.stop()
    
    print_matrix(result1, "Output Distance Matrix")
    print(f"Execution Time: {end_time - start_time:.6f} seconds")
    print(f"Energy Consumed: {tracker.final_emissions_data.energy_consumed} kWh")
    print(f"CO2 Emissions: {tracker.final_emissions} kg")
    
    match = all(math.isclose(result1[i][j], expected_output1[i][j], abs_tol=1e-9)
                for i in range(len(result1)) for j in range(len(result1)))
    print(f"Test Passed: {'YES' if match else 'NO'}")
    visualize_graph(result1, "Shortest Paths Graph (Test 1)")
    
    input_matrix2 = [
        [0, 3, float('inf')],
        [2, 0, 5],
        [float('inf'), 1, 0]
    ]
    expected_output2 = [
        [0, 3, 8],
        [2, 0, 5],
        [3, 4, 0]
    ]
    
    print("\nTest Case 2: Graph with Cycles (No Negative)")
    print_matrix(input_matrix2, "Input Distance Matrix")
    visualize_graph(input_matrix2, "Input Graph (Test 2)")
    
    result2 = copy.deepcopy(input_matrix2)
    tracker = EmissionsTracker()
    tracker.start()
    start_time = time.time()
    result2 = floyd_warshall(result2)
    end_time = time.time()
    tracker.stop()
    
    print_matrix(result2, "Output Distance Matrix")
    print(f"Execution Time: {end_time - start_time:.6f} seconds")
    print(f"Energy Consumed: {tracker.final_emissions_data.energy_consumed} kWh")
    print(f"CO2 Emissions: {tracker.final_emissions} kg")
    
    match2 = all(math.isclose(result2[i][j], expected_output2[i][j], abs_tol=1e-9)
                 for i in range(len(result2)) for j in range(len(result2)))
    print(f"Test Passed: {'YES' if match2 else 'NO'}")
    visualize_graph(result2, "Shortest Paths Graph (Test 2)")
    
    n = 10
    print(f"\nTest Case 3: Random Graph (n={n})")
    input_matrix3 = generate_random_graph(n, edge_prob=0.7, max_weight=20)
    print("Sample Input (first 5x5 submatrix for brevity):")
    sub = [[input_matrix3[i][j] if j < 5 else '...' for j in range(5)] for i in range(5)]
    print_matrix(sub, "Input Submatrix")
    visualize_graph(input_matrix3, f"Input Random Graph (n={n})")
    
    result3 = copy.deepcopy(input_matrix3)
    tracker = EmissionsTracker()
    tracker.start()
    start_time = time.time()
    result3 = floyd_warshall(result3)
    end_time = time.time()
    tracker.stop()
    
    print(f"Execution Time: {end_time - start_time:.6f} seconds")
    print(f"Energy Consumed: {tracker.final_emissions_data.energy_consumed} kWh")
    print(f"CO2 Emissions: {tracker.final_emissions} kg")
    print("Output: Shortest paths computed (full matrix too large to print)")
    visualize_graph(result3, f"Shortest Paths Random Graph (n={n})")

def project_benchmark():
    print("\nProject Benchmark: Energy & Time Complexity Analysis")
    print("=" * 50)
    n_sizes = [5, 10, 20, 50]  # Low to High input sizes
    times = []
    energies_kwh = []
    emissions_kg = []
    
    for n in n_sizes:
        print(f"Running for n={n}...")
        matrix = generate_random_graph(n)
        tracker = EmissionsTracker()
        tracker.start()
        start_time = time.time()
        floyd_warshall(matrix)
        end_time = time.time()
        tracker.stop()
        
        avg_time = (end_time - start_time)  # For simplicity, single run; avg over 10 in real
        times.append(avg_time)
        energies_kwh.append(tracker.final_emissions_data.energy_consumed)
        emissions_kg.append(tracker.final_emissions)
        print(f"n={n}: Time={avg_time:.4f}s, Energy={tracker.final_emissions_data.energy_consumed:.6f} kWh, CO2={tracker.final_emissions:.6f} kg")
    
    # Plot Time vs Energy vs n
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(n_sizes, times, marker='o', label='Time (s)')
    ax1.set_xlabel('Input Size (n)')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title('Time Complexity O(n^3)')
    ax1.grid(True)
    
    ax2.plot(n_sizes, energies_kwh, marker='o', label='Energy (kWh)')
    ax2.set_xlabel('Input Size (n)')
    ax2.set_ylabel('Energy Consumed (kWh)')
    ax2.set_title('Energy Complexity E(n)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Summary Table
    print("\nBenchmark Summary Table:")
    print("| n | Time (s) | Energy (kWh) | CO2 (kg) |")
    print("|---|----------|--------------|----------|")
    for i, n in enumerate(n_sizes):
        print(f"| {n} | {times[i]:.4f} | {energies_kwh[i]:.6f} | {emissions_kg[i]:.6f} |")

if __name__ == "__main__":
    test_floyd_warshall()
    project_benchmark()