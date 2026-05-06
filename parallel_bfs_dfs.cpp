#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
#include <chrono>
#include <algorithm>

//  g++ -fopenmp parallel_bfs_dfs.cpp -o parallel_bfs_dfs
// ./parallel_bfs_dfs
using namespace std;

class Graph {
public:
    int V;
    vector<vector<int>> adj;

    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // =====================
    // Sequential BFS
    // =====================
    void sequentialBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int node = q.front();
            q.pop();

            for (int neigh : adj[node]) {
                if (!visited[neigh]) {
                    visited[neigh] = true;
                    q.push(neigh);
                }
            }
        }
    }

    // =====================
    // Parallel BFS (Fixed)
    // =====================
    void parallelBFS(int start) {

        vector<int> visited(V,0);
        vector<int> frontier;

        frontier.push_back(start);
        visited[start]=1;

        while (!frontier.empty()) {
            vector<int> next_frontier;

            if (frontier.size() < 100) {
                for (int node : frontier) {
                    for (int neigh : adj[node]) {
                        if (!visited[neigh]) {
                            visited[neigh] = 1;
                            next_frontier.push_back(neigh);
                        }
                    }
                }
            } else {
                #pragma omp parallel
                {
                    vector<int> local_buffer;
                    local_buffer.reserve(frontier.size() / omp_get_num_threads());

                    #pragma omp for schedule(dynamic, 64) nowait
                    for (int i = 0; i < frontier.size(); i++) {
                        int node = frontier[i];

                        for (int neigh : adj[node]) {
                            int expected = 0;
                            #pragma omp atomic capture
                            {
                                expected = visited[neigh];
                                visited[neigh] = 1;
                            }

                            if (expected == 0) {
                                local_buffer.push_back(neigh);
                            }
                        }
                    }

                    #pragma omp critical
                    {
                        next_frontier.insert(next_frontier.end(), local_buffer.begin(), local_buffer.end());
                    }
                }
            }

            frontier.swap(next_frontier);
        }
    }

    // =====================
    // Sequential DFS
    // =====================
    void sequentialDFS(int start) {
        vector<bool> visited(V, false);
        stack<int> st;

        st.push(start);

        while (!st.empty()) {
            int node = st.top();
            st.pop();

            if (visited[node]) continue;

            visited[node] = true;

            for (int neigh : adj[node]) {
                if (!visited[neigh]) {
                    st.push(neigh);
                }
            }
        }
    }

    // =====================
    // Parallel DFS (Fixed)
    // =====================
    void parallelDFS(int start) {
        vector<int> visited(V, 0);
        vector<int> frontier, next;

        frontier.push_back(start);
        visited[start] = 1;

        while (!frontier.empty()) {
            next.clear();

            #pragma omp parallel
            {
                vector<int> local;

                #pragma omp for nowait
                for (int i = 0; i < frontier.size(); i++) {
                    int node = frontier[i];

                    // Reverse order to mimic DFS behavior
                    for (int j = adj[node].size() - 1; j >= 0; j--) {
                        int neigh = adj[node][j];
                        int expected = 0;

                        #pragma omp atomic capture
                        {
                            expected = visited[neigh];
                            visited[neigh] = 1;
                        }

                        if (expected == 0) {
                            local.push_back(neigh);
                        }
                    }
                }

                #pragma omp critical
                {
                    next.insert(next.end(), local.begin(), local.end());
                }
            }

            // Reverse to simulate stack-like (DFS) behavior
            reverse(next.begin(), next.end());

            frontier.swap(next);
        }
    }
};

int main() {
    // Set number of threads (adjust based on your CPU)
    omp_set_num_threads(4);

    cout << "Using " << omp_get_max_threads() << " threads" << endl;

    int N = 1000000;
    Graph g(N);

    // Creating a denser graph for better parallelization
    for (int i = 0; i < N; i++) {
        for (int j = 1; j <= 50; j++) {
            int v = (i + j) % N;
            g.addEdge(i, v);
        }
    }

    cout << "\n--- Running Benchmarks ---\n" << endl;

    // Warm-up run
    g.sequentialBFS(0);

    auto start = chrono::high_resolution_clock::now();
    g.sequentialBFS(0);
    auto end = chrono::high_resolution_clock::now();

    double seq_bfs_time = chrono::duration<double>(end - start).count();
    cout << "Sequential BFS Time: " << seq_bfs_time << " seconds" << endl;

    start = chrono::high_resolution_clock::now();
    g.parallelBFS(0);
    end = chrono::high_resolution_clock::now();

    double par_bfs_time = chrono::duration<double>(end - start).count();
    cout << "Parallel BFS Time:   " << par_bfs_time << " seconds" << endl;
    cout << "BFS Speedup:         " << seq_bfs_time / par_bfs_time << "x\n" << endl;

    start = chrono::high_resolution_clock::now();
    g.sequentialDFS(0);
    end = chrono::high_resolution_clock::now();

    double seq_dfs_time = chrono::duration<double>(end - start).count();
    cout << "Sequential DFS Time: " << seq_dfs_time << " seconds" << endl;

    start = chrono::high_resolution_clock::now();
    g.parallelDFS(0);
    end = chrono::high_resolution_clock::now();

    double par_dfs_time = chrono::duration<double>(end - start).count();
    cout << "Parallel DFS Time:   " << par_dfs_time << " seconds" << endl;
    cout << "DFS Speedup:         " << seq_dfs_time / par_dfs_time << "x" << endl;

    return 0;
}
