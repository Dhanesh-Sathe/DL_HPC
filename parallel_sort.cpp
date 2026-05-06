#include <iostream>
#include <omp.h>
#include <vector>
#include <ctime>
#include <cstdlib>

//  g++ -fopenmp parallel_sort.cpp -o parallel_sort   
// ./parallel_sort  

using namespace std;

// Generate random array
void generateRandomArray(vector<int>& arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100000;
    }
}

// Sequential Bubble Sort
void seqBubbleSort(vector<int>& arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort (Odd-Even Transposition Sort)
void parBubbleSort(vector<int>& arr, int n) {
    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) {
            #pragma omp parallel for
            for (int j = 0; j < n - 1; j += 2) {
                if (arr[j] > arr[j + 1]) {
                    swap(arr[j], arr[j + 1]);
                }
            }
        } else {
            #pragma omp parallel for
            for (int j = 1; j < n - 1; j += 2) {
                if (arr[j] > arr[j + 1]) {
                    swap(arr[j], arr[j + 1]);
                }
            }
        }
    }
}

// Merge function
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++) {
        L[i] = arr[left + i];
    }

    for (int j = 0; j < n2; j++) {
        R[j] = arr[mid + 1 + j];
    }

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }

    while (i < n1) {
        arr[k++] = L[i++];
    }

    while (j < n2) {
        arr[k++] = R[j++];
    }
}

// Sequential Merge Sort
void seqMergeSort(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        seqMergeSort(arr, left, mid);
        seqMergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

// Parallel Merge Sort
void parMergeSort(vector<int>& arr, int left, int right, int depth = 0) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        if (depth <= 4) {
            #pragma omp task shared(arr)
            parMergeSort(arr, left, mid, depth + 1);

            #pragma omp task shared(arr)
            parMergeSort(arr, mid + 1, right, depth + 1);

            #pragma omp taskwait
        } else {
            seqMergeSort(arr, left, mid);
            seqMergeSort(arr, mid + 1, right);
        }

        merge(arr, left, mid, right);
    }
}

int main() {
    int n = 50000;
    // omp_set_num_threads(4);  this will take no of cores available on your device like 4 core= 4 threads
    srand(time(0));

    vector<int> arr1(n), arr2(n);

    generateRandomArray(arr1, n);

    double start, end;
    double seqBubbleTime, parBubbleTime, seqMergeTime, parMergeTime;

    // Sequential Bubble Sort
    arr2 = arr1;
    start = omp_get_wtime();
    seqBubbleSort(arr2, n);
    end = omp_get_wtime();
    seqBubbleTime = end - start;
    cout << "Sequential Bubble Sort Time: " << seqBubbleTime << " seconds" << endl;

    // Parallel Bubble Sort
    arr2 = arr1;
    start = omp_get_wtime();
    parBubbleSort(arr2, n);
    end = omp_get_wtime();
    parBubbleTime = end - start;
    cout << "Parallel Bubble Sort Time: " << parBubbleTime << " seconds" << endl;

    cout << "Bubble Sort Speedup: " << seqBubbleTime / parBubbleTime << "x" << endl;

    // Sequential Merge Sort
    arr2 = arr1;
    start = omp_get_wtime();
    seqMergeSort(arr2, 0, n - 1);
    end = omp_get_wtime();
    seqMergeTime = end - start;
    cout << "Sequential Merge Sort Time: " << seqMergeTime << " seconds" << endl;

    // Parallel Merge Sort
    arr2 = arr1;
    start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            parMergeSort(arr2, 0, n - 1);
        }
    }

    end = omp_get_wtime();
    parMergeTime = end - start;
    cout << "Parallel Merge Sort Time: " << parMergeTime << " seconds" << endl;

    cout << "Merge Sort Speedup: " << seqMergeTime / parMergeTime << "x" << endl;

    return 0;
}