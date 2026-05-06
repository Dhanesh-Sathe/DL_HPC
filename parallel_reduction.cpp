#include <iostream>
#include <omp.h>
#include <ctime>
#include <cstdlib>

//  g++ -fopenmp parallel_bfs_dfs.cpp -o parallel_bfs_dfs
// ./parallel_bfs_dfs

using namespace std;

void min(int *arr, int n)
{
      double min_val = arr[0];
      int i;

      // Display total threads
      #pragma omp parallel
      {
         #pragma omp single
            cout << "\n[MIN] Total threads = " << omp_get_num_threads() << endl;
      }

      double start_parallel = omp_get_wtime();

      #pragma omp parallel for reduction(min : min_val)
         for (i = 0; i < n; i++)
         {
            #pragma omp critical
               cout << "[MIN] Thread " << omp_get_thread_num() << " handles index " << i << endl;

            if (arr[i] < min_val)
            {
               min_val = arr[i];
            }
         }

      double end_parallel = omp_get_wtime();

      cout << "\nmin_val = " << min_val << endl;
      cout << "Parallel Time = " << (end_parallel - start_parallel) << endl;
}

void max(int *arr, int n)
{
      double max_val = arr[0];
      int i;

      #pragma omp parallel
      {
         #pragma omp single
            cout << "\n[MAX] Total threads = " << omp_get_num_threads() << endl;
      }

      double start_parallel = omp_get_wtime();

      #pragma omp parallel for reduction(max : max_val)
         for (i = 0; i < n; i++)
         {
            #pragma omp critical
               cout << "[MAX] Thread " << omp_get_thread_num() << " handles index " << i << endl;

            if (arr[i] > max_val)
            {
               max_val = arr[i];
            }
         }

      double end_parallel = omp_get_wtime();

      cout << "\nmax_val = " << max_val << endl;
      cout << "Parallel Time = " << (end_parallel - start_parallel) << endl;
}

void avg(int *arr, int n)
{
   int i;
   float avg = 0, sum = 0;

   #pragma omp parallel
      {
         #pragma omp single
            cout << "\n[AVG] Total threads = " << omp_get_num_threads() << endl;
      }

   double start_parallel = omp_get_wtime();

   #pragma omp parallel reduction(+:sum)
   {
      #pragma omp for
         for (i = 0; i < n; i++)
         {
            #pragma omp critical
               cout << "[AVG] Thread " << omp_get_thread_num() << " handles index " << i << endl;

            sum = sum + arr[i];
         }
   }

   double end_parallel = omp_get_wtime();

   cout << "\nSum = " << sum << endl;
   avg = sum / n;
   cout << "Average = " << avg << endl;

   cout << "Parallel Time = " << (end_parallel - start_parallel) << endl;
}

int main()
{
   omp_set_num_threads(4);   //setting 4 threads
   int n, i;

   cout << "Enter the number of elements in the array: ";
   cin >> n;
   int arr[n];

   srand(time(0));
   for (int i = 0; i < n; ++i)
   {
      arr[i] = rand() % 100;
   }

   cout << "\nArray elements are: ";
   for (i = 0; i < n; i++)
   {
      cout << arr[i] << ",";
   }

   min(arr, n);
   max(arr, n);
   avg(arr, n);

   return 0;
}