/* Bryson Goad
 * Parallel Quicksort
 * Implementation of a parallel quicksort using OpenMP
 * uses in place recursive quicksort algorithm
 * parallel sort reverts to sequential at cutoff partition size to eliminate unnecessary overhead and improve performance
 * optimal cutoff point may vary, in my testing around 440000 worked well
 *
 * randomly generates an array of ints for ease of testing with various sizes
 * runs sequential and parallel sorts with same data and calculates average time for each
 *
 * Program Arguments: <thread_count> <array_length> <cutoff> <test_count>
 */

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <string.h>

//#define DISPLAY_ARR   // displays arrays when defined
#define SEQ             // runs sequential quicksort when defined
#define PAR             // runs parallel quicksort when defined

void qSort(int[], int);
int partition(int[], int);
void qSortPar(int[], int);

int thread_count;       // number of threads used for parallel sections
int cutoff;             // array sizes less than the cutoff will revert to sequential sort

int main(int argc, char* argv[]) {
    int length;                     // length of the array being sorted
    int *A1, *A2;                   // arrays to be sorted
    double time_start, time_end;    // start and end time of the sort
    int test_count;                 // number of times to run the sort
    double seq_sum = 0;             // sum of sequential run times, used for calculating average
    double par_sum = 0;             //  sum of parallel run times, used for calculating average

    // get program arguments
    thread_count = atoi(argv[1]);
    length = atoi(argv[2]);
    cutoff = atoi(argv[3]);
    test_count = atoi(argv[4]);

    printf("thread_count: %d\n", thread_count);
    printf("length: %d\n", length);
    printf("cutoff: %d\n", cutoff);
    printf("test_count: %d\n\n", test_count);

    // allocate arrays
    A1 = malloc(length * sizeof(int));
    A2 = malloc(length * sizeof(int));

    // seed random number generator
    srand(time(NULL));

    for (int i = 0; i < test_count; i++) {
        // generate random data
        printf("generating data...");
        #pragma omp parallel for num_threads(thread_count)
        for (int i = 0; i < length; i++) {
            A1[i] = rand();
        }
        memcpy(A2, A1, length * sizeof(int));
        printf("done generating\n");

        #ifdef DISPLAY_ARR
        // display unsorted array
        for (int i = 0; i < length; ++i) {
            printf("%d, ", A1[i]);
        }
        printf("\n");
        #endif

        /*--------- sequential sort --------------------------------------*/
        #ifdef SEQ
        printf("\nsequential sort:\n");

        time_start = omp_get_wtime();   // start timer

        qSort(A1, length);          // sequential sort

        time_end = omp_get_wtime();     // end timer

        #ifdef DISPLAY_ARR
        // display sorted array
        for (int i = 0; i < length; ++i) {
            printf("%d, ", A1[i]);
        }
        printf("\n");
        #endif

        // add elapsed time to sum
        seq_sum += (time_end - time_start);
        // display elapsed time
        printf("elapsed time: %f\n\n", time_end - time_start);
        #endif
        /*--------------------------------------------------------------*/

        /*--------- parallel sort ---------------------------------------*/
        #ifdef PAR
        printf("parallel sort:\n");

        time_start = omp_get_wtime();   // start timer

        // parallel sort
        #pragma omp parallel num_threads(thread_count)
        {
            // start by creating a partition with a single thread
            #pragma omp single nowait
            qSortPar(A2, length);
        }

        time_end = omp_get_wtime();     // end timer

        #ifdef DISPLAY_ARR
        // display sorted array
        for (int i = 0; i < length; ++i) {
            printf("%d, ", A2[i]);
        }
        printf("\n");
        #endif

        // add elapsed time to sum
        par_sum += (time_end - time_start);
        // display elapsed time
        printf("elapsed time: %f\n\n", time_end - time_start);
        #endif
        /*--------------------------------------------------------------*/
    }
    printf("\n");

    // calculate and display average times
    #ifdef SEQ
    printf("sequential average: %f\n", seq_sum/test_count);
    #endif
    #ifdef PAR
    printf("parallel average: %f\n", par_sum/test_count);
    #endif

    return 0;
}

// Sequential Quicksort
// Parameters:
//      A : pointer to start of partition to sort
//      Length : length of partition
void qSort(int A[], int length) {
    // a single element is already sorted
    if (length > 1) {
        int p = partition(A, length);   // create partition
        qSort(A, p);                    // sort left side
        qSort(A + p, length - p);       // sort right side
    }
}

// Creates partition for quicksort
// Parameters:
//      A : pointer to start of partition to sort
//      Length : length of partition
// Returns:
//      pivot point
int partition(int A[], int length) {
    int pivot = A[length / 2];  // select pivot
    int i = 0;                  // left side index
    int j = length - 1;         // right side index

    // find elements on the wrong side of pivot and swap them in place
    while (true) {
        while (A[i] < pivot) i++;
        while (A[j] > pivot) j--;

        if (i >= j) return i;

        int temp = A[i];
        A[i] = A[j];
        A[j] = temp;

        i++;
        j--;
    }
}

// Parallel Quicksort
// must be started by a single thread
// creates OpenMP tasks for each side of the partition when partition size is above cutoff
// reverts to sequential quicksort when partition size is below cutoff
// Parameters:
//      A : pointer to start of partition to sort
//      Length : length of partition
void qSortPar(int A[], int length) {
    // a single element is already sorted
    if (length > 1) {
        int p = partition(A, length);   // create partition

        // if below cutoff revert to sequential
        if (length < cutoff) {
            qSortPar(A, p);                 // sort left side
            qSortPar(A + p, length - p);    // sort right side
        }
        else {
            // create a task to sort the left side
            #pragma omp task default(none) firstprivate(A, p)
            qSortPar(A, p);

            // create a task to sort the right side
            #pragma omp task default(none) firstprivate(A, p, length)
            qSortPar(A + p, length - p);
        }
    }
}

