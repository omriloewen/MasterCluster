#define _GNU_SOURCE
#include "symnmf.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/**
 * @brief Prints a square matrix of doubles.
 *
 * This function prints a 2D array (matrix) of doubles to the console.
 * Each element is printed with 4 decimal places, and elements in the 
 * same row are separated by commas.
 *
 * @param M A pointer to a pointer representing the matrix to be printed.
 *           It is assumed that M is a square matrix of size N x N.
 * @param N The number of rows (and columns) in the matrix M.
 *
 * @return None. The function prints directly to the console.
 */
void printM(double** M,int N){
    int i,j;
    /* Check if the matrix pointer is NULL, indicating that no matrix exists. */
    if (M == NULL){
        return;
    }
    /* Loop through each row of the matrix. */
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){ /* Loop through each element in the current row. */
            printf("%.4f", M[i][j]);
            if(j != N-1){ /*  If this is not the last element in the row, print a comma. */
                printf(",");
            }
        }
        printf("\n");/* After printing all elements in the current row, print a newline. */
    }
}

/**
 * @brief deep_free - Free memory allocated for a 2D array of doubles.
 *
 * @param M: A pointer to a 2D array of doubles (double**).
 * @param num_rows: The number of rows in the 2D array.
 *
 * This function iterates through each row of the 2D array pointed to by M,
 * freeing the memory allocated for each row. Once all rows are freed, it
 * also frees the memory allocated for the array of row pointers itself.
 * 
 * Note: This function assumes that the pointer is not a NULL pointer.
 */
void deep_free(double** M, int num_rows){
    int row;
    for ( row = 0; row < num_rows; row++){
        free(M[row]);
    }
    free(M);
}



/**
 * @brief Calculate the squared Euclidean distance between two points.
 *
 * This function computes the squared distance between two points in
 * a d-dimensional space.
 *
 * @param point1 A pointer to the first point (array of coordinates).
 * @param point2 A pointer to the second point (array of coordinates).
 * @param d The number of dimensions (size of the point arrays).
 * @return The squared Euclidean distance between the two points.
 */
double EDsq(double* point1, double* point2, int d){
    double dist = 0.0 , diff;
    int i;

    for(i = 0; i < d; i++){ /* Loop through each dimension. */
        diff = point1[i] - point2[i]; /* Calculate the difference between the corresponding coordinates. */
        dist += diff * diff; /* Accumulate the square of the difference to the distance. */
    }
    return dist; /* Return the total squared distance. */
}

/**
 * @brief Calculates the squared Frobenius C_norm of the difference 
 *        between two matrices H1 and H2.
 * 
 * The Frobenius C_norm of a matrix is defined as the square root 
 * of the sum of the absolute squares of its elements. 
 * This function computes the squared version, which is useful for 
 * optimization and comparison purposes.
 * 
 * @param H1: A pointer to a 2D array (matrix) of doubles representing 
 *             the first matrix.
 * @param H2: A pointer to a 2D array (matrix) of doubles representing 
 *             the second matrix.
 * @param N: The number of rows in the matrices.
 * @param k: The number of columns in the matrices.
 * 
 * @return The squared Frobenius C_norm of the difference between H1 and H2.
 */
double FNsq(double** H1, double** H2, int N, int k){
    int i,j;
    double  diff, res = 0;
    /* Iterate over each row of the matrices */
    for (i = 0; i < N; i++){
        /* Iterate over each column of the matrices */
        for ( j = 0; j < k; j++){
            /* Calculate the difference between the corresponding elements */
            diff = H1[i][j] - H2[i][j];
            /* Accumulate the square of the difference into the result */
            res += diff * diff;
        }
    }
    /* Return the final result, which is the squared C_norm */
    return res;
}

/**
 * @brief Based on the similarity matrix A, creates the diagonal degree matrix D where the diagonal
 * elements are the sums of the respective rows of the A matrix.
 * 
 * @param A: A pointer to a 2D array of double values (the similarity matrix A).
 * @param N: An integer representing the size of the 2D array (number of rows/columns).
 * @return A pointer to a new 2D array of double values (the diagonal degree matrix D).
 * 
 * Note:
 * -The function dynamically allocates memory for the output array D. 
 * -Ensure to free the allocated memory after use.
 */
double** AtoD(double** A,int N){
    int i,j;
    double S;
    double **D = (double**)malloc(N * sizeof(double*));
    if (D == NULL){ /* Exit if memory allocation fails  */
                printf("An Error Has Occurred");
                exit(1);
                }
    for ( i = 0; i < N; i++){
        D[i] = (double*)malloc(N * sizeof(double)); /* Allocate memory for each row of D */
        if (D[i] == NULL){ /* Exit if memory allocation fails  */
                printf("An Error Has Occurred");
                exit(1);
                }
        S = 0; /*  Reset sum for the current row */
        for ( j = 0; j < N; j++){ /* Loop through each column */
            D[i][j] = 0; /* Initialize the row to 0 */
            S += A[i][j]; /* calculate the sum */
        }
        D[i][i] = S; /* Assign the calculated sum to the diagonal element */
    }
    return D;
}

/**
 * @brief based on the similarity matrix A and the diagonal degree matrix D, 
 *        Create the Laplacian matrix W.
 *
 *   @param A: A pointer to a 2D array of double values (the similarity matrix)
 *   @param D: A pointer to a 2D array of double values ( the diagonal degree matrix)
 *   @param N: The size (number of rows/columns) of the square matrices A and D.
 *
 * @returns: A A pointer to a 2D array of double values W that is the Laplacian matrix.
 * 
 * Note: 
 * -The function dynamically allocates memory for the new matrix W.
 * -Make sure to free the allocated memory after use.
 */
double** ADtoW(double** A, double** D,int N){
    int i,j;
    double **W = (double**)malloc(N * sizeof(double*));
    if (W == NULL){ /* Exit if memory allocation fails */
                printf("An Error Has Occurred");
                exit(1);
                }
    for ( i = 0; i < N; i++){
        W[i] = (double*)malloc(N * sizeof(double)); /* Allocate memory for each row of the matrix W */
        if (W[i] == NULL){ /* Exit if memory allocation for a row fails */
                printf("An Error Has Occurred");
                exit(1);
                }
        for ( j = 0; j < N; j++){
            /* normalizes A[i][j] using the corresponding diagonal values of D */
            W[i][j] = A[i][j] * (1/sqrt(D[i][i])) * (1/sqrt(D[j][j]));
        }   
    }
    return W;
}

/**
 * @brief Creates the similarity matrix A based on the squared Euclidean distances.
 *
 * @param X: The input 2D array containing N points, each with d dimensions.
 * @param N: The number of points (rows in X).
 * @param d: The number of dimensions of each point.
 *
 * @returns: A pointer to a dynamically allocated 2D array (matrix) of doubles.
 * The matrix is symmetric, with diagonal values set to 0 and other values 
 * calculated using the formula exp(-EDsq(X[i], X[j], d)/2).
 * 
 * Note:
 * - The function dynamically allocates memory for the output array A.
 * - Ensure to free the allocated memory after use.
 */
double** C_sym(double** X, int N, int d){
    int i, j;
    double **A = (double**)malloc(N * sizeof(double*));
    if (A == NULL){ /* Exit if memory allocation fails */
                printf("An Error Has Occurred");
                exit(1);
                }

    for (i = 0; i < N; i++)
    {
        A[i] = (double*)malloc(N * sizeof(double)); /* Allocate memory for each row of the matrix */
        if (A[i] == NULL){ /*  Exit if memory allocation fails */
                printf("An Error Has Occurred");
                exit(1);
                }
        for (j = 0 ; j < N; j++)
        {
            if (j == i){
                A[i][j] = 0; /* Set diagonal elements to 0 */
            }else if (j < i){ /* if the symmetric position was already calculated */
                /* Since the matrix is symmetric, copy the values from the symmetric position */
                A[i][j] = A[j][i];
            }else{ /* Populate the upper triangle of the matrix */
                /* Calculate the value using the squared Euclidean distance */
                A[i][j] = exp(-(EDsq(X[i],X[j], d)/2));
                }
            }  
        }
    return A; /* Return the constructed similarity matrix */
    }

/**
 * @brief Performs symmetric Non-negative Matrix Factorization (NMF).
 *
 * This function computes the factorization of a given matrix W using a matrix H,
 * modifying H iteratively to minimize the difference between W and WH.
 *
 * @param W The input matrix (double**), where W is of size N x N.
 * @param H The matrix being factored (double**), of size N x k.
 * @param N The size of the square matrix W (number of rows/columns).
 * @param k The number of components (rank) for the factorization.
 * 
 * @return A pointer to the modified matrix H (double**) that approximates W.
 * 
 * @note The function update H inplace, so the original values will not be saved.
 */
double** C_symnmf(double** W, double** H,int N,int k){
    int i,j,t,l, iter ;
    double WHij, HHTHij;
    /* Allocate memory for the new matrix H1 */
    double **H1 = (double**)malloc(N * sizeof(double*)); /* The H(t+1) matrix */
    if (H1 == NULL){ /* Error handling for memory allocation failure */
        printf("An Error Has Occurred");
        exit(1);
    }
    for (i = 0; i < N; i++){
        H1[i] = (double*)malloc(k * sizeof(double));
        if (H1[i] == NULL){ /* Error handling for memory allocation failure */
            printf("An Error Has Occurred");
            exit(1);
            }
    }
    for ( iter = 0; iter < 300; iter++){ /* Main iterative loop */
        /* Update H(t+1) based on current values of H(t) */
        for ( i = 0; i < N; i++){
            for ( j = 0; j < k; j++){
                WHij = 0;
                HHTHij = 0;
                for (t = 0; t < N; t++){
                    WHij += W[i][t] * H[t][j]; /* Calculate i j element of the dot product W * H */
                    for (l = 0; l < k; l++){
                        HHTHij += H[i][l] * H[t][l] * H[t][j]; /* Calculate i j element of the dot product H*H^T*H */
                    }
                }
                if (HHTHij == 0){ /* Error handling for division by zero */
                     printf("An Error Has Occurred");
                     exit(1);
                }
                /* Update the value of H1 based on the computed terms */
                H1[i][j] = H[i][j] *(0.5 + 0.5 *(WHij / HHTHij));
            }  
        }
        /* Check for convergence based on the FNsq function */
        if (FNsq(H, H1, N, k) < 0.0001){
            /* If converged, update H with H1's values and exit the loop */
            for (i = 0; i < N; i++){
                for (j = 0; j < k; j++){
                    H[i][j] = H1[i][j];
                }
            }
            break;
        }
        /* Update H with H1's values if not converged yet */
        for (i = 0; i < N; i++){
            for (j = 0; j < k; j++){
                H[i][j] = H1[i][j];
            }
        } 
    }
    deep_free(H1,N); /* Free memory for H1 after use */
    return H; /* Return the modified matrix H */
}


/**
 * C_ddg - Computes the diagonal degree matrix D from the input matrix X.
 *
 * This function takes a matrix X, computes its similarity form A,
 * converts A to the diagonal degree matrix D, frees the memory allocated for A,
 * and returns the diagonal matrix D.
 *
 * @param X: A pointer to a double pointer representing the input matrix.
 *           This is a 2D array of size (N x d).
 * @param N: The number of rows in the matrix X.
 * @param d: The number of columns in the matrix X.
 * 
 * @return: A pointer to a double pointer representing the diagonal degree matrix D.
 *          The caller is responsible for freeing this memory after use.
 */
double** C_ddg(double** X,int N, int d){
    double **A, **D;
    A = C_sym(X, N, d); /* Compute the similarity form of matrix X and store it in A */
    D = AtoD(A,N); /* Convert the similarity matrix A to its diagonal degree matrix D */
    deep_free(A,N); /* Free the memory allocated for the matrix A */
    return D; 
}

/**
 * C_norm - Computes the Laplacian matrix W from the input matrix X.
 *
 * This function takes a matrix X, computes its similarity form A,
 * converts A to the diagonal degree matrix D, converts D to the Laplacian matrix W, frees the memory allocated for A and D,
 * and returns the Laplacian matrix W.
 *
 * @param X: A pointer to a double pointer representing the input matrix.
 *           This is a 2D array of size (N x d).
 * @param N: The number of rows in the matrix X.
 * @param d: The number of columns in the matrix X.
 * 
 * @return: A pointer to a double pointer representing the Laplacian matrix W.
 *          The caller is responsible for freeing this memory after use.
 */
double** C_norm(double** X,int  N, int d){
    double **A, **D, **W;
    A = C_sym(X, N, d); /* Convert the input matrix X to a similarity matrix A */
    D = AtoD(A,N); /* Convert the similarity matrix A to a diagonal degree matrix D */
    W = ADtoW(A, D, N); /* Convert the similarity matrix A and diagonal degree matrix D to the Laplacian matrix W */
    /* Free any allocated memory for matrices A and D */
    deep_free(A, N);
    deep_free(D,N);
    return W;
}
 