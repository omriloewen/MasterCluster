#define PY_SSIZE_T_CLEAN
#define _GNU_SOURCE
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


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
 
/**
 * @brief Converts a Python matrix (list of lists) to a C double matrix.
 * 
 * This function takes a Python object representing a matrix, specified by 
 * the number of rows and columns, and allocates memory for a corresponding 
 * C double pointer (2D array). It then fills this array with the values 
 * from the Python matrix.
 *
 * @param PyM A PyObject representing a matrix (list of lists) in Python.
 * @param num_rows The number of rows in the Python matrix.
 * @param num_cols The number of columns in the Python matrix.
 * @return A pointer to a double pointer representing the C matrix (2D array).
 * 
 * @note The function allocate memory for the result matrix, ensure to free it after use.
 */
double** Py_to_C(PyObject* PyM, int num_rows, int num_cols){
    int row, col;
    PyObject *Pyrow, *Pyval;
    double** C_M;
     /*allocate memory for the C version of the matrix*/
    C_M = (double**)malloc(num_rows * sizeof(double*));
    if (C_M == NULL){ /* Exit if memory allocation fails */
        printf("An Error Has Occurred");
        exit(1);
    }
    for(row = 0; row < num_rows; row++){ /* Loop through each row */
        C_M[row] = (double*)malloc(num_cols * sizeof(double)); /* Allocate memory for each row */
        if (C_M[row] == NULL){ /* Exit if memory allocation fails */
            printf("An Error Has Occurred");
            exit(1);
        }
        Pyrow = PyList_GetItem(PyM, row); /* Get the Python list representing the current row */
        
        for(col = 0; col < num_cols; col++){ /* Loop through each column in the current row */
            Pyval = PyList_GetItem(Pyrow, col); /* Get the value at the current position */
            C_M[row][col] = PyFloat_AsDouble(Pyval); /* Convert the Python float to a C double and set it in the C matrix */
            if(PyErr_Occurred()){ /* Check for errors in the conversion process */
                PyErr_Clear();
            }
        }
    }
    return C_M;
}

/**
 *  @brief Convert a C-style 2D array of doubles to a Python list of lists.
 *
 * @param C_M: Pointer to a 2D array (array of pointers) representing the C matrix.
 * @param num_rows: The number of rows in the C matrix.
 * @param num_cols: The number of columns in the C matrix.
 *
 * @returns: A PyObject* that represents a list of lists in Python, where 
 * each inner list contains the elements of the corresponding row in the
 * input C matrix.
 */
static PyObject* C_to_Py(double** C_M, int num_rows, int num_cols){
    int col, row;
    PyObject *Py_M, *Pyrow;
    Py_M = PyList_New(num_rows); /* Create a new Python list to hold the rows */
    
    for(row = 0; row < num_rows; row++){ /* Loop over each row of the C matrix */
        Pyrow = PyList_New(num_cols); /* Create a new Python list for the current row */
        for(col = 0; col < num_cols; col++){ /* Loop over each column in the current row */
            PyList_SET_ITEM(Pyrow, col, PyFloat_FromDouble(C_M[row][col])); /* Convert the C double to a Python float and set it in the row list */
        }
        PyList_SET_ITEM(Py_M,row,Pyrow); /* Set the completed row list in the main Python list */
    }
    return Py_M;
}

/**
 * Function: symnmf
 * -----------------
 * This function performs Symmetric Non-negative Matrix Factorization (SymNMF)
 * on two matrices W and H provided by the user in Python. The function converts
 * the input Python lists to C arrays, calls a C function to perform the NMF
 * computation, and converts the resulting matrix back to a Python object for the
 * caller.
 *
 * Parameters:
 *    self: The module or object this function is attached to (not used in this function).
 *    args: A tuple containing:
 *        - PyList of W (the Laplacian matrix)
 *        - PyList of H (the initial H matrix)
 *        - N (the size of matrix W)
 *        - k (the number of components in matrix H)
 *
 * Returns:
 *    A PyObject* pointing to the resulting matrix H after the NMF computation,
 *    or NULL if an error occurred during input parsing or matrix operations.
 */
static PyObject* symnmf(PyObject* self, PyObject* args){
    PyObject *PyW, *PyH;
    int N, k;
    double **W, **H;
    /*Get the Python arguments and check for validity*/
    if (! PyArg_ParseTuple(args, "O!O!ii",&PyList_Type, &PyW, &PyList_Type, &PyH, &N, &k)){
        return NULL; /* Return NULL if argument parsing fails */
    }
    /* convert Python lists to C 2D arrays, memory allocation is within the conversion */
    W = Py_to_C(PyW,N,N); 
    H = Py_to_C(PyH,N,k);
    /* Perform the SymNMF computation on W and H */
    C_symnmf(W,H,N,k); /* H updates in place */
    PyH = C_to_Py(H,N,k); /* Convert the resulting C array H back to a Python object */
    /*printM(H,  N);  Print the resulting matrix H (for debugging purposes) */
    /* Free the dynamically allocated C arrays */
    deep_free(W,N);
    deep_free(H,N);
    return PyH;
}

/**
 * This function applies a specified goal function (e.g., C_sym, C_ddg, C_norm) 
 * to a 2D array represented as a Python list.
 *
 * @param self: The module (unused in this context).
 * @param args: The arguments passed from Python.
 * @param goal: A pointer to a function which takes a 2D array and dimensions 
 *              and returns a 2D array after processing.
 * @return: A Python object representing the result of the goal function.
 */
static PyObject* goal_M(PyObject* self, PyObject* args, double** (*goal)(double**, int, int)){
    PyObject *PyX, *PyM;
    int N, d;
    double **X, **M;
    /* Parse the input arguments from Python; expects a list (PyX), and two integers (N, d) */
    if (! PyArg_ParseTuple(args, "O!ii",&PyList_Type, &PyX, &N, &d)){
        return NULL; /* Return NULL if parsing fails */
    }
    X = Py_to_C(PyX,N,N); /* Convert the Python list X to a C-style 2D array */
    M = goal(X,N,d); /* Call the goal function */
    PyM = C_to_Py(M,N,N); /* Convert the resulting C-style 2D array back to a Python object */
    /*printM(M,  N);  Optionally print the resulting matrix (for debugging purposes) */
    /* Free up the memory allocated for the input and output 2D arrays */
    deep_free(X,N);
    deep_free(M,N);
    return PyM;
}

/**
 * Wrapper function for applying the sym goal function (C_sym).
 *
 * @param self: The module (unused in this context).
 * @param args: The arguments passed from Python.
 * @return: A Python object representing the result of the C_sym function.
 */
static PyObject* sym(PyObject* self, PyObject* args){
    PyObject* M = goal_M(self, args, C_sym); /* Call goal_M with the C_sym function */
    return M;
}

/**
 * Wrapper function for applying the diagonal degree goal function (C_ddg).
 *
 * @param self: The module (unused in this context).
 * @param args: The arguments passed from Python.
 * @return: A Python object representing the result of the C_ddg function.
 */
static PyObject* ddg(PyObject* self, PyObject* args){
    return goal_M(self, args, C_ddg); /* Call goal_M with the C_ddg function */
}

/**
 * Wrapper function for applying the normalization goal function (C_norm).
 *
 * @param self: The module (unused in this context).
 * @param args: The arguments passed from Python.
 * @return: A Python object representing the result of the C_norm function.
 */
static PyObject* norm(PyObject* self, PyObject* args){
    return goal_M(self, args, C_norm); /* Call goal_M with the C_norm function */
}

 

/*module methods*/
static PyMethodDef MyMethods[] = {
    {"symnmf", symnmf, METH_VARARGS, PyDoc_STR("")},
    {"sym", sym, METH_VARARGS, PyDoc_STR("")},
    {"ddg", ddg, METH_VARARGS, PyDoc_STR("")},
    {"norm", norm, METH_VARARGS, PyDoc_STR("")},
    {NULL, NULL, 0, NULL}
};

/*module definitions*/
static struct PyModuleDef mysymnmf = {
    PyModuleDef_HEAD_INIT,
    "mysymnmf",
    NULL,
    -1,
    MyMethods
};

PyMODINIT_FUNC PyInit_mysymnmf(void){
    return PyModule_Create(&mysymnmf);
}
