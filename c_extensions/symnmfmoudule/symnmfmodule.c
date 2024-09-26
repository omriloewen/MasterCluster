#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include"symnmf.h"

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
