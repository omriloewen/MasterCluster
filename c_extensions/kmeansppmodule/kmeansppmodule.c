#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * @brief Deallocates memory for a 2D array.
 *
 * This function frees the memory allocated for a 2D array of doubles.
 *
 * @param array A pointer to the 2D array to be freed.
 * @param rows The number of rows in the array.
 */
void deep_free(double** array, int rows){
    int i;
    for(i = 0; i < rows; i++){
        free(array[i]);
    }
    free(array);
}

/**
 * @brief Calculates the Euclidean distance between two vectors.
 *
 * This function computes the Euclidean distance between two points in d-dimensional space.
 *
 * @param vec1 A pointer to the first vector.
 * @param vec2 A pointer to the second vector.
 * @param d The dimension of the vectors.
 * @return The Euclidean distance between vec1 and vec2.
 */
static double calc_ED(double* vec1, double* vec2, int d){
    double dist, diff;
    int i;

    dist = 0;
    for(i = 0; i < d; i++){ /*for each cord*/
        diff = vec1[i] - vec2[i];/*find the distance*/
        dist += diff * diff;/*add the square distance to the result*/
    }
    return sqrt(dist);/*the square root of the square distances sum*/
}

/**
 * @brief Checks if the centroids have converged.
 *
 * This function checks if the change in centroids is smaller than a given threshold.
 *
 * @param cents A pointer to the old centroids.
 * @param curr_cents A pointer to the current centroids.
 * @param k The number of centroids.
 * @param d The dimension of the centroids.
 * @param e The convergence threshold.
 * @return 1 if the change is small enough, 0 otherwise.
 */
static int is_delta_small_enough(double** cents, double** curr_cents, int k, int d, double e){
    int i;

    for(i = 0; i < k; i++){
        if(calc_ED(cents[i], curr_cents[i], d) >= e){
            return 0; /*not enough*/
        }
    }
    return 1; /*enough*/
}

/**
 * @brief Finds the index of the nearest centroid for a given vector.
 *
 * This function identifies which of the k centroids is closest to the given vector.
 *
 * @param cents A pointer to the centroids.
 * @param vector A pointer to the vector.
 * @param k The number of centroids.
 * @param d The dimension of the vector.
 * @return The index of the closest centroid.
 */
static int closest_cent_ind(double** cents, double* vector, int k, int d){
    int cent_ind,i;
    double min_dist, curr_dist;

    min_dist = calc_ED(vector, cents[0], d); /*set the distance from the first centroid as the initial minimum*/
    cent_ind = 0;

    for(i = 1; i < k; i++){ /*go over all the other centers*/
        curr_dist = calc_ED(vector, cents[i], d);
        if(curr_dist < min_dist){ /*update the minimum distance if found*/
            cent_ind = i;
            min_dist = curr_dist;
        }
    }
    
    return cent_ind;
}
/**
 * @brief Computes new centroids from the clusters formed by the vectors.
 *
 * This function updates the positions of centroids based on the assigned clusters.
 *
 * @param curr_cents A pointer to the current centroid positions.
 * @param vectors A pointer to the vectors.
 * @param d The dimension of the vectors.
 * @param k The number of centroids.
 * @param N The number of vectors.
 * @param clus_of An array of cluster assignments for each vector.
 */
static void update_cents(double** curr_cents, double** vectors, int d, int k, int N, int* clus_of){
    double** sum;
    double* size;
    int j,i,vec,cord,clus;

    /*allocate memory to save the sums and sizes of the clusters*/
    sum = (double**)malloc(k * sizeof(double*));
    for(i = 0; i < k; i++){
        sum[i] = (double*)malloc(d * sizeof(double));
    }
    size = (double*)malloc(k * sizeof(double));

    /*compute each cluster size and each cluster coordinate sum*/
    for(i = 0; i < k; i++){
        size[i] = 0;
        for(j = 0; j < d; j++){
            sum[i][j] = 0;
        }
    }
    
    for(vec = 0; vec < N; vec++){
        for(cord = 0; cord < d; cord++){
            sum[clus_of[vec]][cord] += vectors[vec][cord];
        }
        size[clus_of[vec]] = size[clus_of[vec]] + 1.0;
    }
   
   /*compute and update the centroids by the sums and the sizes*/
    for(clus = 0; clus < k; clus++){
        for(cord = 0; cord < d; cord++){
            curr_cents[clus][cord] = sum[clus][cord] / size[clus];
        }
    }

    /*free the memory allocated*/
    deep_free(sum, k);
    free(size);
    
}

/**
 * @brief Implements the k-means clustering algorithm.
 *
 * This function iteratively assigns vectors to clusters and updates centroids until convergence or until
 * the maximum number of iterations is reached.
 *
 * @param N The number of vectors.
 * @param d The dimension of each vector.
 * @param k The number of centroids (clusters).
 * @param iter The maximum number of iterations.
 * @param e The convergence threshold.
 * @param vectors A pointer to the input vectors.
 * @param cents A pointer to the initial centroids.
 * @return An array of cluster assignments for each vector.
 */
static int* kmeans(int N, int d, int k, int iter, double e, double** vectors, double** cents) {
    int num_iter, enough, clus, i, j, vec;
    double** curr_cents;
    int* clus_of;

    num_iter = 0;
    enough = 0;
    
    /*allocate memory for the computed centroids and the clusters*/
    clus_of = (int*)malloc(N * sizeof(int));
    curr_cents = (double**)malloc(k * sizeof(double*));
    for(i = 0; i < k; i++){
        curr_cents[i] = (double*)calloc(d, sizeof(double)); 
    }
    
    

    while(!enough && num_iter < iter){ /*until convergence or maximum iteration*/
        
         /*Assign every vectors to the closest cluster*/
        for(vec = 0; vec < N; vec++){
            clus = closest_cent_ind(cents, vectors[vec], k, d);
            clus_of[vec] = clus;
        }
        
        /*update the centroids by the new clusters*/
        update_cents(curr_cents, vectors, d, k, N, clus_of);

        /*check if the change was small enough*/
        enough = is_delta_small_enough(cents, curr_cents, k, d, e);

        num_iter++;

        /*save the compute centroids as the new centroids*/
        for(i = 0; i < k; i++){
            for(j = 0; j < d; j++){
                cents[i][j] = curr_cents[i][j];
            }
        }
    }
    
    /*free allocated memory*/
    deep_free(curr_cents, k);
    deep_free(cents, k);
    
    return clus_of;
}

/**
 * @brief The Python interface for fitting the k-means model.
 *
 * This function accepts input vectors and initial centroids from Python, and returns the final cluster 
 * assignments using the k-means algorithm.
 *
 * @param self The Python object reference.
 * @param args A tuple containing the Python arguments.
 * @return A Python list of cluster assignments for each vector.
 */
static PyObject* fit(PyObject* self, PyObject* args){
    PyObject *PyVectors, *PyCents, *PyVec, *PyCord, *PyLabels;
    int N, d, k, iter, vec, cord;
    double e;
    double** vectors;
    double** cents;
    int* labels;

    /*get the python arguments*/
    if (! PyArg_ParseTuple(args, "O!O!iiiid",&PyList_Type, &PyVectors, &PyList_Type, &PyCents, &N, &d, &k, &iter, &e)){
        return NULL;
    }

    /*allocate memory for the vectors and centroids*/
    vectors = (double**)malloc(N * sizeof(double*));
    for(vec = 0; vec < N; vec++){
        vectors[vec] = (double*)malloc(d * sizeof(double));
    }
    cents = (double**)malloc(k * sizeof(double*));
    for(vec = 0; vec < k; vec++){
        cents[vec] = (double*)malloc(d * sizeof(double));
    }

    /*convert the python vectors and centroids to c arrays*/
    for(vec = 0; vec < N; vec++){
        PyVec = PyList_GetItem(PyVectors, vec);
        for(cord = 0; cord < d; cord++){
            PyCord = PyList_GetItem(PyVec, cord);
            vectors[vec][cord] = PyFloat_AsDouble(PyCord);
        }
    }
    for(vec = 0; vec < k; vec++){
        PyVec = PyList_GetItem(PyCents, vec);
        for(cord = 0; cord < d; cord++){
            PyCord = PyList_GetItem(PyVec, cord);
            cents[vec][cord] = PyFloat_AsDouble(PyCord);
        }
    }

    /*compute the final centroids with the kmeans algorithm*/
    labels = kmeans(N, d, k, iter, e, vectors, cents);

    /*create the python list of the centroids*/
    PyLabels = PyList_New(N);
    for(vec = 0; vec < N; vec++){
            PyList_SetItem(PyLabels, vec, PyLong_FromLong(labels[vec]));
        }

    /*free allocated memory*/
    deep_free(vectors, N);
    free(labels);
 
    return PyLabels;
}

/*module methods*/
static PyMethodDef MyMethods[] = {
    {"fit", fit, METH_VARARGS, PyDoc_STR("calculate the centroids with the kmeans algorithm \n(vectors list, initial centers list, number of vectors, the dimension, number of centers to compute, maximum iteration, epsilon)")},
    {NULL, NULL, 0, NULL}
};

/*module definitions*/
static struct PyModuleDef mykmeanssp = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    MyMethods
};

PyMODINIT_FUNC PyInit_mykmeanssp(void){
    return PyModule_Create(&mykmeanssp);
}
