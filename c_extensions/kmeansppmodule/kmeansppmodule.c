#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void deep_free(double** array, int rows){
    int i;
    for(i = 0; i < rows; i++){
        free(array[i]);
    }
    free(array);
}

/*calculta the ecludian distance between two vectors*/
static double calc_ED(double* vec1, double* vec2, int d){
    double dist, diff;
    int i;

    dist = 0;
    for(i = 0; i < d; i++){ /*for each cord*/
        diff = vec1[i] - vec2[i];/*find the difrance*/
        dist += diff * diff;/*add the squre difrance to the result*/
    }
    return sqrt(dist);/*the squre root of the squre difrances sum*/
}

/*check if the new cetroids and the old ones are near enough to finish*/
static int is_delta_small_enough(double** cents, double** curr_cents, int k, int d, double e){
    int i;

    for(i = 0; i < k; i++){
        if(calc_ED(cents[i], curr_cents[i], d) >= e){
            return 0; /*not enough*/
        }
    }
    return 1; /*enough*/
}

/*find the index of the nearest center from the given vector*/
static int closest_cent_ind(double** cents, double* vector, int k, int d){
    int cent_ind,i;
    double min_dist, curr_dist;

    min_dist = calc_ED(vector, cents[0], d); /*set the distance from the first centes as the initial minimum*/
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
/*compute the new centroids given the clusters as the vectors and each vector ckuster index*/
static void update_cents(double** curr_cents, double** vectors, int d, int k, int N, int* clus_of){
    double** sum;
    double* size;
    int j,i,vec,cord,clus;

    /*alucate mmemory to save the sums and sizes of the clusters*/
    sum = (double**)malloc(k * sizeof(double*));
    for(i = 0; i < k; i++){
        sum[i] = (double*)malloc(d * sizeof(double));
    }
    size = (double*)malloc(k * sizeof(double));

    /*cumpute each cluster size and each cluster cordinate sum*/
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

    /*free the memmory allocated*/
    deep_free(sum, k);
    free(size);
    
}

/*the main kmeans algorythm implemantation*/
static int* kmeans(int N, int d, int k, int iter, double e, double** vectors, double** cents) {
    int num_iter, enough, clus, i, j, vec;
    double** curr_cents;
    int* clus_of;

    num_iter = 0;
    enough = 0;
    
    /*allocate memory for the cumputed centroids and the clusters*/
    clus_of = (int*)malloc(N * sizeof(int));
    curr_cents = (double**)malloc(k * sizeof(double*));
    for(i = 0; i < k; i++){
        curr_cents[i] = (double*)calloc(d, sizeof(double)); 
    }
    
    

    while(!enough && num_iter < iter){ /*until convergence or maximun itertation*/
        
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
    
    /*free alocated memory*/
    deep_free(curr_cents, k);
    deep_free(cents, k);
    
    return clus_of;
}


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

    /*allocta memory for the vectors and centroids*/
    vectors = (double**)malloc(N * sizeof(double*));
    for(vec = 0; vec < N; vec++){
        vectors[vec] = (double*)malloc(d * sizeof(double));
    }
    cents = (double**)malloc(k * sizeof(double*));
    for(vec = 0; vec < k; vec++){
        cents[vec] = (double*)malloc(d * sizeof(double));
    }

    /*convert the python vectors and cetroids to c arrays*/
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

    /*compute the final centroids with the kmeans algorythm*/
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

/*mudule methods*/
static PyMethodDef MyMethods[] = {
    {"fit", fit, METH_VARARGS, PyDoc_STR("clculate the centroids with the kmeans algorythm \n(vectors list, initial centers list, number of vectors, the dimension, number of centers to compute, maximum iteration, epsilon)")},
    {NULL, NULL, 0, NULL}
};

/*module defenitions*/
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
