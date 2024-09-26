#ifndef SYMNMF_H
#define SYMNMF_H


extern double** C_symnmf(double** W, double** H,int N,int k);
extern double** C_ddg(double** X,int N, int d);
extern double** C_norm(double** X,int  N, int d);
extern double** C_sym(double** X, int N, int d);
extern void deep_free(double** M, int num_rows);
void printM(double** M,int N);

#endif  /* !SYMNMF_H */
