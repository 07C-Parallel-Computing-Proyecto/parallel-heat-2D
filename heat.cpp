#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <mpi.h>

#define min(A,B) ((A) < (B) ? (A) : (B))
#define max(A,B) ((A) > (B) ? (A) : (B))

const int n     = 80;
const int itmax = 20000;

int main(int argc, char* argv[])
{
    // fase 0: inicializacion MPI
    MPI_Init(&argc, &argv);

    int s, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &s);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    double eps = 1.0e-08;
    double dx, dy, dx2, dy2, dx2i, dy2i, dt;

    dx  = 1.0 / n;
    dy  = 1.0 / n;
    dx2 = dx * dx;
    dy2 = dy * dy;
    dx2i = 1.0 / dx2;
    dy2i = 1.0 / dy2;
    dt   = min(dx2, dy2) / 4.0;

    // fase 1: Descomposición desigual
    int n_interior = n - 1;             // filas interiores globales (1..n-1)
    int base = n_interior / p;
    int rem  = n_interior % p;

    int local_rows, i_start;
    if (s < rem) {
        local_rows = base + 1;
        i_start    = 1 + s * (base + 1);
    } else {
        local_rows = base;
        i_start    = 1 + rem * (base + 1) + (s - rem) * base;
    }
    int i_end = i_start + local_rows - 1;

    // fase 2: Memoria con halos
    // phi[0..local_rows+1][0..n]  (halos en filas 0 y local_rows+1)
    // Índice lineal: fila il → il*(n+1)+k
    int rows_with_halo = local_rows + 2;
    double* phi  = new double[rows_with_halo * (n + 1)]();
    double* phin = new double[rows_with_halo * (n + 1)]();

    auto PHI  = [&](int il, int k) -> double& { return phi [il*(n+1)+k]; };
    auto PHIN = [&](int il, int k) -> double& { return phin[il*(n+1)+k]; };

    // borde k=n (derecha): phi[i][n] = 1.0 para todos
    for (int il = 0; il < rows_with_halo; il++)
        PHI(il, n) = 1.0;

    // proceso 0: halo superior (fila global i=0)
    if (s == 0) {
        PHI(0, 0) = 0.0;
        for (int k = 1; k < n; k++)
            PHI(0, k) = PHI(0, k-1) + dx;   // phi[0][k] = k*dx
        PHI(0, n) = 1.0;
    }

    // proceso p-1: halo inferior (fila global i=n)
    if (s == p - 1) {
        PHI(local_rows + 1, 0) = 0.0;
        for (int k = 1; k < n; k++)
            PHI(local_rows + 1, k) = PHI(local_rows + 1, k-1) + dx;
        PHI(local_rows + 1, n) = 1.0;
    }

    // fase 3: Tipo derivado MPI
    MPI_Datatype T_fila;
    MPI_Type_contiguous(n + 1, MPI_DOUBLE, &T_fila);
    MPI_Type_commit(&T_fila);

    MPI_Request req[4];
    double dphimax_local, dphimax_global;
    int it;

    if (s == 0) {
        printf("\nTransmision de calor 2D con MPI\n");
        printf("dx = %12.4g, dy = %12.4g, dt = %12.4g, eps = %12.4g\n",
               dx, dy, dt, eps);
        printf("Procesos MPI: %d\n", p);
    }

    double t_start = MPI_Wtime();

    // fase 4: iteracion
    for (it = 1; it <= itmax; it++) {
        dphimax_local = 0.0;
        int nreq = 0;

        // comunicacion no bloqueante
        if (s > 0) {
            MPI_Isend(&PHI(1, 0), 1, T_fila, s-1, 0, MPI_COMM_WORLD, &req[nreq++]);
            MPI_Irecv(&PHI(0, 0), 1, T_fila, s-1, 1, MPI_COMM_WORLD, &req[nreq++]);
        }
        if (s < p - 1) {
            MPI_Isend(&PHI(local_rows, 0),   1, T_fila, s+1, 1, MPI_COMM_WORLD, &req[nreq++]);
            MPI_Irecv(&PHI(local_rows+1, 0), 1, T_fila, s+1, 0, MPI_COMM_WORLD, &req[nreq++]);
        }

        // computo interior (filas que NO dependen de halos)
        if (local_rows > 2) {
            for (int il = 2; il <= local_rows - 1; il++) {
                for (int k = 1; k < n; k++) {
                    double dphi = (PHI(il+1,k) + PHI(il-1,k) - 2.0*PHI(il,k)) * dy2i
                                + (PHI(il,k+1) + PHI(il,k-1) - 2.0*PHI(il,k)) * dx2i;
                    dphi *= dt;
                    dphimax_local = max(dphimax_local, fabs(dphi));
                    PHIN(il, k) = PHI(il, k) + dphi;
                }
            }
        }

        // esperar a que lleguen los halos
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

        // computo de fronteras locales (filas 1 y local_rows)
        int border_rows[2] = {1, local_rows};
        for (int b = 0; b < 2; b++) {
            int il = border_rows[b];
            if (il == 1          && s == 0)     continue;
            if (il == local_rows && s == p-1)   continue;

            for (int k = 1; k < n; k++) {
                double dphi = (PHI(il+1,k) + PHI(il-1,k) - 2.0*PHI(il,k)) * dy2i
                            + (PHI(il,k+1) + PHI(il,k-1) - 2.0*PHI(il,k)) * dx2i;
                dphi *= dt;
                dphimax_local = max(dphimax_local, fabs(dphi));
                PHIN(il, k) = PHI(il, k) + dphi;
            }
        }

        // reduccion global del criterio de convergencia
        MPI_Allreduce(&dphimax_local, &dphimax_global, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // actualizacion
        for (int il = 1; il <= local_rows; il++)
            for (int k = 1; k < n; k++)
                PHI(il, k) = PHIN(il, k);

        if (dphimax_global < eps) break;
    }

    double t_elapsed = MPI_Wtime() - t_start;

    if (s == 0) {
        printf("\n%d iteraciones\n", it);
        printf("Tiempo MPI = %#12.4g seg\n", t_elapsed);
    }

    // fase final
    MPI_Type_free(&T_fila);
    delete[] phi;
    delete[] phin;
    MPI_Finalize();
    return 0;
}