#include <iostream>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#define min(A,B) ((A) < (B) ? (A) : (B))
#define max(A,B) ((A) > (B) ? (A) : (B))

int main(int argc, char* argv[])
{
    // fase 0: inicializacion MPI
    MPI_Init(&argc, &argv);

    int s, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &s);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // n e itmax ahora entran por consola:
    // mpirun -np <p> ./heat_mpi <n> <itmax>
    int n = 80;
    int itmax = 20000;

    if (argc >= 2) n = std::atoi(argv[1]);
    if (argc >= 3) itmax = std::atoi(argv[2]);

    if (n < 2 || itmax < 1) {
        if (s == 0) {
            std::cerr << "Uso: mpirun -np <p> ./heat_mpi <n> <itmax>\n";
            std::cerr << "Ejemplo: mpirun -np 4 ./heat_mpi 200 20000\n";
            std::cerr << "Restricciones: n >= 2, itmax >= 1\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (p > n - 1) {
        if (s == 0) {
            std::cerr << "Error: el numero de procesos debe ser <= n-1, "
                      << "porque solo hay n-1 filas interiores.\n";
        }
        MPI_Finalize();
        return 1;
    }

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
    int iterations_done = 0;

    // Conteo de FLOPs del stencil numerico por punto actualizado:
    // vertical:   (a + b - 2*c) * dy2i  -> 4 FLOPs
    // horizontal: (d + e - 2*f) * dx2i  -> 4 FLOPs
    // suma vertical + horizontal         -> 1 FLOP
    // dphi *= dt                         -> 1 FLOP
    // phi_new = phi + dphi               -> 1 FLOP
    // Total aproximado: 11 FLOPs por punto interior actualizado.
    const long long FLOPS_PER_POINT = 11;
    const long long local_points = 1LL * local_rows * (n - 1);
    long long local_flops = 0;

    if (s == 0) {
        printf("\nTransmision de calor 2D con MPI\n");
        printf("dx = %12.4g, dy = %12.4g, dt = %12.4g, eps = %12.4g\n",
               dx, dy, dt, eps);
        printf("n = %d, itmax = %d\n", n, itmax);
        printf("Procesos MPI: %d\n", p);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // fase 4: iteracion
    for (it = 1; it <= itmax; it++) {
        iterations_done = it;
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
        // NOTA: il=1 e il=local_rows son filas interiores globales, no fronteras fijas.
        // Las fronteras fisicas reales estan en los halos:
        //   - PHI(0, k) para el proceso 0
        //   - PHI(local_rows + 1, k) para el ultimo proceso
        auto compute_local_row = [&](int il) {
            for (int k = 1; k < n; k++) {
                double dphi = (PHI(il+1,k) + PHI(il-1,k) - 2.0*PHI(il,k)) * dy2i
                            + (PHI(il,k+1) + PHI(il,k-1) - 2.0*PHI(il,k)) * dx2i;
                dphi *= dt;
                dphimax_local = max(dphimax_local, fabs(dphi));
                PHIN(il, k) = PHI(il, k) + dphi;
            }
        };

        if (local_rows >= 1) compute_local_row(1);
        if (local_rows >= 2) compute_local_row(local_rows);

        // Se actualizan local_rows * (n-1) puntos por iteracion en este proceso.
        local_flops += FLOPS_PER_POINT * local_points;

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

    // Checksum local del resultado final.
    // Solo se suman las filas interiores reales de cada proceso y las columnas interiores.
    // No se incluyen halos para evitar duplicar informacion entre procesos vecinos.
    double local_checksum = 0.0;
    for (int il = 1; il <= local_rows; il++) {
        for (int k = 1; k < n; k++) {
            local_checksum += PHI(il, k);
        }
    }

    // El tiempo paralelo correcto es el maximo entre procesos,
    // porque el programa termina cuando termina el proceso mas lento.
    double t_max = 0.0;
    long long global_flops = 0;
    double global_checksum = 0.0;

    MPI_Reduce(&t_elapsed, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flops, &global_flops, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (s == 0) {
        double gflops = global_flops / t_max / 1.0e9;
        printf("\n%d iteraciones\n", iterations_done);
        printf("Tiempo MPI max = %#12.4g seg\n", t_max);
        printf("FLOPs totales  = %lld\n", global_flops);
        printf("GFLOP/s        = %#12.4g\n", gflops);
        printf("Checksum phi   = %.12e\n", global_checksum);
    }

    // fase final
    MPI_Type_free(&T_fila);
    delete[] phi;
    delete[] phin;
    MPI_Finalize();
    return 0;
}