#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <iomanip>
#include "random"
#include <stdlib.h>
#include <random>
#include <cstdlib>
#include <ctime>
#include <typeinfo>
#include <armadillo>
#include <cmath>
#include <string>
#include "mpi.h"

using namespace std;
ofstream ofile;

// Declaring functions
void MCnoSaving(int mcs,
                int N,
                double epsilon,
                double netWorth,
                double *m);

void MCwithSaving(int mcs,
                  int N,
                  double epsilon,
                  double *m,
                  double lambda,
                  double delta);

void MCneighbors(int mcs,
                 int N,
                 double epsilon,
                 double *m,
                 double lambda,
                 double delta,
                 double delta_m,
                 double faktor,
                 double alpha,
                 double p_ij,
                 double number,
                 double difference);

void MCpartners(int mcs,
                int N,
                double epsilon,
                double *m,
                double lambda,
                double delta,
                double delta_m,
                double faktor,
                double alpha,
                double p_ij,
                double number,
                double difference,
                double gamma,
                int **c,
                int **maxValue);

void histogram(int N,
               double delta_m,
               int indeks,
               double *m,
               int *w);

void initialDistribution(int N,
                         double initialAmount,
                         double *m);

int main(int argc, char *argv[])
{

    // Variables
    int experiments = 1e2;
    int N = 1000;
    double faktor = 1.0/10000.0;
    double alpha = 2.0;
    double gamma = 4.0;
    double lambda = 0.5;
    int mcs = 1e7;
    double delta_m = 0.01;
    int n = N/delta_m;
    int indeks;
    double initialAmount = 1.0;
    double epsilon, netWorth, delta;
    char* outfilename;
    int NProcesses, RankProcess;
    double p_ij, number, difference;

    //  MPI initializations
    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &NProcesses);
    MPI_Comm_rank (MPI_COMM_WORLD, &RankProcess);

    if (RankProcess == 0 && argc < 2) {
        cout << "Bad Usage: " << argv[0] << " read in output file" << endl;
        exit(1);
    }

    if (RankProcess == 0 && argc == 2) {
        outfilename = argv[1];
        ofile.open(outfilename);
    }

    // Vectors
    double *m = new double[N];
    double *w_final = new double[n];
    double *Total_w = new double[n];
    int *w = new int[n];

    for (int i=0;i<n;i++) {
        w[i] = 0;
    }

    // Matrix
    int **c = new int*[N];
    for(int i=0;i<N;i++) {
        c[i] = new int[N];
    }

    int **maxValue = new int*[N];
    for(int i=0;i<N;i++) {
        maxValue[i] = new int[N];
    }

    for (int i=0;i<N;i++) {
        for (int j=0;j<N;j++) {
            c[i][j] = 0;
            maxValue[i][j] = 0;
        }
    }

    // Independent experiments
    for (int experiment=1;experiment<=experiments;experiment++) {
        // Setting all fortunes equal before each experiment
        initialDistribution(N, initialAmount, m);
        // Distributing wealth
        //MCnoSaving(mcs, N, epsilon, netWorth, m);
        //MCwithSaving(mcs, N, epsilon, m, lambda, delta);
        //MCneighbors(mcs, N, epsilon, m, lambda, delta, delta_m, faktor, alpha, p_ij, number, difference);
        MCpartners(mcs, N, epsilon, m, lambda, delta, delta_m, faktor, alpha, p_ij, number, difference, gamma, c, maxValue);
        // Adding to histogram
        histogram(N, delta_m, indeks, m, w);
    }

    // Final distribution/histogram
    for(int i=0;i<n;i++) {
        w_final[i] = ((double) w[i]) / ((double) experiments);
    }

    for(int i=0;i<n; i++){
        MPI_Reduce(&w_final[i], &Total_w[i], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    for (int i=0;i<n;i++) {
        Total_w[i] /= ((double) NProcesses);
    }

    if (RankProcess == 0) {
        for(int i=0;i<n;i++) {
            ofile << setw(15) << setprecision(8) << Total_w[i] << endl;
        }
    }

    // Closing outputfile
    if (RankProcess == 0) {
        ofile.close();
    }

    // End MPI
    MPI_Finalize ();

    // Deleting matrix
    for(int i=0;i<N;i++) {
            delete [] c[i];
    }
    for(int i=0;i<N;i++) {
            delete [] maxValue[i];
    }


    // Deleting vectors
    delete [] maxValue;
    delete [] c;
    delete [] m;
    delete [] w;
    delete [] w_final;
    delete [] Total_w;
}

void MCnoSaving(int mcs,
                int N,
                double epsilon,
                double netWorth,
                double *m)
{
    // Initializing the seed and calling the Mersienne algorithm
    random_device rd;
    mt19937_64 gen(rd());
    uniform_real_distribution<double> RNG(0.0,1.0);

    // MC-simulation
    for (int k=1;k<=mcs;k++) {
        int i = floor(RNG(gen)*N);
        int j = floor(RNG(gen)*N);
        epsilon = RNG(gen);
        netWorth = m[i] + m[j];
        m[i] = epsilon*netWorth;
        m[j] = (1.0-epsilon)*netWorth;
    }
}

void MCwithSaving(int mcs,
                  int N,
                  double epsilon,
                  double *m,
                  double lambda,
                  double delta)
{
    // Initializing the seed and calling the Mersienne algorithm
    random_device rd;
    mt19937_64 gen(rd());
    uniform_real_distribution<double> RNG(0.0,1.0);

    // MC-simulation
    for (int k=1;k<=mcs;k++) {
        int i = floor(RNG(gen)*N);
        int j = floor(RNG(gen)*N);
        epsilon = RNG(gen);
        delta = (1.0-lambda)*(epsilon*m[j] - (1.0-epsilon)*m[i]);
        m[i] += delta;
        m[j] -= delta;
    }
}

void MCneighbors(int mcs,
                 int N,
                 double epsilon,
                 double *m,
                 double lambda,
                 double delta,
                 double delta_m,
                 double faktor,
                 double alpha,
                 double p_ij,
                 double number,
                 double difference)
{
    // Initializing the seed and calling the Mersienne algorithm
    random_device rd;
    mt19937_64 gen(rd());
    uniform_real_distribution<double> RNG(0.0,1.0);

    // MC-simulation
    for (int k=1;k<=mcs;k++) {
        int i = floor(RNG(gen)*N);
        int j = floor(RNG(gen)*N);

        if (m[i] - m[j] == 0) {
            p_ij = 1.0;
        }
        else {
            difference = fabs(m[i] - m[j]);
            number = (ceil(difference/delta_m))/100.0;
            p_ij = faktor*pow(number,-alpha);
        }

        if (RNG(gen)<p_ij) {
            epsilon = RNG(gen);
            delta = (1.0-lambda)*(epsilon*m[j] - (1.0-epsilon)*m[i]);
            m[i] += delta;
            m[j] -= delta;
        }
    }
}

void MCpartners(int mcs,
                int N,
                double epsilon,
                double *m,
                double lambda,
                double delta,
                double delta_m,
                double faktor,
                double alpha,
                double p_ij,
                double number,
                double difference,
                double gamma,
                int **c,
                int **maxValue)
{
    // Initializing the seed and calling the Mersienne algorithm
    random_device rd;
    mt19937_64 gen(rd());
    uniform_real_distribution<double> RNG(0.0,1.0);

    for (int i=0;i<N;i++) {
        for (int j=0;j<N;j++) {
            c[i][j] = 0;
            maxValue[i][j] = 0;
        }
    }

    // MC-simulation
    for (int k=1;k<=mcs;k++) {

        int i = floor(RNG(gen)*N);
        int j = floor(RNG(gen)*N);

        if(i == j)
            continue;

        if (m[i] - m[j] == 0) {
            p_ij = 1.0;
        }

        else {
            difference = fabs(m[i] - m[j]);
            number = (ceil(difference/delta_m))/100.0;
            p_ij = faktor*pow(number,-alpha)*pow(((double) (c[i][j]+1))/((double) (maxValue[i][j]+1)),gamma);
        }

        if (RNG(gen) < p_ij) {
            epsilon = RNG(gen);
            delta = (1.0-lambda)*(epsilon*m[j] - (1.0-epsilon)*m[i]);
            m[i] += delta;
            m[j] -= delta;
            c[i][j] += 1;
        }

        maxValue[i][j] += 1;

    }

}

void histogram(int N,
               double delta_m,
               int indeks,
               double *m,
               int *w)
{
    for(int i=0;i<N;i++) {
        indeks = floor(m[i]/delta_m);
        w[indeks] += 1;
    }
}

void initialDistribution(int N,
                         double initialAmount,
                         double *m)
{
    for (int i=0;i<N;i++) {
        m[i] = initialAmount;
    }
}
