#include <mpi.h>

#include <array>
#include <print>

// A: {{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15},{16,17,18,19,20}}
// B: {{1,2,3,4,5,6},{7,8,9,10,11,12},{13,14,15,16,17,18},{19,20,21,22,23,24},{25,26,27,28,29,30}}
// C: {{255, 270, 285, 300, 315, 330}, {580, 620, 660, 700, 740, 780}, {905, 970, 1035, 1100, 1165, 1230}, {1230, 1320, 1410, 1500, 1590, 1680}}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank {};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size {};
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::array<double, 20> A {};  // 4*5

  std::array<double, 30> B {};  // 5*6

  if (rank == 0)
  {
    for (int i {}; i < 20; ++i) A[i] = i + 1;

    std::println("Матрица A (4x5):");
    for (int i {}; i < 4; ++i)
    {
      for (int j {}; j < 5; ++j) std::print("{}\t", A[i * 5 + j]);
      std::println();
    }

    for (int i {}; i < 30; ++i) B[i] = i + 1;

    std::println("Матрица B (5x6):");
    for (int i {}; i < 5; ++i)
    {
      for (int j {}; j < 6; ++j) std::print("{}\t", B[i * 6 + j]);
      std::println();
    }
  }

  std::array<double, 5> A_row {};
  MPI_Scatter(A.data(), A_row.size(), MPI_DOUBLE, A_row.data(), A_row.size(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Bcast(B.data(), B.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::array<double, 6> C_row {};
  for (int j {}; j < 6; ++j)
    for (int k {}; k < 5; ++k) C_row[j] += A_row[k] * B[k * 6 + j];

  std::array<double, 24> C {};  // 4*6
  MPI_Gather(C_row.data(), C_row.size(), MPI_DOUBLE, C.data(), C_row.size(),
             MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    std::println("Матрица C (4x6):");
    for (int row {}; row < 4; ++row)
    {
      for (int col {}; col < 6; ++col) std::print("{}\t", C[row * 6 + col]);
      std::println();
    }
  }

  MPI_Finalize();
}
