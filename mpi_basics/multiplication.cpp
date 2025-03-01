#include <mpi.h>

#include <print>
#include <ranges>
#include <vector>

// A: {{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15},{16,17,18,19,20}}
// B: {{1,2,3,4,5,6},{7,8,9,10,11,12},{13,14,15,16,17,18},{19,20,21,22,23,24},{25,26,27,28,29,30}}
// C: {{255, 270, 285, 300, 315, 330}, {580, 620, 660, 700, 740, 780}, {905, 970, 1035, 1100, 1165, 1230}, {1230, 1320, 1410, 1500, 1590, 1680}}

// 0 5 10 15
// 1 6 11 16
// 2 7 12 17
// 3 8 13 18
// 4 9 14 19

// 0  1  2  3  4
// 5  6  7  8  9
// 10 11 12 13 14
// 15 16 17 18 19

int main(int argc, char** argv)
{
  using namespace std::views;
  using namespace std::ranges;

  MPI_Init(&argc, &argv);

  int rank {};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size {};
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<double> A {}, B {}, C(4 * 6);  // 4*5, 5*6, 4*6

  std::vector<double> transposed(6 * 5);

  if (rank == 0)
  {
    A = iota(1) | take(4 * 5) | to<std::vector<double>>();

    B = iota(1) | take(5 * 6) | to<std::vector<double>>();

    for (int i {}; i < 5; ++i)
      for (int j {}; j < 6; ++j) transposed[j * 5 + i] = B[i * 6 + j];

    std::println("Матрица A (4x5):");
    for (auto const& row : A | chunk(5))
    {
      for (auto const& value : row) std::print("{}\t", value);
      std::println();
    }

    std::println("Матрица B (5x6):");
    for (auto const& row : B | chunk(6))
    {
      for (auto const& value : row) std::print("{}\t", value);
      std::println();
    }
  }

  MPI_Comm col_comm {}, row_comm {};
  MPI_Comm_split(MPI_COMM_WORLD, rank % 5, rank, &col_comm);
  MPI_Comm_split(MPI_COMM_WORLD, rank / 5, rank, &row_comm);

  double element {};
  MPI_Scatter(A.data(), 1, MPI_DOUBLE, &element, 1, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  std::array<double, 6> local {};

  for (auto const& [result, column] : zip(local, transposed | chunk(5)))
  {
    if (rank % 5 == 0)
      MPI_Bcast(column.data(), column.size(), MPI_DOUBLE, 0, col_comm);

    double other_element {};
    MPI_Scatter(column.data(), 1, MPI_DOUBLE, &other_element, 1, MPI_DOUBLE, 0,
                row_comm);

    double sum { element * other_element };
    MPI_Reduce(&sum, &result, 1, MPI_DOUBLE, MPI_SUM, 0, row_comm);
  }

  MPI_Gather(local.data(), local.size(), MPI_DOUBLE, C.data(), local.size(),
             MPI_DOUBLE, 0, col_comm);

  MPI_Comm_free(&col_comm);
  MPI_Comm_free(&row_comm);

  if (rank == 0)
  {
    std::println("Матрица C (4x6):");
    for (auto const& row : C | chunk(6))
    {
      for (auto const& value : row) std::print("{}\t", value);
      std::println();
    }
  }

  MPI_Finalize();
}
