#include <mpi.h>

#include <print>
#include <ranges>
#include <vector>

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

  constexpr int A_ROWS { 4 }, A_COLS { 5 };
  std::vector<double> A(A_ROWS * A_COLS);

  constexpr int B_ROWS { 5 }, B_COLS { 6 };
  std::vector<double> B(B_ROWS * B_COLS);

  constexpr int C_ROWS { A_ROWS }, C_COLS { B_COLS };
  std::vector<double> C(C_ROWS * C_COLS);

  if (rank == 0)
  {
    // Заполнение матрицы А значениями
    std::ranges::copy(std::views::iota(1, 21), A.begin());

    std::println("Матрица A (4x5):");
    for (auto const& row : A | std::views::chunk(A_COLS))
    {
      for (auto const& element : row) std::print("{}\t", element);
      std::println();
    }

    // Заполнение матрицы B значениями
    std::ranges::copy(std::views::iota(1, 31), B.begin());

    std::println("Матрица B (5x6):");
    for (auto const& row : B | std::views::chunk(B_COLS))
    {
      for (auto const& element : row) std::print("{}\t", element);
      std::println();
    }
  }

  // Отправка элементов матрицы A некорневым процессам
  double A_element {};
  MPI_Scatter(A.data(), 1, MPI_DOUBLE, &A_element, 1, MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

  if (rank == 0)
    for (int destination { 1 };
         auto const& row : B | std::views::chunk(B_COLS) | std::views::drop(1))
      MPI_Send(row.data(), row.size(), MPI_DOUBLE, destination++, 0,
               MPI_COMM_WORLD);

  // 0 5 10 15
  // 1 6 11 16
  // 2 7 12 17
  // 3 8 13 18
  // 4 9 14 19

  MPI_Comm col_comm {};
  MPI_Comm_split(MPI_COMM_WORLD, rank % A_COLS, rank, &col_comm);

  int col_rank {};
  MPI_Comm_rank(col_comm, &col_rank);

  std::vector<double> B_row(B_COLS);

  if (rank == 0)
    B_row = B | std::views::take(B_row.size()) | std::ranges::to<std::vector>();

  if (rank != 0 && col_rank == 0)
    MPI_Recv(B_row.data(), B_row.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

  MPI_Bcast(B_row.data(), B_row.size(), MPI_DOUBLE, 0, col_comm);
  MPI_Comm_free(&col_comm);

  std::vector<double> local_row(B_COLS);
  for (int j = 0; j < B_row.size(); ++j) local_row[j] = A_element * B_row[j];

  // 0  1  2  3  4
  // 5  6  7  8  9
  // 10 11 12 13 14
  // 15 16 17 18 19

  MPI_Comm row_comm {};
  MPI_Comm_split(MPI_COMM_WORLD, rank / A_COLS, rank, &row_comm);

  std::vector<double> row_result(B_COLS);
  MPI_Reduce(local_row.data(), row_result.data(), local_row.size(), MPI_DOUBLE,
             MPI_SUM, 0, row_comm);

  if (rank == 0)
  {
    std::ranges::copy(row_result, C.begin());
    for (int r = 1; r < C_ROWS; ++r)
      MPI_Recv(C.data() + r * C_COLS, C_COLS, MPI_DOUBLE, r * A_COLS, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (rank != 0 && rank % A_COLS == 0)
    MPI_Send(row_result.data(), row_result.size(), MPI_DOUBLE, 0, 0,
             MPI_COMM_WORLD);

  MPI_Comm_free(&row_comm);

  if (rank == 0)
  {
    std::println("Матрица C (4x6):");
    for (auto const& row : C | std::views::chunk(C_COLS))
    {
      for (auto const& element : row) std::print("{}\t", element);
      std::println();
    }
  }

  MPI_Finalize();
}
