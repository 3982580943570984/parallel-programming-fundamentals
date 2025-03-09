#include <mpi.h>

#include <print>
#include <ranges>
#include <vector>

int main(int argc, char** argv)
{
  using namespace std::views;
  using std::ranges::to;

  MPI_Init(&argc, &argv);

  int size {};
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank {};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> matrix(size * size), vector(size), column(size);

  if (rank == 0)
  {
    matrix = iota(0) | take(size * size) | to<std::vector>();

    vector = iota(0) | take(size) | to<std::vector>();

    std::println("Исходная матрица:");
    for (auto const& row : matrix | chunk(size))
    {
      for (auto const& value : row) std::print("{} ", value);
      std::println();
    }

    std::println("Исходный вектор:");
    for (auto const& value : vector) std::print("{} ", value);
    std::println();
  }

  std::vector<int> transposed_matrix(matrix.size());

  for (int i {}; i < size; ++i)
    for (int j {}; j < size; ++j)
      transposed_matrix[j * size + i] = matrix[i * size + j];

  MPI_Scatter(transposed_matrix.data(), size, MPI_INT, column.data(),
              column.size(), MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Comm communicator {};
  int dimensions[] { size }, periods[] { 1 };
  MPI_Cart_create(MPI_COMM_WORLD, 1, dimensions, periods, 0, &communicator);

  int previous_rank {}, next_rank {};
  MPI_Cart_shift(communicator, 0, 1, &previous_rank, &next_rank);

  std::vector<int> results(size);
  auto& result { results.front() };

  for (int i {}; i < size; ++i)
  {
    int element {};

    rank == 0 ? element = vector.at(i)
              : MPI_Recv(&element, 1, MPI_INT, previous_rank, 0, communicator,
                         MPI_STATUS_IGNORE);

    result += column.at(i) * element;

    if (next_rank > 0)
      MPI_Send(&element, 1, MPI_INT, next_rank, 0, communicator);
  }

  if (rank == 0)
    for (int i { 1 }; i < size; ++i)
      MPI_Recv(&results[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  else
    MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    std::print("Результат: ");
    for (auto const& result : results) std::print("{} ", result);
    std::println();
  }

  MPI_Finalize();
}