#include <mpi.h>

#include <algorithm>
#include <format>
#include <print>
#include <random>
#include <ranges>
#include <vector>

int main(int argc, char** argv)
{
  using namespace std::views;
  using std::ranges::to, std::ranges::shuffle, std::ranges::sort;

  MPI_Init(&argc, &argv);

  int size {};
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank {};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<int> vector(size);

  if (rank == 0)
  {
    vector = iota(1) | take(size * size) | to<std::vector>();

    shuffle(vector, std::mt19937 { std::random_device {}() });

    std::println("Сортируемый массив");
    for (auto const& value : vector) std::print("{} ", value);
    std::println();
  }

  MPI_Comm communicator {};
  int dimensions[] { size }, periods[] { 0 };
  MPI_Cart_create(MPI_COMM_WORLD, 1, dimensions, periods, 0, &communicator);

  int previous_rank {}, next_rank {};
  MPI_Cart_shift(communicator, 0, 1, &previous_rank, &next_rank);

  std::vector<int> data(size), partner_data(size);
  MPI_Scatter(vector.data(), size, MPI_INT, data.data(), size, MPI_INT, 0,
              communicator);

  for (int phase {}; phase < size; ++phase)
  {
    // Локальная сортировка данных
    sort(data);

    // Определение партнера
    int partner { ((phase + rank) % 2 == 0) ? next_rank : previous_rank };

    // Партнер отсутствует
    if (partner < 0 || partner >= size) continue;

    // Отправка и получение данных партнера
    MPI_Sendrecv(data.data(), data.size(), MPI_INT, partner, 0,
                 partner_data.data(), partner_data.size(), MPI_INT, partner, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<int> temporary(size);

    // Партнер находится справа
    if (rank < partner)
    {
      int ours {}, theirs {};
      for (int i {}; i < size; i++)
        temporary.at(i) = (data.at(ours) < partner_data.at(theirs))
                              ? data.at(ours++)
                              : partner_data.at(theirs++);
    }
    else
    {
      int ours { size - 1 }, theirs { size - 1 };
      for (int i { size - 1 }; i >= 0; i--)
        temporary.at(i) = (data.at(ours) > partner_data.at(theirs))
                              ? data.at(ours--)
                              : partner_data.at(theirs--);
    }

    data = temporary;
  }

  MPI_Gather(data.data(), data.size(), MPI_INT, vector.data(), data.size(),
             MPI_INT, 0, communicator);

  if (rank == 0)
  {
    std::println("Результат сортировки");
    for (auto const& value : vector) std::print("{} ", value);
    std::println();
  }

  MPI_Finalize();
}