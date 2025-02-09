#include <semaphore.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <functional>
#include <print>
#include <random>
#include <ranges>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using time_point = std::chrono::system_clock::time_point;

enum class Fuel
{
  AI76,
  AI92,
  AI95,
};

struct Column
{
  void serve();

  Fuel fuel {};
  double mean_service_time {};
  double standard_deviation {};

  std::thread thread {};
};

struct Car
{
  int id {};
  Fuel fuel {};
  time_point timestamp {};
};

sem_t mutex {};

struct Queue
{
  explicit Queue() = default;

  ~Queue() { sem_destroy(&mutex); }

  static auto lock() { sem_wait(&mutex); }

  static auto unlock() { sem_post(&mutex); }

  [[nodiscard]] std::optional<Car> find_nearest_car(const Fuel& fuel)
  {
    std::optional<Car> result {};

    lock();

    const auto element { std::ranges::find_if(
        cars, [&](const auto car) { return car.fuel == fuel; }) };

    if (element != cars.cend())
    {
      result = *element;
      cars.erase(element);
    }

    unlock();

    return result;
  }

  auto insert_car(const Car& car)
  {
    lock();

    if (cars.size() < max_size) cars.emplace_back(car);

    unlock();
  }

  std::vector<Car> cars {};

  static constexpr auto max_size { 200 };
};

struct Generator
{
  explicit Generator() = default;

  void generate();

  static constexpr auto requests { 150 };
  static constexpr auto mean_generation_time { 1 };
  static constexpr auto standard_deviation { 0.5 };

  std::thread thread {};
};

std::array columns {
  Column {
          .fuel = Fuel::AI76,
          .mean_service_time = 10,
          .standard_deviation = 0.5,
          },
  Column {
          .fuel = Fuel::AI76,
          .mean_service_time = 10,
          .standard_deviation = 0.5,
          },
  Column {
          .fuel = Fuel::AI92,
          .mean_service_time = 12.5,
          .standard_deviation = 0.6,
          },
  Column {
          .fuel = Fuel::AI92,
          .mean_service_time = 12.5,
          .standard_deviation = 0.6,
          },
  Column {
          .fuel = Fuel::AI95,
          .mean_service_time = 15,
          .standard_deviation = 0.7,
          },
};

Queue* queue { nullptr };

constexpr auto fuel_types { 3 };

std::array<sem_t, fuel_types> fuel_semaphores;

void Generator::generate()
{
  thread = std::thread {
    [&] {
      std::random_device random_device;
      std::mt19937 number_generator(random_device());
      std::normal_distribution<> distribution(mean_generation_time,
                                              standard_deviation);

      for (int request { 0 }; request < requests; ++request)
      {
        const auto sleep_duration { std::chrono::duration<double>(
            distribution(number_generator)) };

        // std::println("Sleeping for {}", sleep_duration);

        std::this_thread::sleep_for(sleep_duration);

        const auto fuel { std::invoke([&] {
          std::uniform_int_distribution<> distribution(0, 4);
          const auto value { distribution(number_generator) };
          return value < 2 ? Fuel::AI76 : value < 4 ? Fuel::AI92 : Fuel::AI95;
        }) };

        // std::println("Generated fuel type");

        Car car {
          .id = request,
          .fuel = fuel,
          .timestamp = std::chrono::system_clock::now(),
        };

        queue->insert_car(car);

        std::println("Created car entry: {} {} {}", car.id,
                     std::to_underlying(car.fuel),
                     car.timestamp.time_since_epoch());

        // const auto fuel_as_int { std::to_underlying(fuel) };

        sem_post(&fuel_semaphores[std::to_underlying(fuel)]);

        // int sem_val = 0;
        // sem_getvalue(&fuel_semaphores[fuel_as_int], &sem_val);
        // std::println("Incrementing semaphore with index: {}", fuel_as_int);
        // std::println("Semaphore {} value: {}", fuel_as_int, sem_val);
      }
    },
  };
}

void Column::serve()
{
  thread = std::thread {
    [&] {
      std::random_device random_device;
      std::mt19937 number_generator(random_device());
      std::normal_distribution<> distribution(mean_service_time,
                                              standard_deviation);

      while (true)
      {
        std::println("Decrementing fuel semaphore");

        // const auto fuel_as_int { std::to_underlying(fuel) };

        sem_wait(&fuel_semaphores[std::to_underlying(fuel)]);

        // int sem_val = 0;
        // sem_getvalue(&fuel_semaphores[fuel_as_int], &sem_val);
        // std::println("Semaphore {} value: {}", fuel_as_int, sem_val);

        std::println("Searching for nearest car");

        const auto car { queue->find_nearest_car(fuel) };

        if (!car) continue;

        std::println("{} Started car service: {} {} {}",
                     std::this_thread::get_id(), car->id,
                     std::to_underlying(car->fuel),
                     car->timestamp.time_since_epoch());

        std::this_thread::sleep_for(
            std::chrono::duration<double>(distribution(number_generator)));

        std::println("Stopped car service");
      }
    },
  };
}

};  // namespace

int main(int argc, char** argv)
{
  sem_init(&mutex, 1, 1);

  const auto shm_id { shmget(ftok(".", 'S'), sizeof(Queue), IPC_CREAT | 0666) };

  void* raw_memory { shmat(shm_id, nullptr, 0) };

  queue = new (raw_memory) Queue;

  for (auto& fuel_semaphore : fuel_semaphores) sem_init(&fuel_semaphore, 0, 0);

  Generator generator {};
  generator.generate();

  for (auto& column : columns) column.serve();

  generator.thread.join();

  for (auto& column : columns) column.thread.join();

  shmdt(queue);

  shmctl(shm_id, IPC_RMID, nullptr);

  for (auto& fuel_semaphore : fuel_semaphores) sem_destroy(&fuel_semaphore);

  sem_destroy(&mutex);
}