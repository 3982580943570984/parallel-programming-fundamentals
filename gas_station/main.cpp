#include <semaphore.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <fstream>
#include <functional>
#include <limits>
#include <nlohmann/detail/macro_scope.hpp>
#include <print>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

using time_point = std::chrono::system_clock::time_point;

enum class Fuel
{
  AI76,
  AI92,
  AI95,
  COUNT,
};

template <>
struct std::formatter<Fuel> : std::formatter<std::string_view>
{
  auto format(Fuel fuel, std::format_context& ctx) const -> decltype(ctx.out())
  {
    std::string_view name {};

    if (fuel == Fuel::AI76)
      name = "АИ76";
    else if (fuel == Fuel::AI92)
      name = "АИ92";
    else if (fuel == Fuel::AI95)
      name = "АИ95";
    else
      name = "Unknown";

    return formatter<std::string_view>::format(name, ctx);
  }
};

inline void from_json(const nlohmann::json& j, Fuel& fuel)
{
  auto str = j.get<std::string>();
  if (str == "АИ76")
    fuel = Fuel::AI76;
  else if (str == "АИ92")
    fuel = Fuel::AI92;
  else if (str == "АИ95")
    fuel = Fuel::AI95;
  else
    throw std::runtime_error("Unknown fuel type: " + str);
}

struct Column
{
  void serve(int);

  Fuel fuel {};

  double mean_service_time {};

  double standard_deviation {};

  std::thread thread {};

  friend void from_json(const nlohmann ::json& nlohmann_json_j,
                        Column& nlohmann_json_t)
  {
    const Column nlohmann_json_default_obj {};
    nlohmann_json_t.fuel =
        nlohmann_json_j.value("fuel", nlohmann_json_default_obj.fuel);
    nlohmann_json_t.mean_service_time = nlohmann_json_j.value(
        "mean_service_time", nlohmann_json_default_obj.mean_service_time);
    nlohmann_json_t.standard_deviation = nlohmann_json_j.value(
        "standard_deviation", nlohmann_json_default_obj.standard_deviation);
  };
};

struct Car
{
  int id {};
  Fuel fuel {};
  time_point timestamp {};
};

struct Queue
{
  explicit Queue() { sem_init(&mutex, 1, 1); };

  ~Queue()
  {
    sem_destroy(&mutex);
    std::fclose(inserted_cars);
    std::fclose(dropped_cars);
  }

  auto lock() { sem_wait(&mutex); }

  auto unlock() { sem_post(&mutex); }

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

    if (cars.size() < max_size)
    {
      cars.emplace_back(car);

      const auto formatted_string { std::format(
          "Машина попала в очередь: номер - {}, тип топлива - {}, "
          "временная метка - {:%Y-%m-%d %H:%M:%S}",
          car.id, car.fuel, car.timestamp) };

      std::println("{}", formatted_string);
      std::println(inserted_cars, "{}", formatted_string),
          std::fflush(inserted_cars);
    }
    else
    {
      const auto formatted_string { std::format(
          "Машина не попала в очередь: номер - {}, тип топлива - {}, "
          "временная метка - {:%Y-%m-%d %H:%M:%S}",
          car.id, car.fuel, car.timestamp) };

      std::println("{}", formatted_string);
      std::println(dropped_cars, "{}", formatted_string),
          std::fflush(dropped_cars);
    }

    unlock();
  }

  std::vector<Car> cars {};

  bool finished { false };

  std::FILE* inserted_cars { std::fopen("inserted.log", "a") };

  std::FILE* dropped_cars { std::fopen("dropped.log", "a") };

  sem_t mutex {};

  static constexpr auto max_size { 15 };
}* queue { nullptr };

struct Generator
{
  explicit Generator() = default;

  void generate();

  std::thread thread {};

  static constexpr auto requests { 150 };

  static constexpr auto mean_generation_time { 1 };

  static constexpr auto standard_deviation { 0.5 };
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

std::array<sem_t, std::to_underlying(Fuel::COUNT)> fuel_semaphores;

void Generator::generate()
{
  thread = std::thread {
    [&] {
      std::normal_distribution<> distribution(mean_generation_time,
                                              standard_deviation);
      std::mt19937 number_generator(std::random_device {}());

      for (int request { 0 }; request < requests; ++request)
      {
        std::this_thread::sleep_for(
            std::chrono::duration<double>(distribution(number_generator)));

        const auto fuel { std::invoke([&] {
          std::uniform_int_distribution<> distribution(0, 4);
          const auto value { distribution(number_generator) };
          return value < 2 ? Fuel::AI76 : value < 4 ? Fuel::AI92 : Fuel::AI95;
        }) };

        queue->insert_car({
            .id = request,
            .fuel = fuel,
            .timestamp = std::chrono::system_clock::now(),
        });

        sem_post(&fuel_semaphores[std::to_underlying(fuel)]);
      }

      queue->finished = true;
    },
  };
}

void Column::serve(int index)
{
  thread = std::thread {
    [this, index] {
      std::normal_distribution<> distribution(mean_service_time,
                                              standard_deviation);
      std::mt19937 number_generator(std::random_device {}());

      const auto log { std::fopen(std::format("column_{}.log", index).data(),
                                  "a") };

      while (!queue->finished)
      {
        sem_wait(&fuel_semaphores[std::to_underlying(fuel)]);

        const auto car { queue->find_nearest_car(fuel) };

        if (!car) continue;

        const auto formatted_string { std::format(
            "Колонка {} начала обслуживание машины с номером {} и "
            "типом топлива {} во временной метке {:%Y-%m-%d "
            "%H:%M:%S}",
            index, car->id, car->fuel, car->timestamp) };

        std::println("{}", formatted_string);
        std::println(log, "{}", formatted_string), std::fflush(log);

        std::this_thread::sleep_for(
            std::chrono::duration<double>(distribution(number_generator)));
      }

      std::fclose(log);
    },
  };
}

void Configure(const std::string& path)
{
  nlohmann::json configuration {};
  std::ifstream { path } >> configuration;
  columns = configuration["columns"].get<std::array<Column, 5>>();
}

int main(int argc, char** argv)
{
  if (argc == 1)
  {
    std::print("Не передан путь до конфигурационного файла");
    return EXIT_FAILURE;
  }

  Configure(argv[1]);

  const auto shm_id { shmget(ftok(".1", 'S'), sizeof(Queue),
                             IPC_CREAT | 0666) };

  void* raw_memory { shmat(shm_id, nullptr, 0) };

  queue = new (raw_memory) Queue;

  for (auto& fuel_semaphore : fuel_semaphores) sem_init(&fuel_semaphore, 0, 0);

  Generator generator {};
  generator.generate();

  for (int index { 0 }; auto& column : columns) column.serve(++index);

  generator.thread.join();

  for (auto& column : columns) column.thread.join();

  shmdt(queue);

  shmctl(shm_id, IPC_RMID, nullptr);

  for (auto& fuel_semaphore : fuel_semaphores) sem_destroy(&fuel_semaphore);
}