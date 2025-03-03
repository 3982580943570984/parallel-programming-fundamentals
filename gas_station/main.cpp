#include <semaphore.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <fstream>
#include <functional>
#include <print>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

using time_point = std::chrono::system_clock::time_point;

template <typename T>
class SharedMemory
{
 public:
  ~SharedMemory()
  {
    shmdt(object);
    shmctl(identity, IPC_RMID, nullptr);
  }

  [[nodiscard]] T* operator->() const { return object; }

  [[nodiscard]] T& operator*() const { return *object; }

 private:
  int identity { shmget(IPC_PRIVATE, sizeof(T), IPC_CREAT | 0666) };

  T* object { new (shmat(identity, nullptr, 0)) T {} };
};

enum class Fuel
{
  AI76,
  AI92,
  AI95,
  COUNT,
};

inline static void from_json(const nlohmann::json& j, Fuel& fuel)
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

struct Column
{
  void serve(int) const;

  Fuel fuel {};

  double mean_service_time {};

  double standard_deviation {};

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
  explicit Queue()
  {
    sem_init(&mutex, 1, 1);

    std::ranges::for_each(fuel_semaphores,
                          [](auto& semaphore) { sem_init(&semaphore, 1, 0); });
  };

  ~Queue()
  {
    sem_destroy(&mutex);

    std::ranges::for_each(fuel_semaphores,
                          [](auto& semaphore) { sem_destroy(&semaphore); });

    std::fclose(inserted);
    std::fclose(dropped);
  }

  auto lock() { sem_wait(&mutex); }

  auto unlock() { sem_post(&mutex); }

  [[nodiscard]] std::optional<Car> find_nearest_car(const Fuel& fuel)
  {
    std::optional<Car> result {};

    lock();

    auto it = std::find_if(cars.begin(), cars.begin() + current_size,
                           [&](const auto& c) { return c.fuel == fuel; });

    if (it != cars.begin() + current_size)
    {
      result = *it;

      std::move(it + 1, cars.begin() + current_size, it);
      current_size--;
    }

    unlock();

    return result;
  }

  auto insert_car(const Car& car)
  {
    lock();

    if (current_size < max_size)
    {
      cars[current_size++] = car;

      const auto formatted_string { std::format(
          "Машина попала в очередь: номер - {}, тип топлива - {}, "
          "временная метка - {:%Y-%m-%d %H:%M:%S}",
          car.id, car.fuel, car.timestamp) };

      std::println("{}", formatted_string);
      std::println(inserted, "{}", formatted_string), std::fflush(inserted);
    }
    else
    {
      const auto formatted_string { std::format(
          "Машина не попала в очередь: номер - {}, тип топлива - {}, "
          "временная метка - {:%Y-%m-%d %H:%M:%S}",
          car.id, car.fuel, car.timestamp) };

      std::println("{}", formatted_string);
      std::println(dropped, "{}", formatted_string), std::fflush(dropped);
    }

    unlock();
  }

  static constexpr auto max_size { 15 };

  std::array<Car, max_size> cars {};

  std::size_t current_size {};

  std::array<sem_t, std::to_underlying(Fuel::COUNT)> fuel_semaphores {};

  sem_t mutex {};

  bool finished { false };

  std::FILE* inserted { std::fopen("inserted.log", "w") };

  std::FILE* dropped { std::fopen("dropped.log", "w") };
};

struct Generator
{
  static void generate();

  static inline auto requests { 150 };

  static inline auto mean_generation_time { 1 };

  static inline auto standard_deviation { 0.5 };
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Generator, requests, mean_generation_time, standard_deviation);

auto const queue { SharedMemory<Queue> {} };

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

void Generator::generate()
{
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

    sem_post(&queue->fuel_semaphores[std::to_underlying(fuel)]);
  }

  queue->finished = true;
}

void Column::serve(int index) const
{
  std::normal_distribution<> distribution(mean_service_time,
                                          standard_deviation);
  std::mt19937 number_generator(std::random_device {}());

  const auto log { std::fopen(std::format("column_{}.log", index).data(),
                              "w") };

  while (!queue->finished)
  {
    sem_wait(&queue->fuel_semaphores[std::to_underlying(fuel)]);

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

  while (true)
  {
    queue->lock();
    bool has_car = std::any_of(queue->cars.begin(), queue->cars.begin() + queue->current_size,
                               [&](const Car& c) { return c.fuel == fuel; });
    queue->unlock();

    if (!has_car)
      break;

    auto car = queue->find_nearest_car(fuel);
    if (car)
    {
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
    else
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  std::fclose(log);
}

int main(int argc, char** argv)
{
  nlohmann::json configuration {};
  std::ifstream { argv[1] } >> configuration;
  columns = configuration["columns"].get<std::array<Column, 5>>();

  configuration["generator"].get<Generator>();

  std::vector<pid_t> pids;

  for (int index {}; auto column : columns)
  {
    ++index;

    pids.emplace_back(fork());
    if (pids.back() != 0) continue;

    column.serve(index);
    return 0;
  }

  Generator::generate();

  for (const auto& column : columns)
    sem_post(&queue->fuel_semaphores[std::to_underlying(column.fuel)]);

  for (auto const& pid : pids) waitpid(pid, nullptr, 0);
}