#include <mpi.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <utility>
#pragma warning(disable : 4996)

using namespace std;

#define SIZE 6
#define DIMS_DG 2
#define DIMS_SFG 1

int Receive(int source);
void Send(int destination, int X);
void Error(const char* message);
int* Sort_DG(int X[SIZE], int rank, MPI_Comm comm);
int* Sort_SFG(int X[SIZE], int rank, MPI_Comm comm);

int main(int argc, char* argv[])
{
  int rank;
  int size;
  MPI_Comm comm;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool is_DG = true;
  if (size <= SIZE)
  {
    is_DG = false;
  }

  int X[SIZE] = { 89, 206, 73, 3, 19, 55 };
  int* result;

  if (is_DG)
  {
    int dims[DIMS_DG];
    int periods[DIMS_DG];

    for (int i = 0; i < DIMS_DG; i++)
    {
      dims[i] = SIZE;
      periods[i] = 0;
    }

    MPI_Cart_create(MPI_COMM_WORLD, DIMS_DG, dims, periods, 1, &comm);

    if (rank == 0)
    {
      printf("GRAF ZAVISIMOSTEY...\n");
    }

    result = Sort_DG(X, rank, comm);

    if (rank == 0)
    {
      printf("Result sequence:\n");
      for (int i = 0; i < SIZE; ++i)
      {
        printf("%d\t", result[i]);
      }
      printf("\n");
    }
  }
  else
  {
    int dims[DIMS_SFG];
    int periods[DIMS_SFG];

    for (int i = 0; i < DIMS_SFG; i++)
    {
      dims[i] = SIZE;
      periods[i] = 0;
    }

    MPI_Cart_create(MPI_COMM_WORLD, DIMS_SFG, dims, periods, 1, &comm);

    if (rank == 0)
    {
      printf("Graph potoka signalov...\n");
    }

    result = Sort_SFG(X, rank, comm);

    if (rank == 0)
    {
      printf("Result sequence:\n");
      for (int i = 0; i < SIZE; ++i)
      {
        printf("%d\t", result[i]);
      }
      printf("\n");
    }
  }

  free(result);

  MPI_Finalize();
  return 0;
}

int* Sort_DG(int X[SIZE], int rank, MPI_Comm comm)
{
  int min, max;
  int coords[2];

  MPI_Cart_coords(comm, rank, DIMS_DG, coords);

  if (coords[0] <= coords[1])
  {
    int x_source, x_dest, y_source, y_dest;

    MPI_Cart_shift(comm, 0, 1, &x_source, &x_dest);
    MPI_Cart_shift(comm, 1, 1, &y_source, &y_dest);

    min = coords[0] == coords[1] ? INT_MAX : Receive(y_source);

    if (coords[0] == 0)
    {
      if (coords[1] == 0)
      {
        max = X[0];
        for (int i = 1; i < SIZE; ++i) Send(i, X[i]);
      }
      else
      {
        max = Receive(0);
      }
    }
    else
    {
      max = Receive(x_source);
    }

    if (min > max)
    {
      int temp = min;
      min = max;
      max = temp;
    }

    if (y_dest < 0)
    {
      Send(0, min);
    }
    else
    {
      Send(y_dest, min);
    }

    if (coords[0] != coords[1]) Send(x_dest, max);

    if (rank == 0)
    {
      int* result = (int*)malloc(sizeof(int) * SIZE);
      int z = 0;

      for (int i = 0; i < SIZE; ++i)
      {
        if (i == 0)
          z += SIZE - 1;
        else
          z += SIZE;

        int k = Receive(z);
        result[i] = k;
      }
      return result;
    }
  }

  return NULL;
}

int* Sort_SFG(int X[SIZE], int rank, MPI_Comm comm)
{
  int min = INT_MAX;
  int previous {}, next {};

  MPI_Cart_shift(comm, 0, 1, &previous, &next);

  for (int i = 0; i < SIZE; ++i)
  {
    int max = rank == 0 ? X[i] : Receive(previous);

    if (min > max) std::swap(min, max);

    if (next > 0) Send(next, max);
  }

  int* result = rank == 0 ? (int*)malloc(sizeof(int) * SIZE) : NULL;

  MPI_Gather(&min, 1, MPI_INT, result, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return result;
}

int Receive(int source)
{
  int value;
  MPI_Status status;
  MPI_Recv(&value, 1, MPI_INT, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  return value;
}

void Send(int destination, int X)
{
  MPI_Send(&X, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
}

void Error(const char* message)
{
  puts(message);
  MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
  exit(0);
}
