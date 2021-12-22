# include <mpi.h>
# include <iostream>
# include <vector>
# include <random>
# include <chrono>
using namespace std::chrono;

int main(int argc, char *argv[])
{
std::size_t N = 1'073'741'824;
std::vector<double> numbers;
numbers.resize(N);

int master = 0;
int my_id;
int numprocs;
MPI_Status status;

//  Initialize MPI.
int ierr = MPI_Init ( &argc, &argv );
if ( ierr != 0 )
    throw std::runtime_error("Fatal error in MPI initialization");

//  Get the number of processes.
MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

//  Determine the rank of this process.
MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

high_resolution_clock::time_point start_time;
high_resolution_clock::time_point end_time;
std::uint64_t ns;

//  The master process generating the vector
if (my_id == master)
{
    std::random_device rd;
    static std::mt19937 rng(rd());
    std::uniform_real_distribution<double> unif(1., 1000.);

    for ( std::size_t i = 0; i < N; ++i )
    {
        numbers[i] = unif(rng);
    }

    // start measuring time
    start_time = high_resolution_clock::now();
}


// The master process broadcasts the computed initial values to all the other processes.
MPI_Bcast(numbers.data(), numbers.size(), MPI_DOUBLE, master, MPI_COMM_WORLD);

//  Each process adds up its entries.
double sum = 0.0;

for (std::size_t i = my_id; i < N; i += numprocs)
    sum = sum + numbers[i];

double sum_all, average;

// Each worker process sends its sum back to the master process.
if (my_id != master)
{
    MPI_Send(&sum, 1, MPI_DOUBLE, master, 1, MPI_COMM_WORLD);
}
else
{
    // master proccess receive and sum up all sums from other processes
    sum_all = sum;
    for (int i = 1; i < numprocs; ++i)
    {
        MPI_Recv (&sum, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        sum_all = sum_all + sum;
    }
    average = sum_all / numbers.size();

    // end measuring time
    end_time = high_resolution_clock::now();

    ns = static_cast<std::uint64_t>(duration_cast<nanoseconds>(end_time - start_time).count());
}

if (my_id == master)
{
    std::cout << "Master process. Total sum is " << sum_all << ". Average is " << average << std::endl;
    std::cout << "We had " << numprocs << " MPI tasks. ns: " << ns << std::endl;
}

//Terminate MPI.
MPI_Finalize( );

return 0;
}

