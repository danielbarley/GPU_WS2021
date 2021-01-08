#include <iostream>
#include <string>
#include <vector>

#include <chTimer.hpp>

using namespace std;

void printHelp(char const* argv)
{
	std::cout << "Help:" << std::endl
			  << "  Usage: " << std::endl
			  << "  " << argv << " [<num-elements>]" << std::endl;
}

float reduction(std::vector<float> vec)
{
    float sum = 0;
    for (auto element: vec)
    {
        sum += element;
    }
    return sum;
}

int main(int argc, char** argv)
{
    if (argc != 2)
        printHelp(argv[0]);

    long N = stol(argv[1]);
    
    vector<float> vec(N, 1);

    ChTimer reductionTimer;
    reductionTimer.start();
    float result = reduction(vec);
    reductionTimer.stop();

    if (result != N)
    {
        std::cout << "Wrong reduction result" << std::endl;
        return -1;
    }

    std::cout << "reduction took " << reductionTimer.getTime() * 1e3 << " ms" << std::endl;
    std::cout << "reduction bandwidth: " << reductionTimer.getBandwidth(N * sizeof(float)) * 1e-9 << " GB / s" << std::endl;
}
