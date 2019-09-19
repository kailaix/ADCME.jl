#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
#include <map>
using namespace std::chrono;

class Timer{
private:
  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
public:
  Timer(){
    start = high_resolution_clock::now();
    // printf("New timer\n");
  };
  void Set(){
    start = high_resolution_clock::now();
    // printf("Set\n");
  }
  double Get(){
    end = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(end - start);
    // printf("Get\n");
    return time_span.count();
  }
};

std::map<int32,Timer*> TimerMaps;


