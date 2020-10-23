#include <thread>
#include <chrono>
#include <condition_variable>
#include <atomic>

void print_thread(std::atomic_int &cnt, std::condition_variable &cv){
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  printf("My thread ID is %d\n", std::this_thread::get_id());
  cnt++;
  if (cnt==7) cv.notify_one();
}

void threadpool_print(OpKernelContext* context){
  thread::ThreadPool * const tp = context->device()->tensorflow_cpu_worker_threads()->workers;
  std::atomic_int cnt = 0;
  std::condition_variable cv;
  std::mutex mu;

  printf("Number of intra-parallel thread = %d\n", tp->NumThreads());
  printf("Maximum Parallelism = %d\n", port::MaxParallelism());

  for (int i = 0; i < 7; i++)
    tp->Schedule([&cnt, &cv](){print_thread(cnt, cv);});
  
  {
    std::unique_lock<std::mutex> lck(mu);
    cv.wait(lck, [&cnt](){return cnt==7;});
  }
  printf("Op finished\n");
}