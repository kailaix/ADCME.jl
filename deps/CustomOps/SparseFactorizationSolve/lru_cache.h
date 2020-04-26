
// Copyright (c) 2014 Robert Davis
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
using namespace tensorflow;


#include <memory>
#include <utility>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <vector>
#include <iostream>
// #include <chrono> s
// using namespace std::chrono; 

using namespace std;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

template<typename T, typename U>
class LRUCache {
  public:
    explicit LRUCache(int max_size);
  
    void put(T key, U value);

    std::pair<U, bool> retrieve(T key);

    bool remove(T key);

    void evict_all();

    void print();

    int size() const;

    int max_size() const;

    int hit_count() const;

    int miss_count() const;

    void resize(int n); 

    int get_new_id();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    LRUCache(const LRUCache&);
    void operator=(const LRUCache&);
};

#include <iostream>
#include <mutex>
#include <unordered_map>

template<typename T, typename U>
struct LRUCache<T, U>::Impl {
    class Node;

    using map_type = std::unordered_map<T, Node*>;

    struct Node {
        Node(U data) : data(data), prev(NULL), next(NULL) {}

        U data;
        Node* prev;
        Node* next;
        typename map_type::iterator map_it;
    };

    Impl(int max_size) : max_size_(max_size), size_(0),
                         list_head_(NULL), list_tail_(NULL),
                         hit_count_(0), miss_count_(0) {
    }

    void put(T key, U value);

    std::pair<U, bool> retrieve(T key);

    bool remove(T key);

    void evict_all();

    void print();

    int size() const;

    int max_size() const;

    int hit_count() const;

    int miss_count() const;

    int get_new_id();

    void bump_to_front(Node* node);

    map_type hashmap_;
    Node* list_head_;
    Node* list_tail_;
    int size_;
    int max_size_;

    int hit_count_;
    int miss_count_;

    int count_;

    std::mutex mutex_;
};

extern LRUCache<int, Eigen::SparseLU<SpMat>*> cache1;
extern LRUCache<int, Eigen::SparseLU<SpMat>*> cache2;
extern LRUCache<int, SpMat*> SpMat_cache;