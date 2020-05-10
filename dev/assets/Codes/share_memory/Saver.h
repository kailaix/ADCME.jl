#include <map>
#include <string>
#include <vector>

struct DataStore
{        
    std::map<std::string, std::vector<double>> vdata;
};
extern DataStore ds;

