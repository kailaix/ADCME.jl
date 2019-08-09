#include <string>
#include <fstream>
#include <iostream>
#include <string>
// modified from https://stackoverflow.com/questions/1090428/how-to-output-array-of-doubles-to-hard-drive

template<typename T>
bool saveArray( const T* pdata, size_t length, const std::string& file_path )
{
    std::ofstream os(file_path.c_str(), std::ios::binary | std::ios::out);
    if ( !os.is_open() )
        return false;
    os.write(reinterpret_cast<const char*>(pdata), std::streamsize(length*sizeof(T)));
    os.close();
    return true;
}

template<typename T>
bool loadArray( T* pdata, size_t length, const std::string& file_path)
{
    std::ifstream is(file_path.c_str(), std::ios::binary | std::ios::in);
    if ( !is.is_open() )
        return false;
    is.read(reinterpret_cast<char*>(pdata), std::streamsize(length*sizeof(T)));
    is.close();
    return true;
}