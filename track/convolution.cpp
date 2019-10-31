#include <vector>
#include <stdio>
#include <pair>
#include <string>
#include <cstdlib>
using namespace std;

vector<vector<unsigned char>> createMatrix() {
    int rows = -1;
    int cols = -1;
    while (!getline(cin, rows) || !getline(cin, cols)) {
        if (rows == 0 || cols == 0) {
            break;
        }
    }
    return vector<vector<unsigned char>>(rows, vector<unsigned char>(cols, rand() % 256))
}

int main() {
    for (vector<unsigned char> a : createMatrix()) {
        for (unsigned char c : a) {
            cout << c << end;
        }
        cout << endl;
    }
    return 0;
}
