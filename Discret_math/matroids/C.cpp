#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <queue>
 
using namespace std;
 
bool cmp(pair<int, int> a, pair<int, int> b) {
    return a.first > b.first;
}
 
vector<vector<int>> e;
vector<int> matching_from_left;
vector<int> matching_from_right;
vector<int> been;
 
bool try_replace_match(int i) {
    if (been[i] != 0)
    {
        return false;
    } else {
        been[i] = 1;
    }
    for (auto now : e[i])
    {
        if (matching_from_right[now] == -1 || try_replace_match(matching_from_right[now])) {
            matching_from_left[i] = now;
            matching_from_right[now] = i;
            return true;
        }
    }
    return false;
}
 
void try_stand_matching(int i_r, int n) {
    been.clear();
    been.resize(n, 0);
    try_replace_match(i_r);
}
 
 
int main() {
    freopen("matching.in", "r+", stdin);
    freopen("matching.out", "w+", stdout);
    int n;
    cin >> n;
    vector<pair<int, int>> w;
    for (int i = 0, tmp_w; i < n; i++)
    {
        cin >> tmp_w;
        w.emplace_back(tmp_w, i);
    }
    e.resize(n);
    sort(w.begin(), w.end(), cmp);
 
    for (int i = 0, size; i < n; i++)
    {
        cin >> size;
        for (int j = 0, now; j < size; j++)
        {
            cin >> now;
            e[i].push_back(--now);
        }
    }
 
    matching_from_left.resize(n, -1);
    matching_from_right.resize(n, -1);
 
    for (auto now : w)
    {
        try_stand_matching(now.second, n);
    }
    for (auto now : matching_from_left)
    {
        if (now == -1) {
            cout << "0 ";
        } else {
            cout << ++now << " ";
        }
    }
    return 0;
}
