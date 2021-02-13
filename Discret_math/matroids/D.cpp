#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <bitset>
#include <queue>
 
 
using namespace std;
 
bool cmp(pair<int, int> a, pair<int, int> b)
{
    return a.first > b.first;
}
 
int main() {
    freopen("cycles.in", "r+", stdin);
    freopen("cycles.out", "w+", stdout);
    int n, m;
    cin >> n >> m;
    vector<pair<int, int>> w;
    for (int i = 0, tmp_w; i < n; i++)
    {
        cin >> tmp_w;
        w.emplace_back(tmp_w, i);
    }
    sort(w.begin(), w.end(), cmp);
    unordered_set<bitset<21>> in_S;
    queue<bitset<21>> que;
    for (int i = 0, size; i < m; i++)
    {
        bitset<21> bits(0);
        cin >> size;
        for (int j = 0, bit; j < size; j++)
        {
            cin >> bit;
            bits[--bit] = true;
        }
        in_S.insert(bits);
        que.push(bits);
    }
    while (!que.empty())
    {
        auto tmp = que.front();
        que.pop();
        for (int i = 0; i < n; i++)
        {
            if (!tmp[i]) {
                tmp[i] = true;
                if (in_S.find(tmp) == in_S.end())
                {
                    in_S.insert(tmp);
                    que.push(tmp);
                }
                tmp[i] = false;
            }
        }
    }
    long long ans = 0;
    bitset<21> bits(0);
    for (auto now : w)
    {
        bits[now.second] = true;
        if (in_S.find(bits) == in_S.end())
        {
            ans += now.first;
        } else {
          bits[now.second] = false;
        }
    }
    cout << ans;
    return 0;
}
