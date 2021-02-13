#include <iostream>
#include <vector>
#include <algorithm>
 
using namespace std;
 
vector<int> tree;
 
bool cmp(pair<int, int> a, pair<int, int> b)
{
    return a.second > b.second;
}
 
int find(int i)
{
    if (i == -1) {
        return i;
    }
    if (tree[i] == i) {
        return i;
    }
    tree[i] = find(tree[i]);
    return tree[i];
}
 
void union_tree(int l, int r) {
    int i = find(r);
    tree[i] = find(l);
}
 
int main() {
    freopen("schedule.in", "r+", stdin);
    freopen("schedule.out", "w+", stdout);
    int n;
    cin >> n;
    vector<pair<int, int>> tasks; // first == 0 + ban
    for (int i = 0; i < n; i++)
    {
        pair<int,int> tmp;
        cin >> tmp.first >> tmp.second;
        tmp.first = min(tmp.first, 200000);
        tasks.push_back(tmp);
    }
    sort(tasks.begin(), tasks.end(), cmp);
    long long sum = 0;
    for (int i = 0; i < 200001; i++)
    {
        tree.push_back(i);
    }
    for (auto t : tasks)
    {
        int insert = find(t.first - 1);
        if (insert != -1)
        {
            union_tree(insert - 1, insert);
        } else {
            sum += t.second;
        }
    }
    cout << sum;
    return 0;
}
