#include <iostream>
#include <vector>
#include <algorithm>
 
using namespace std;
 
vector<int> tree;
 
int find(int i)
{
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
 
struct edge {
    int num;
    long long w;
    pair<int, int> v;
};
 
bool cmp(edge a, edge b)
{
    return a.w > b.w;
}
 
int main() {
    freopen("destroy.in", "r+", stdin);
    freopen("destroy.out", "w+", stdout);
    int n, m;
    long long s;
    cin >> n >> m >> s;
    vector<edge> tasks; // first == 0 + ban
    for (int i = 1; i <= m; i++)
    {
        edge tmp;
        tmp.num = i;
        cin >> tmp.v.first >> tmp.v.second >> tmp.w;
        tasks.push_back(tmp);
    }
    sort(tasks.begin(), tasks.end(), cmp);
    vector<edge> rest_edges;
    for (int i = 0; i < n + 1; i++)
    {
        tree.push_back(i);
    }
    for (auto a : tasks)
    {
        if (find(a.v.first) != find(a.v.second))
        {
            union_tree(a.v.first, a.v.second);
        } else {
            rest_edges.push_back(a);
        }
    }
    reverse(rest_edges.begin(), rest_edges.end());
    vector<int> ans;
    long long sum = 0;
    for (auto i : rest_edges)
    {
        if (sum + i.w > s) {
            break;
        }
        sum += i.w;
        ans.push_back(i.num);
    }
    sort(ans.begin(), ans.end());
    cout << ans.size() << endl;
    for (auto now : ans)
    {
        cout << now << " ";
    }
    return 0;
}
