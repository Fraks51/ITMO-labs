#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <queue>
 
using namespace std;
 
int n, m;
 
struct edge {
    int color;
    vector<int> v;
    edge(int v1, int v2, int color) : color(color)
    {
        v = {v1, v2};
    }
};
 
vector<edge> edges;
 
struct sub_graph {
    vector<vector<int>> e;
};
 
struct dsu {
    vector<int> p;
 
    dsu ()
    {
        p.resize(n);
        for (int i = 0; i < n; i++)
            p[i] = i;
    }
 
    int find(int i)
    {
        if (p[i] == i) {
            return i;
        }
        p[i] = find(p[i]);
        return p[i];
    }
 
    void union_dsu(int i, int j)
    {
        i = find(i);
        p[i] = find(j);
    }
};
 
sub_graph build_sub_graph(const set<int>& J, const set<int>& S)
{
    sub_graph sg;
    set<int> colors;
    sg.e.resize(m);
 
    // <rainbow>
    for (auto now : J)
    {
        colors.insert(edges[now].color);
    }
    for (int now : S)
    {
        int color = edges[now].color;
        if (colors.find(color) == colors.end())
        {
            for (auto now_J : J)
            {
                sg.e[now].push_back(now_J);
            }
        } else {
            for (auto now_J : J)
            {
                if (edges[now_J].color == color)
                {
                    sg.e[now].push_back(now_J);
                }
            }
        }
    }
    // </rainbow>
    // <non cycles>
    for (int i : J)
    {
        dsu d = dsu();
        for (int now : J)
        {
            if (now != i)
            {
                d.union_dsu(edges[now].v[0], edges[now].v[1]);
            }
        }
        for (int now : S)
        {
            if (d.find(edges[now].v[0]) != d.find(edges[now].v[1]))
            {
                sg.e[i].push_back(now);
            }
        }
    }
    // </non cycles>
    return sg;
}
 
set<int> build_X1(const set<int>& J, const set<int>& S, const sub_graph& sg)
{
    set<int> X1;
    dsu d = dsu();
    for (int now : J)
    {
        d.union_dsu(edges[now].v[0], edges[now].v[1]);
    }
    for (int now : S)
    {
        if (d.find(edges[now].v[0]) != d.find(edges[now].v[1]))
        {
            X1.insert(now);
        }
    }
    return X1;
}
 
set<int> build_X2(const set<int>& J, const set<int>& S, const sub_graph& sg)
{
    set<int> colors;
    set<int> X2;
    for (auto now : J)
    {
        colors.insert(edges[now].color);
    }
    for (int now : S)
    {
        int color = edges[now].color;
        if (colors.find(color) == colors.end())
        {
            X2.insert(now);
        }
    }
    return X2;
}
 
 
 
vector<int> build_P(const set<int>& X1, const set<int>& X2, const sub_graph& sg)
{
    vector<int> parent(m, -1);
    queue<int> q;
    for (auto now : X1)
    {
        q.push(now);
        parent[now] = now;
    }
    vector<int> P;
    int final = -1;
    while (!q.empty())
    {
        auto current = q.front();
        q.pop();
        if (X2.find(current) != X2.end())
        {
            final = current;
            break;
        }
        for (auto now : sg.e[current])
        {
            if (parent[now] == -1)
            {
                q.push(now);
                parent[now] = current;
            }
        }
    }
    if (final != -1)
    {
        while (true) {
            P.push_back(final);
            if (final == parent[final])
            {
                break;
            }
            final = parent[final];
        }
    }
    reverse(P.begin(), P.end());
    return P;
}
 
int main() {
 
#ifdef _DEBUG
#else
    freopen("rainbow.in", "r+", stdin);
    freopen("rainbow.out", "w+", stdout);
#endif
 
    cin >> n >> m;
    set<int> S;
    for (int i = 0, v1, v2, color; i < m ; i++)
    {
        S.insert(i);
        cin >> v1 >> v2 >> color;
        v1--;
        v2--;
        edges.emplace_back(v1, v2, color);
    }
    set<int> J; // non cycles - M1; rainbow - M2;
    bool is_max = false;
    while (! is_max)
    {
        auto D_J = build_sub_graph(J, S);
        auto X1 = build_X1(J, S, D_J);
        auto X2 = build_X2(J, S, D_J);
        auto P = build_P(X1, X2, D_J);
        if (P.empty())
        {
            is_max = true;
        } else {
            for (int i = 0; i < P.size(); i++)
            {
                if (i % 2 == 0)
                {
                    S.erase(P[i]);
                    J.insert(P[i]);
                } else {
                    J.erase(P[i]);
                    S.insert(P[i]);
                }
            }
        }
    }
    cout << J.size() << "\n";
    for (int i : J)
    {
        cout << ++i << " ";
    }
    return 0;
}
