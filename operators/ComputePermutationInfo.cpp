//
// Created by pal on 18-3-29.
//
#include <unordered_map>
#include <vector>

void computeIdxsMap(
    int * voxel_idxs,
    int pn,
    std::vector<int>& origin2permutation_idxs,
    std::vector<int>& voxel_idxs_lens
)
{
    std::unordered_map<unsigned long long, int> map;
    int voxel_num=0;
    std::vector<int> ds_gidxs(pn);
    for(int i=0;i<pn;i++)
    {
        unsigned long long x=voxel_idxs[i*3];
        unsigned long long y=voxel_idxs[i*3+1];
        unsigned long long z=voxel_idxs[i*3+2];
        unsigned long long idx=0;
        idx=(idx|x)|(y<<16|z<<32);

        auto it=map.find(idx);
        if(it!=map.end())
        {
            ds_gidxs[i]=it->second;
        }
        else
        {
            map[idx]=voxel_num;
            ds_gidxs[i]=voxel_num;
            voxel_num++;
        }
    }

    std::vector<std::vector<int>> list_idxs(voxel_num);
    for(int i=0;i<pn;i++)
        list_idxs[ds_gidxs[i]].push_back(i);

    origin2permutation_idxs.resize(pn);
    voxel_idxs_lens.resize(voxel_num);
    int* o2p_it=origin2permutation_idxs.begin().base();
    int cur_len=0;
    for(int i=0;i<voxel_num;i++)
    {
        int len=list_idxs[i].size();
        std::copy(list_idxs[i].begin(),list_idxs[i].end(),o2p_it);
        o2p_it+=len;
        voxel_idxs_lens[i]=len;
        cur_len+=len;
    }
}