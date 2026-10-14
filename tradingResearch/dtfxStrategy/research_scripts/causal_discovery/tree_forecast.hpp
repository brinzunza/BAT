#pragma once
#include <functional>
// Histogram gradient boosting with depth-three interactions, trained only on
// past closed labels. Query columns 0:7 (including future outcomes) are not read.
py::tuple learn_trees(D data,D queries,int rounds) {
    auto a=data.unchecked<2>();auto q=queries.unchecked<2>();int64_t n=a.shape(0),nf=a.shape(1)-7;
    if(n<1000)throw std::runtime_error("At least 1000 past outcomes required");
    constexpr int bins=16;std::vector<std::vector<double>> cuts(nf);std::vector<uint8_t> code(n*nf);
    std::vector<double> pred(n),out(q.shape(0)),residual(n),model;double mean=0;
    struct Node {int f=-1,left=-1,right=-1;double threshold=0,value=0;};std::vector<Node> nodes;
    std::vector<int64_t> roots;
    {py::gil_scoped_release unlock;
    for(int64_t i=0;i<n;i++)mean+=std::clamp(a(i,6),-5.,5.);mean/=n;
    std::fill(pred.begin(),pred.end(),mean);std::fill(out.begin(),out.end(),mean);
    for(int64_t f=0;f<nf;f++) {
        std::vector<double> values;values.reserve(n);for(int64_t i=0;i<n;i++)if(std::isfinite(a(i,f+7)))values.push_back(a(i,f+7));
        std::sort(values.begin(),values.end());for(int b=1;b<bins;b++)cuts[f].push_back(values.empty()?0:values[std::min(values.size()-1,values.size()*b/bins)]);
        for(int64_t i=0;i<n;i++){double x=std::isfinite(a(i,f+7))?a(i,f+7):0;code[i*nf+f]=std::upper_bound(cuts[f].begin(),cuts[f].end(),x)-cuts[f].begin();}
    }
    std::function<int(const std::vector<int64_t>&,int)> grow;
    grow=[&](const std::vector<int64_t>& ids,int depth) {
        int id=nodes.size();nodes.push_back(Node{});double total=0;for(auto i:ids)total+=residual[i];
        nodes[id].value=.05*total/(ids.size()+20.);
        int minleaf=std::max(int64_t(100),n/200);if(depth==3||ids.size()<size_t(2*minleaf))return id;
        std::vector<double> sums(nf*bins,0);std::vector<int> counts(nf*bins,0);
        for(auto i:ids)for(int f=0;f<nf;f++){int k=f*bins+code[i*nf+f];sums[k]+=residual[i];counts[k]++;}
        double best=1e-8,threshold=0;int feature=-1,bestbin=0;
        for(int f=0;f<nf;f++){double left=0;int count=0;for(int b=0;b<bins-1;b++) {
            left+=sums[f*bins+b];count+=counts[f*bins+b];int rightn=ids.size()-count;
            if(count<minleaf||rightn<minleaf)continue;
            double right=total-left,gain=left*left/(count+20.)+right*right/(rightn+20.)-total*total/(ids.size()+20.);
            if(gain>best){best=gain;feature=f;bestbin=b;threshold=cuts[f][b];}
        }}
        if(feature<0)return id;
        std::vector<int64_t> left,right;for(auto i:ids)(code[i*nf+feature]<=bestbin?left:right).push_back(i);
        int l=grow(left,depth+1),r=grow(right,depth+1);nodes[id].f=feature+7;nodes[id].threshold=threshold;nodes[id].left=l;nodes[id].right=r;return id;
    };
    for(int iteration=0;iteration<rounds;iteration++) {
        std::vector<int64_t> ids(n);for(int64_t i=0;i<n;i++){ids[i]=i;residual[i]=std::clamp(a(i,6),-5.,5.)-pred[i];}
        int root=grow(ids,0);roots.push_back(root);
        for(int64_t i=0;i<n;i++){int k=root;while(nodes[k].f>=0){double x=a(i,nodes[k].f);if(!std::isfinite(x))x=0;k=x<nodes[k].threshold?nodes[k].left:nodes[k].right;}pred[i]+=nodes[k].value;}
        for(int64_t i=0;i<q.shape(0);i++){int k=root;while(nodes[k].f>=0){double x=q(i,nodes[k].f);if(!std::isfinite(x))x=0;k=x<nodes[k].threshold?nodes[k].left:nodes[k].right;}out[i]+=nodes[k].value;}
    }
    for(auto node:nodes)model.insert(model.end(),{double(node.f),node.threshold,double(node.left),double(node.right),node.value});
    }
    return py::make_tuple(mean,arr(roots),arr(model),arr(out));
}
