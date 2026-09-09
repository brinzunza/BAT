#pragma once
// Synthetic regime evidence from resolved shadow opportunities only. An exit at
// the entry timestamp is excluded, avoiding ambiguous same-time event ordering.
py::array_t<double> resolved_features(D data,double memory) {
    auto a=data.unchecked<2>();const auto n=a.shape(0),cols=a.shape(1);
    py::array_t<double> result({n,cols+4});auto out=result.mutable_unchecked<2>();
    std::vector<int64_t> order(n);std::iota(order.begin(),order.end(),0);
    double mean[8]={},second[8]={},weight[8]={},count[8]={};
    const double decay=std::exp(-1./memory);int64_t p=0;
    py::gil_scoped_release unlock;
    std::stable_sort(order.begin(),order.end(),[&](int64_t i,int64_t j){return a(i,1)<a(j,1);});
    for(int64_t i=0;i<n;i++) {
        while(p<n&&a(order[p],1)<a(i,0)) {
            int64_t j=order[p++];int k=int(a(j,2))*2+int(a(j,3));
            if(a(j,0)>=a(j,1)||!std::isfinite(a(j,6)))continue;
            double v=a(j,6);mean[k]=decay*mean[k]+v;
            second[k]=decay*second[k]+v*v;weight[k]=decay*weight[k]+1;count[k]++;
        }
        for(int64_t f=0;f<cols;f++)out(i,f)=a(i,f);
        int k=int(a(i,2))*2+int(a(i,3));double w=weight[k];
        double m=w?mean[k]/w:0,sd=w?std::sqrt(std::max(0.,second[k]/w-m*m)):0;
        out(i,cols)=m;out(i,cols+1)=sd;out(i,cols+2)=w;out(i,cols+3)=count[k];
    }
    return result;
}
