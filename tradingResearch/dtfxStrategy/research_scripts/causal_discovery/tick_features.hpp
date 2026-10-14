#pragma once
// Input: seconds, ask, bid, available ask size, available bid size. These are
// quote observations, not executions. Output timestamps are minute CLOSE times.
py::array_t<double> tick_features(D ticks,double pip) {
    auto a=ticks.unchecked<2>();std::vector<double> rows;
    {py::gil_scoped_release unlock;
    for(int64_t i=0;i<a.shape(0);) {
        double bucket=std::floor(a(i,0)/60)*60;int64_t j=i;
        double first=(a(i,1)+a(i,2))/2,prev=first,low=first,high=first;
        double imbalance=0,micro=0,spread=0,size=0,variation=0,moves=0,direction=0;
        while(j<a.shape(0)&&a(j,0)<bucket+60) {
            double ask=a(j,1),bid=a(j,2),av=a(j,3),bv=a(j,4),mid=(ask+bid)/2;
            double total=av+bv,imb=total>0?(bv-av)/total:0,change=mid-prev;
            imbalance+=imb;micro+=imb*(ask-bid)/2/pip;spread+=(ask-bid)/pip;size+=total;
            if(change!=0){direction+=change>0?1:-1;moves++;}
            variation+=(change/pip)*(change/pip);low=std::min(low,mid);high=std::max(high,mid);prev=mid;j++;
        }
        double n=j-i;
        rows.insert(rows.end(),{bucket+60,n,moves?direction/moves:0,imbalance/n,micro/n,
            spread/n,(high-low)/pip,(prev-first)/pip,std::sqrt(variation),size/n,prev});
        i=j;
    }}
    py::array_t<double> result({py::ssize_t(rows.size()/11),py::ssize_t(11)});
    std::copy(rows.begin(),rows.end(),result.mutable_data());return result;
}
